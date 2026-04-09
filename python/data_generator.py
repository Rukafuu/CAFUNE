"""
data_generator.py — Gerador de Dataset via Gemini + Web Grounding

O Gemini usa Google Search para buscar contexto real e gera pares
(prompt, resposta ideal) que são salvos em bercario_data.jsonl.
O Curriculum Scheduler do CAFUNE consome esses pares automaticamente.

Fluxo:
    Tópico → Gemini (busca web) → par {prompt, target, intent, source_url}
           → bercario_data.jsonl → Curriculum Scheduler → Julia treino

Execução contínua: gera N pares por ciclo, pausa e repete.
    python python/data_generator.py
"""

import os
import sys
import io
import json
import time
import random
import logging
from dotenv import load_dotenv

# Força stdout UTF-8 no Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Env ───────────────────────────────────────────────────────────────────────
# Tenta múltiplos caminhos para encontrar o .env
for _candidate in [
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")),
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".env")),
    os.path.normpath(os.path.join(os.path.dirname(__file__), ".env")),
]:
    if os.path.exists(_candidate):
        load_dotenv(_candidate)
        break

BERCARIO_FILE = os.path.normpath(os.path.join(os.path.dirname(__file__), "bercario_data.jsonl"))
PAIRS_PER_CYCLE = 10     # pares gerados por ciclo
CYCLE_SLEEP     = 60     # segundos entre ciclos (1 min)
MAX_ENTRIES     = 3000   # não deixa o arquivo crescer infinitamente

# ── Tópicos para diversificar o dataset ───────────────────────────────────────
TOPICS = [
    # Conversação natural PT-BR
    "como fazer amigos sendo introvertido",
    "lidar com ansiedade no trabalho",
    "dicas para estudar com foco",
    "como pedir desculpas de forma sincera",
    "como manter uma conversa interessante",
    # Cultura e curiosidades
    "curiosidades sobre o universo",
    "história da internet em resumo",
    "como funciona a memória humana",
    "por que sonhamos",
    "o que é inteligência artificial",
    # Habilidades práticas
    "como aprender a cozinhar do zero",
    "dicas para economizar dinheiro",
    "como montar uma rotina produtiva",
    "como lidar com críticas no trabalho",
    "técnicas de respiração para acalmar",
    # Filosofia e emoções
    "o que é felicidade segundo a filosofia",
    "como superar um término de relacionamento",
    "significado de empatia na prática",
    "por que pessoas mentem",
    "como encontrar propósito de vida",
]

INTENT_MAP = {
    "amigos": "social",
    "ansiedade": "emotional_support",
    "estudar": "productivity",
    "desculpas": "social",
    "conversa": "social",
    "universo": "curiosity",
    "internet": "curiosity",
    "memória": "curiosity",
    "sonhamos": "curiosity",
    "inteligência": "curiosity",
    "cozinhar": "practical",
    "economizar": "practical",
    "rotina": "productivity",
    "críticas": "emotional_support",
    "respiração": "emotional_support",
    "felicidade": "philosophy",
    "término": "emotional_support",
    "empatia": "philosophy",
    "mentem": "philosophy",
    "propósito": "philosophy",
}


def get_intent(topic: str) -> str:
    for kw, intent in INTENT_MAP.items():
        if kw in topic.lower():
            return intent
    return "general"


# ── Gemini client ─────────────────────────────────────────────────────────────

def build_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY não encontrada.")
        return None
    try:
        from google import genai as google_genai
        from google.genai import types
        client = google_genai.Client(api_key=api_key)
        logger.info("[OK] Gemini client (google.genai) pronto.")
        return client, types
    except ImportError:
        logger.error("Instale google-genai: pip install google-genai")
        return None


def generate_pair(client, types, topic: str) -> dict | None:
    """
    Pede ao Gemini (com Google Search Grounding) para gerar um par
    conversacional realista sobre o tópico dado.
    Retorna dict {prompt, target, intent, topic, grounded} ou None.
    """
    system_prompt = f"""Você é um gerador de dados de treino para uma IA conversacional em português brasileiro chamada CAFUNE.

Sua tarefa: dado o tópico abaixo, crie UM par de conversa natural.
Use a web para buscar contexto real e factual quando relevante.

Tópico: "{topic}"

Regras:
- O "prompt" deve ser uma frase curta e natural que um usuário real enviaria (1-2 frases)
- O "target" deve ser uma resposta empática, informativa e natural da CAFUNE (2-4 frases)
- Escreva APENAS um JSON válido, sem markdown, sem explicações extras
- Formato exato:
{{"prompt": "...", "target": "..."}}"""

    try:
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.9,
        )
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=system_prompt,
            config=config,
        )
        text = resp.text.strip()

        # Extrai JSON da resposta (Gemini às vezes adiciona ```json)
        import re
        json_match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if not json_match:
            logger.warning("Resposta sem JSON válido: %s", text[:100])
            return None

        data = json.loads(json_match.group(0))
        prompt_text = data.get("prompt", "").strip()
        target_text = data.get("target", "").strip()

        if not prompt_text or not target_text:
            return None

        # Verifica se houve grounding (search usado)
        grounded = False
        try:
            grounding = resp.candidates[0].grounding_metadata
            grounded  = grounding is not None and len(getattr(grounding, "search_entry_point", None) or "") > 0
        except Exception:
            pass

        return {
            "prompt":   prompt_text,
            "target":   target_text,
            "intent":   get_intent(topic),
            "topic":    topic,
            "grounded": grounded,
            "source":   "gemini-web",
        }

    except Exception as e:
        logger.error("Erro ao gerar par para tópico '%s': %s", topic, e)
        return None


# ── Persistência ──────────────────────────────────────────────────────────────

def load_existing_prompts() -> set:
    """Carrega prompts existentes para deduplicar."""
    seen = set()
    if not os.path.exists(BERCARIO_FILE):
        return seen
    with open(BERCARIO_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                seen.add(entry.get("prompt", ""))
            except json.JSONDecodeError:
                pass
    return seen


def count_entries() -> int:
    if not os.path.exists(BERCARIO_FILE):
        return 0
    with open(BERCARIO_FILE, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def append_entry(entry: dict):
    with open(BERCARIO_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Loop principal ────────────────────────────────────────────────────────────

def run_generator():
    result = build_client()
    if result is None:
        return
    client, types = result

    logger.info("=== [DATA GENERATOR: GEMINI + WEB GROUNDING] ===")
    logger.info("Gerando %d pares por ciclo | salvando em %s",
                PAIRS_PER_CYCLE, os.path.basename(BERCARIO_FILE))

    cycle = 0
    while True:
        cycle += 1
        total = count_entries()
        if total >= MAX_ENTRIES:
            logger.info("Dataset cheio (%d entradas). Aguardando %ds...", total, CYCLE_SLEEP)
            time.sleep(CYCLE_SLEEP)
            continue

        logger.info("--- Ciclo %d | Dataset: %d/%d entradas ---", cycle, total, MAX_ENTRIES)
        seen_prompts = load_existing_prompts()

        topics_this_cycle = random.sample(TOPICS, min(PAIRS_PER_CYCLE, len(TOPICS)))
        added = 0

        for topic in topics_this_cycle:
            entry = generate_pair(client, types, topic)
            if entry is None:
                continue
            if entry["prompt"] in seen_prompts:
                logger.info("  [skip] Duplicata: %s", entry["prompt"][:50])
                continue

            append_entry(entry)
            seen_prompts.add(entry["prompt"])
            added += 1
            grounded_tag = "[WEB]" if entry["grounded"] else "[local]"
            logger.info("  %s +1 par | intent=%s | prompt: %s",
                        grounded_tag, entry["intent"], entry["prompt"][:60])

            time.sleep(2)  # Respeita rate limit da API

        logger.info("Ciclo %d concluido: %d/%d pares adicionados | proximo em %ds",
                    cycle, added, len(topics_this_cycle), CYCLE_SLEEP)
        time.sleep(CYCLE_SLEEP)


if __name__ == "__main__":
    run_generator()
