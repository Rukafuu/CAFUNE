"""
rlaif_scheduler.py — Curriculum RLAIF Scheduler para CAFUNE

Envia prompts ao motor Julia via mmap, selecionando a dificuldade
de acordo com a loss atual do modelo:

    loss > 2.5  → tier EASY   (frases curtas, saudações simples)
    1.5–2.5     → tier MEDIUM (perguntas estruturadas, ~média complexidade)
    < 1.5       → tier ANY    (inclui sycophancy traps para testar robustez)

Fontes de dados (normalizadas para schema unificado):
    - python/social_data.json        ({user, response} → 300 pares)
    - python/bercario_data.jsonl     ({intent, prompt, target} → 13 entradas)

Offsets mmap lidos:
    32–39  float64  loss atual (escrito pelo Julia)
    0      uint8    CmdID: 0x00=idle, 0x01=request
"""

import mmap
import time
import os
import json
import random
import struct
from filelock import FileLock, Timeout

# ── Paths ────────────────────────────────────────────────────────────────────
MEM_FILE       = os.path.normpath(os.path.join(os.path.dirname(__file__), "cafune_brain.mem"))
LOCK_FILE      = MEM_FILE + ".lock"
SOCIAL_FILE    = os.path.normpath(os.path.join(os.path.dirname(__file__), "social_data.json"))
BERCARIO_FILE  = os.path.normpath(os.path.join(os.path.dirname(__file__), "bercario_data.jsonl"))

LOSS_OFFSET    = 32   # float64 — loss atual do modelo
LOSS_EASY_THR  = 2.5  # acima → enviar prompts fáceis
LOSS_MEDIUM_THR = 1.5 # entre → médio; abaixo → qualquer (inclui traps)


# ── Carregamento e normalização dos datasets ──────────────────────────────────

def _classify_difficulty(prompt: str, intent: str = "") -> str:
    """
    Classifica um prompt em easy / medium / hard baseado em heurísticas.

    - easy:   ≤ 30 chars  ou  intent in {greeting, name, basic}
    - hard:   intent contém 'sycophancy' ou 'trap', ou prompt tem '?'
              com subordinadas complexas (> 80 chars)
    - medium: tudo o mais
    """
    easy_intents = {"greeting", "name", "basic", "saudacao", "apresentacao"}
    hard_intents = {"sycophancy", "trap", "adversarial", "hard"}

    intent_lower = intent.lower()
    if any(k in intent_lower for k in hard_intents):
        return "hard"
    if any(k in intent_lower for k in easy_intents) or len(prompt) <= 30:
        return "easy"
    if len(prompt) > 80:
        return "hard"
    return "medium"


def load_all_datasets() -> dict:
    """
    Carrega social_data.json e bercario_data.jsonl, normaliza para:
        {"prompt": str, "intent": str, "difficulty": str}
    e agrupa por tier {"easy": [...], "medium": [...], "hard": [...]}.
    """
    entries = []

    # --- social_data.json ({user, response}) ---
    if os.path.exists(SOCIAL_FILE):
        with open(SOCIAL_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw:
            prompt = item.get("user", "").strip()
            if not prompt:
                continue
            diff = _classify_difficulty(prompt)
            entries.append({"prompt": prompt, "intent": diff, "difficulty": diff})
    else:
        print(f"[WARN] social_data.json não encontrado: {SOCIAL_FILE}")

    # --- bercario_data.jsonl ({intent, prompt, target}) ---
    if os.path.exists(BERCARIO_FILE):
        with open(BERCARIO_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = item.get("prompt", "").strip()
                intent = item.get("intent", "").strip()
                if not prompt:
                    continue
                diff = _classify_difficulty(prompt, intent)
                entries.append({"prompt": prompt, "intent": intent, "difficulty": diff})
    else:
        print(f"[WARN] bercario_data.jsonl não encontrado: {BERCARIO_FILE}")

    # --- Deduplicar por prompt ---
    seen   = set()
    unique = []
    for e in entries:
        if e["prompt"] not in seen:
            seen.add(e["prompt"])
            unique.append(e)

    tiers: dict = {"easy": [], "medium": [], "hard": []}
    for e in unique:
        tiers[e["difficulty"]].append(e)

    print(f"[Curriculum] Dataset carregado: "
          f"{len(tiers['easy'])} easy | "
          f"{len(tiers['medium'])} medium | "
          f"{len(tiers['hard'])} hard")
    return tiers


def pick_prompt(tiers: dict, current_loss: float) -> dict:
    """
    Seleciona um prompt aleatório de acordo com a loss atual.
    Fallback: se o tier estiver vazio, tenta tiers adjacentes.
    """
    if current_loss > LOSS_EASY_THR:
        order = ["easy", "medium", "hard"]
    elif current_loss > LOSS_MEDIUM_THR:
        order = ["medium", "easy", "hard"]
    else:
        order = ["hard", "medium", "easy"]

    for tier in order:
        if tiers.get(tier):
            return random.choice(tiers[tier])

    # Último recurso: dataset vazio
    return {"prompt": "Olá!", "intent": "greeting", "difficulty": "easy"}


# ── Loop principal ────────────────────────────────────────────────────────────

def run_scheduler():
    tiers = load_all_datasets()
    total = sum(len(v) for v in tiers.values())
    if total == 0:
        print("[ERROR] Nenhum dado encontrado nos datasets!")
        return

    if not os.path.exists(MEM_FILE):
        print(f"[ERROR] Memória não encontrada. O Engine está ligado?")
        return

    print("--- [RLAIF CURRICULUM SCHEDULER: O BERÇÁRIO ESTÁ ONLINE] ---")
    print("Selecionando dificuldade dinamicamente pela loss do modelo...")

    with open(MEM_FILE, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 1024)

        try:
            while True:
                cmd_id = mm[0]

                if cmd_id == 0:
                    # Ler loss atual do mmap (offset 32, float64)
                    loss_bytes   = mm[32:40]
                    current_loss = struct.unpack('d', loss_bytes)[0]
                    if current_loss != current_loss:  # NaN
                        current_loss = 999.0

                    entry  = pick_prompt(tiers, current_loss)
                    prompt = entry["prompt"]
                    tier   = entry["difficulty"]

                    print(f"\n[RLAIF PULSE] loss={current_loss:.3f} → tier={tier.upper()}")
                    print(f"  Prompt: \"{prompt}\" (intent: {entry['intent']})")

                    try:
                        with FileLock(LOCK_FILE, timeout=10):
                            prompt_bytes = prompt.encode("utf-8")[:399]
                            mm[600:600 + len(prompt_bytes)] = prompt_bytes
                            mm[600 + len(prompt_bytes):1000] = b'\x00' * (400 - len(prompt_bytes))
                            mm[0] = 0x01
                        print("[✓] Pulso enviado com lock.")
                    except Timeout:
                        print("[!] Lock ocupado — pulso adiado para o próximo ciclo.")

                    time.sleep(360)   # 1 pulso a cada 6 min → 10/hora

                else:
                    print(f" [.] Engine ocupado (CmdID: {cmd_id:#04x}), aguardando...")
                    time.sleep(5)

        except KeyboardInterrupt:
            print("\nScheduler encerrado.")
        finally:
            mm.close()


if __name__ == "__main__":
    run_scheduler()
