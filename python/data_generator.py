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

# Força stdout UTF-8 no Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

BERCARIO_FILE = os.path.normpath(os.path.join(os.path.dirname(__file__), "bercario_data.jsonl"))
PAIRS_PER_CYCLE = 10     # pares gerados por ciclo
CYCLE_SLEEP     = 60     # segundos entre ciclos (1 min)
MAX_ENTRIES     = 6000   # não deixa o arquivo crescer infinitamente

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


# ── Geração local de pares (sem API) ─────────────────────────────────────────

# Templates de prompts e respostas por tópico
PAIR_TEMPLATES = {
    "como fazer amigos sendo introvertido": [
        ("É difícil fazer amigos sendo introvertido, o que você acha?", "Faz sentido se sentir assim! Uma dica é começar por interesses em comum — grupos de leitura, jogos, hobby. Não precisa forçar — conexões genuínas surgem naturalmente quando você está no seu elemento."),
        ("Me sinto solitário mas tenho dificuldade de me aproximar das pessoas.", "Solidão dói, mas saiba que muita gente sente isso. Tente dar um pequeno passo: responder uma mensagem, participar de um evento online. Cada pequeno contato conta."),
        ("Como posso ser mais sociável sem me sentir esgotado?", "Introvertidos recarregam sozinhos — e tá tudo bem! Escolha interações que valem a pena pra você, e reserve tempo de descanso depois. Qualidade é melhor que quantidade."),
    ],
    "lidar com ansiedade no trabalho": [
        ("Fico muito ansioso antes de reuniões importantes.", "Essa sensação é super comum! Respirar fundo algumas vezes antes de entrar ajuda muito. Lembre que você foi convidado para contribuir — sua presença tem valor."),
        ("Como diminuir a ansiedade no trabalho?", "Tenta identificar o gatilho específico. É prazo? Julgamento? Com isso em mente fica mais fácil agir. Pausas curtas ao longo do dia também fazem uma diferença enorme."),
        ("Meu trabalho me deixa estressado todo dia.", "Isso pesa bastante com o tempo. Tente criar uma rotina de transição entre trabalho e descanso — um passeio curto, música. Sinaliza pro seu cérebro que acabou o expediente."),
    ],
    "dicas para estudar com foco": [
        ("Não consigo me concentrar estudando, me distraio fácil.", "Tenta a técnica Pomodoro: 25 minutos focado, 5 de pausa. Celular longe, ambiente silencioso. Seu cérebro aprende a entrar no ritmo com o tempo."),
        ("Qual o melhor jeito de estudar sem perder o foco?", "Começa pelas matérias mais difíceis quando a energia está alta. Divide o conteúdo em blocos pequenos e anota dúvidas em vez de parar tudo pra pesquisar."),
        ("Me sinto sobrecarregado com tanto conteúdo pra estudar.", "Prioriza o essencial primeiro. Uma lista do que é mais importante hoje ajuda a não se perder. Estudar menos com mais foco supera estudar muito sem absorver."),
    ],
    "como pedir desculpas de forma sincera": [
        ("Errei com um amigo e não sei como pedir desculpa.", "O mais importante é ser específico: diz o que você fez, reconhece o impacto, e mostra que entende por que magoou. Evita 'me desculpa se você se sentiu mal' — assuma de verdade."),
        ("Como pedir desculpas sem parecer falso?", "Fale com suas próprias palavras, sem script. Mostre que pensou no que aconteceu. E dê espaço pra outra pessoa processar — desculpa verdadeira não exige resposta imediata."),
    ],
    "como manter uma conversa interessante": [
        ("Sempre fico sem assunto nas conversas, o que fazer?", "Faz perguntas abertas — aquelas que não têm só sim ou não. Escuta de verdade a resposta, sem pensar no que vai dizer. As melhores conversas surgem da curiosidade genuína."),
        ("Como não deixar a conversa morrer?", "Compartilha algo seu também. Conversas são uma troca. Se perceber que travou, um 'o que você acha de...' sobre algo do contexto quase sempre funciona."),
    ],
    "curiosidades sobre o universo": [
        ("Me conta uma curiosidade legal sobre o universo.", "A luz que vemos das estrelas pode ter milhões de anos — algumas estrelas que vemos já nem existem mais. Você literalmente olha pro passado quando olha pro céu."),
        ("O universo é infinito?", "Ainda não sabemos! O universo observável tem cerca de 93 bilhões de anos-luz de diâmetro, mas o que está além disso é uma das maiores questões abertas da ciência."),
    ],
    "história da internet em resumo": [
        ("Como surgiu a internet?", "A internet começou como ARPANET nos anos 60, um projeto militar dos EUA para comunicação resistente a falhas. A web que conhecemos hoje foi criada por Tim Berners-Lee em 1989 para compartilhar documentos científicos."),
        ("Quando a internet chegou ao Brasil?", "A internet comercial chegou ao Brasil em 1995. Antes disso, só universidades e instituições de pesquisa tinham acesso, desde 1988."),
    ],
    "como funciona a memória humana": [
        ("Como funciona a memória?", "A memória tem várias etapas: codificação, armazenamento e recuperação. O hipocampo é essencial para formar novas memórias. Dormir bem é fundamental — é durante o sono que o cérebro consolida o que aprendeu."),
        ("Por que a gente esquece as coisas?", "Esquecer é normal e até útil — o cérebro filtra o que considera menos importante. Estresse, falta de sono e falta de repetição são os principais vilões da memória."),
    ],
    "por que sonhamos": [
        ("Por que a gente sonha?", "Ainda não há consenso científico, mas as teorias mais aceitas sugerem que sonhos ajudam a processar emoções, consolidar memórias e até resolver problemas. É seu cérebro trabalhando enquanto você descansa."),
        ("Os sonhos têm significado?", "Depende do ponto de vista. Psicologicamente, podem refletir preocupações e desejos. Neurologicamente, são padrões aleatórios que o cérebro tenta organizar em narrativa. As duas coisas podem ser verdade ao mesmo tempo."),
    ],
    "o que é inteligência artificial": [
        ("Me explica o que é inteligência artificial de forma simples.", "IA é quando um computador aprende a fazer tarefas que normalmente precisariam de inteligência humana — reconhecer imagens, entender texto, jogar xadrez. Ele aprende a partir de exemplos, não de regras fixas."),
        ("Como funciona o ChatGPT e IAs assim?", "São modelos de linguagem treinados com enormes quantidades de texto. Eles aprendem padrões e probabilidades de quais palavras seguem outras. Não 'entendem' como humanos, mas ficaram muito bons em imitar compreensão."),
    ],
    "como aprender a cozinhar do zero": [
        ("Nunca cozinhei nada, por onde começo?", "Começa pelo básico: arroz, ovo, macarrão. Receitas simples ensinam técnicas fundamentais. YouTube é seu melhor amigo — vídeo curto, reproduz na cozinha mesmo."),
        ("Qual receita fácil pra quem está aprendendo a cozinhar?", "Ovo mexido é perfeito pra começar — ensina controle de fogo e textura. Depois tenta um macarrão ao alho e óleo: simples, saboroso e você aprende a temperar."),
    ],
    "dicas para economizar dinheiro": [
        ("Como começar a economizar dinheiro?", "O primeiro passo é entender pra onde o dinheiro vai. Anota tudo que gasta por uma semana. Muita gente se surpreende com o que descobre. Depois é mais fácil cortar o supérfluo."),
        ("Nunca consigo guardar dinheiro no fim do mês.", "Tenta inverter: assim que receber, separa uma quantia pra poupança antes de gastar. Mesmo que seja pouco. O hábito de poupar primeiro muda muito a relação com dinheiro."),
    ],
    "como montar uma rotina produtiva": [
        ("Como criar uma rotina que eu consiga manter?", "Começa pequeno — só 2 ou 3 hábitos novos por vez. Ancora eles em algo que já faz: depois do café, antes de dormir. Consistência pequena bate planejamento grandioso que não sai do papel."),
        ("Minha rotina sempre desanda, como manter?", "Rotinas quebram — isso é normal. O truque é não desistir quando quebrar. Um dia ruim não destrói um hábito; desistir, sim. Recomece no dia seguinte sem culpa."),
    ],
    "como lidar com críticas no trabalho": [
        ("Receber crítica no trabalho me afeta muito emocionalmente.", "Faz sentido — críticas mexem com a autoestima. Tenta separar a crítica do trabalho da crítica à pessoa. Um respiro antes de reagir ajuda a ouvir o que é construtivo sem ficar na defensiva."),
        ("Como não levar críticas para o lado pessoal?", "Pergunta a si mesmo: isso me ajuda a melhorar? Se sim, é um presente embrulhado de forma ruim. Se não, é opinião de alguém com o dia ruim. Nem toda crítica merece peso igual."),
    ],
    "técnicas de respiração para acalmar": [
        ("Me ensina uma técnica de respiração pra ansiedade.", "Tenta a respiração 4-7-8: inspira por 4 segundos, segura por 7, solta por 8. Ativa o sistema nervoso parassimpático — o modo 'calma' do seu corpo. Funciona rápido."),
        ("Como a respiração ajuda a acalmar?", "Quando você controla a respiração, manda um sinal pro cérebro de que está seguro. Respiração rápida e rasa alimenta ansiedade; lenta e profunda, acalma. Você tem controle sobre isso mesmo em momentos difíceis."),
    ],
    "o que é felicidade segundo a filosofia": [
        ("O que é felicidade de verdade?", "Pra Aristóteles, felicidade é eudaimonia — viver de acordo com seus valores e potencial, não só sentir prazer. É mais sobre significado do que sobre sensação. Faz sentido pra você?"),
        ("A filosofia tem resposta pra como ser feliz?", "Cada escola tem a sua! Estoicos dizem que está no controle do que é seu. Epicuristas, em prazeres simples e amizades. O ponto em comum: felicidade vem de dentro, não de circunstâncias."),
    ],
    "como superar um término de relacionamento": [
        ("Terminei um relacionamento e estou arrasado.", "Isso dói muito, e tá tudo bem sentir assim. Não tenta pular essa dor — ela é parte do processo. Cuide de si: dorme, come, fala com quem você confia. Um dia de cada vez."),
        ("Quanto tempo leva pra superar um término?", "Não tem prazo certo — e não deveria ter. O que ajuda é não se isolar, manter alguma rotina e permitir sentir sem se afundar. A dor vai diminuindo, mesmo que agora pareça impossível."),
    ],
    "significado de empatia na prática": [
        ("O que é empatia na prática mesmo?", "Empatia é tentar sentir o que o outro está sentindo, não só entender na teoria. Na prática: ouvir sem julgar, perguntar 'como você está?' e de verdade esperar a resposta."),
        ("Como ser mais empático com as pessoas?", "Começa pela escuta. A maioria de nós fica pensando na resposta enquanto o outro fala. Tenta só ouvir, sem preparar nada. Você vai se surpreender com o que percebe."),
    ],
    "por que pessoas mentem": [
        ("Por que as pessoas mentem mesmo sabendo que é errado?", "Geralmente por medo — de julgamento, de consequências, de decepcionar. Mentiras pequenas muitas vezes são tentativas de se proteger ou proteger alguém. Não justifica, mas explica."),
        ("Todo mundo mente?", "Pesquisas sugerem que sim — em média, pessoas mentem algumas vezes por dia, na maioria das vezes sobre coisas pequenas. A capacidade de mentir está ligada à teoria da mente, que é justamente o que nos permite entender os outros."),
    ],
    "como encontrar propósito de vida": [
        ("Como saber qual é o meu propósito de vida?", "Propósito raramente aparece de repente — vai sendo descoberto vivendo. Presta atenção no que te faz perder a noção do tempo, no que te incomoda no mundo e no que você faria mesmo sem ser pago."),
        ("Me sinto perdido sem um propósito claro.", "Essa sensação é muito comum, especialmente em fases de transição. Propósito pode ser pequeno: cuidar bem de quem você ama, aprender algo novo, contribuir onde está. Não precisa ser épico pra ser real."),
    ],
}


def generate_pair_local(topic: str) -> dict | None:
    """Gera um par conversacional a partir dos templates locais."""
    templates = PAIR_TEMPLATES.get(topic)
    if not templates:
        return None
    prompt_text, target_text = random.choice(templates)
    return {
        "prompt":   prompt_text,
        "target":   target_text,
        "intent":   get_intent(topic),
        "topic":    topic,
        "grounded": False,
        "source":   "local-template",
    }


def generate_pair_llm(topic: str) -> dict | None:
    """
    Gera um par conversacional usando BitNet 1-bit se disponível,
    caso contrário usa templates locais como fallback.
    """
    try:
        from bitnet_client import is_server_alive, generate_content
        if not is_server_alive():
            raise ConnectionError("BitNet offline")

        prompt = f"""Estou criando um dataset para treinar uma IA chamada CAFUNE.
Tópico sorteado: "{topic}"
Crie um par de conversa com:
1. Uma pergunta humana natural (prompt).
2. Uma resposta empática e informativa da IA (target).

Responda apenas em JSON:
{{ "prompt": "a pergunta", "target": "a resposta perfeita" }}"""

        resp = generate_content(
            prompt,
            system_instruction="Você constrói datasets JSON válidos em português brasileiro.",
            as_json=True,
            temperature=0.8,
        )
        if not resp:
            raise ValueError("Resposta vazia do BitNet")

        text = resp.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)

        if not data.get("prompt") or not data.get("target"):
            raise ValueError("JSON sem campos prompt/target")

        logger.info("  [BitNet] par gerado | topic: %s", topic[:40])
        return {
            "prompt":  data["prompt"].strip(),
            "target":  data["target"].strip(),
            "intent":  get_intent(topic),
            "topic":   topic,
            "grounded": False,
            "source":  "bitnet-llm",
        }

    except Exception as e:
        # Fallback para templates locais
        logger.debug("BitNet indisponível (%s) — usando template local", e)
        return generate_pair_local(topic)


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
    logger.info("=== [DATA GENERATOR: BitNet Local Server] ===")
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
            entry = generate_pair_llm(topic)
            if entry is None:
                continue

            # Templates locais podem repetir prompt — permite com variação no target
            # Pares do BitNet são deduplicados normalmente (conteúdo único)
            is_template = entry.get("source") == "local-template"
            if entry["prompt"] in seen_prompts and not is_template:
                logger.info("  [skip] Duplicata: %s", entry["prompt"][:50])
                continue

            append_entry(entry)
            seen_prompts.add(entry["prompt"])
            added += 1
            logger.info("  [BitNet] +1 par | intent=%s | prompt: %s",
                        entry["intent"], entry["prompt"][:60])

        logger.info("Ciclo %d concluido: %d/%d pares adicionados | proximo em %ds",
                    cycle, added, len(topics_this_cycle), CYCLE_SLEEP)
        time.sleep(CYCLE_SLEEP)


if __name__ == "__main__":
    run_generator()
