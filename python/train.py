"""
train.py — Script de Treino Principal do CAFUNE

Fluxo completo:
    1. Carregar dados (texto PT-BR)
    2. Construir tokenizador character-level
    3. Tokenizar e criar dataset
    4. Inicializar bridge Python↔Julia
    5. Treinar o modelo DLLM
    6. Testar geração
"""

import sys
import os
import logging
from pathlib import Path

# Adicionar python/ ao path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from tokenizer import BPETokenizer, MASK_TOKEN_ID

# ──────────────────────────────────────────────────────────────
#  Dataset Real (Fase 2: Escala)
# ──────────────────────────────────────────────────────────────

EXTENDED_CORPUS = [
    "Olá! Como posso te ajudar hoje?",
    "Eu sou o CAFUNE, seu assistente virtual de inteligência artificial.",
    "Estou pronto para conversar, ajudar com código ou qualquer outra tarefa.",
    "O segredo para um bom dia é começar com foco e uma boa conversa.",
    "A tecnologia deve ser simples, direta e amigável para todos os usuários.",
    "Eu aprendo a remover o ruído das informações para encontrar a clareza.",
    "O projeto CAFUNE integra Haskell, Julia e Python em uma arquitetura única.",
    "Podemos falar sobre programação, curiosidades ou apenas bater um papo.",
    "Meu nome vem do gesto brasileiro de carinho, refletindo minha natureza prestativa.",
    "Manter a calma e a clareza é essencial para resolver problemas complexos.",
    "A inteligência artificial é uma ferramenta potente para a criatividade humana.",
    "Sempre busque aprender algo novo todos os dias, as sinapses agradecem!",
    "Eu moro no seu computador e estou sempre pronto para processar seus pedidos.",
    "O futuro da tecnologia é ser invisível e estar em sintonia com os humanos.",
    "Qualquer dúvida que você tiver, basta perguntar! Estou aqui para isso."
]


def prepare_dataset(corpus, seq_len, tok):
    """Tokeniza o corpus e aplica padding para formar o dataset de treino."""
    dataset = []
    for text in corpus:
        ids = tok.encode(text, add_special=True)
        # Pad or truncate
        ids = tok.pad(ids[:seq_len], seq_len)
        dataset.append(ids)
    return dataset

def evaluate(bridge, tok):
    """Teste rápido de denoising (revelação de tokens)."""
    print("\n[Avaliação] 🧪 Testando Revelação de Tokens (Denoising)...")
    # Gerar uma sequência do zero via processo de difusão iterativo
    generated_ids = bridge.generate(seq_len=15, num_steps=5, strategy="confidence")
    decoded = tok.decode(generated_ids)
    print(f"   🤖 Resposta do CAFUNE: '{decoded}'")

def main():
    logger.info("CAFUNE — Treino Fase 7 (Tokenização BPE & Escala)")

    # ── Configuração ──
    SEQ_LEN = 128
    D_MODEL = 256
    N_HEADS = 8
    N_LAYERS = 6
    D_FF = 1024
    EPOCHS = 20
    MAX_LR = 4e-4
    WARMUP_RATIO = 0.15

    # ── Tokenizador ──
    logger.info("[1/4] Construindo tokenizador (character-level)...")
    tok = BPETokenizer(vocab_size=512)
    tok.build_vocab(EXTENDED_CORPUS)
    tok.save("vocab.json")

    # ── Dataset ──
    logger.info("[2/4] Preparando dataset...")
    dataset = prepare_dataset(EXTENDED_CORPUS, SEQ_LEN, tok)
    logger.info("Dataset: %d sequências de comprimento %d", len(dataset), SEQ_LEN)

    # ── Bridge Julia ──
    logger.info("[3/4] Inicializando motor Julia (Zygote/Pure-AD)...")
    try:
        from bridge import CAFUNEBridge
        bridge = CAFUNEBridge()
        bridge.build_model(
            vocab_size=tok.vocab_size,
            seq_len=SEQ_LEN,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            d_ff=D_FF,
            num_diff_steps=20,
        )
    except ImportError as e:
        logger.error("Bridge Julia não disponível: %s", e)
        return

    # ── Treino ──
    logger.info("[4/4] Treinando — epochs=%d lr=%.0e", EPOCHS, MAX_LR)
    bridge.train_on_batch(
        dataset,
        epochs=EPOCHS,
        max_lr=MAX_LR,
        warmup_ratio=WARMUP_RATIO
    )

    # ── Avaliação ──
    evaluate(bridge, tok)
    logger.info("Treino Fase 7 concluído.")


if __name__ == "__main__":
    main()
