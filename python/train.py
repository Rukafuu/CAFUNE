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
from pathlib import Path

# Adicionar python/ ao path
sys.path.insert(0, str(Path(__file__).parent))

from tokenizer import CharTokenizer, MASK_TOKEN_ID

# ──────────────────────────────────────────────────────────────
#  Dataset de exemplo (substituir por dados reais)
# ──────────────────────────────────────────────────────────────

SAMPLE_CORPUS = """
O gato preto dormia sobre o telhado quente.
A inteligência artificial vai transformar o mundo em que vivemos.
Frankenstein criou um monstro com partes de diferentes corpos.
A difusão discreta é uma alternativa poderosa aos modelos autoregressivos.
Julia é uma linguagem de programação feita para computação científica de alto desempenho.
Haskell garante a segurança dos tipos em tempo de compilação, evitando bugs sutis.
O processo de denoising remove o ruído iterativamente até revelar o texto limpo.
Cada passo de inferência desmascara os tokens mais confiantes primeiro.
A atenção bidirecional permite que o modelo veja todo o contexto de uma vez.
O transformer foi introduzido no paper Attention is All You Need em 2017.
""".strip().split("\n")


def prepare_dataset(corpus: list[str], seq_len: int, tok: CharTokenizer) -> list[list[int]]:
    """
    Tokeniza o corpus e prepara sequências de comprimento fixo.
    
    - Sequências longas são divididas em chunks
    - Sequências curtas são ignoradas (muito curtas para aprender)
    """
    dataset = []

    for text in corpus:
        text = text.strip()
        if not text:
            continue

        ids = tok.encode(text, add_special=True)

        # Dividir em chunks de seq_len
        for start in range(0, len(ids), seq_len):
            chunk = ids[start:start + seq_len]
            if len(chunk) >= 8:  # Mínimo de 8 tokens para aprender algo
                chunk = tok.pad(chunk, seq_len)
                dataset.append(chunk)

    print(f"Dataset pronto: {len(dataset)} sequencias")
    return dataset


def evaluate(bridge, tok: CharTokenizer, n_samples: int = 3):
    """Gera amostras para avaliar o modelo visualmente."""
    print("\n" + "="*50)
    print("AMOSTRAS GERADAS:")
    print("="*50)

    for i in range(n_samples):
        tokens = bridge.generate(seq_len=50, temperature=0.8, verbose=False)
        text = tok.decode(tokens)
        print(f"\n[{i+1}] {text}")

    print("="*50 + "\n")


def main():
    print("CAFUNE (Composite Architecture for Fast Universal Noise-reduction Engine)")
    print("   Treino Fase 1 -- Python + Julia")
    print("="*50)

    # ── Configuração ──
    SEQ_LEN = 64
    D_MODEL = 128
    N_HEADS = 4
    N_LAYERS = 4
    EPOCHS = 20

    # ── Tokenizador ──
    print("\n[1/4] Construindo tokenizador...")
    tok = CharTokenizer()
    tok.build_vocab(SAMPLE_CORPUS)
    tok.save("vocab.json")

    # ── Dataset ──
    print("\n[2/4] Preparando dataset...")
    dataset = prepare_dataset(SAMPLE_CORPUS, SEQ_LEN, tok)

    if not dataset:
        print("❌ Dataset vazio! Adicione mais texto ao corpus.")
        return

    # ── Bridge Julia ──
    print("\n[3/4] Inicializando motor Julia...")
    try:
        from bridge import CAFUNEBridge
        bridge = CAFUNEBridge()
        bridge.build_model(
            vocab_size=tok.vocab_size,
            seq_len=SEQ_LEN,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            num_diff_steps=30,
        )
    except ImportError as e:
        print(f"\n⚠️  Bridge Julia não disponível: {e}")
        print("   Instale: pip install juliacall")
        print("   Depois execute este script novamente.")
        print("\n   Enquanto isso, o tokenizador pode ser testado:")
        text = "O gato preto"
        ids = tok.encode(text)
        print(f"   '{text}' → {ids} → '{tok.decode(ids)}'")
        return

    # ── Treino ──
    print(f"\n[4/4] Treinando por {EPOCHS} épocas...")
    bridge.train_on_batch(dataset, epochs=EPOCHS)

    # ── Avaliação ──
    evaluate(bridge, tok)

    print("TREINO CONCLUÍDO! Próximos passos:")
    print("   1. Adicionar mais dados ao corpus")
    print("   2. Aumentar o modelo (small config)")
    print("   3. Implementar C/CUDA kernels")
    print("   4. Adicionar orquestrador Haskell")


if __name__ == "__main__":
    main()
