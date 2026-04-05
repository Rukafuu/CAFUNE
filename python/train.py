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

from tokenizer import BPETokenizer, MASK_TOKEN_ID

# ──────────────────────────────────────────────────────────────
#  Dataset Real (Fase 2: Escala)
# ──────────────────────────────────────────────────────────────

EXTENDED_CORPUS = [
    "O CAFUNE é um motor de difusão discreta que opera em múltiplas linguagens.",
    "Brazilians love coffee and the word cafuné means to run your fingers through someone's hair.",
    "A inteligência artificial evolui rapidamente através de arquiteturas como o Transformer.",
    "O projeto utiliza Haskell para orquestração, Python para dados e Julia para o motor matemático.",
    "O kernel CUDA em C++ permite que a atenção seja calculada com velocidade máxima em GPUs NVIDIA.",
    "No processo de difusão, o ruído é adicionado gradualmente e o modelo aprende a revertê-lo.",
    "Ao contrário de modelos GPT, o CAFUNE vê todo o contexto simultaneamente via atenção bidirecional.",
    "O Zygote em Julia permite a diferenciação automática de funções arbitrárias de alto desempenho.",
    "A pureza funcional em Haskell garante que o estado do escalonador de difusão seja determinístico.",
    "O motor LLaDA-style permite gerar texto de forma paralela e iterativa, ganhando eficiência.",
    "A convergência do treino é otimizada por um agendamento de aprendizado em cosseno com aquecimento.",
    "FrankensteinDLLM foi o nome interno do projeto antes de ser batizado como CAFUNE.",
    "A beleza da computação está na integração de diferentes paradigmas e linguagens de programação.",
    "Treinar um modelo de linguagem requer dados de alta qualidade e infraestrutura estável.",
    "O futuro da IA aponta para sistemas híbridos onde a lógica e a matemática se complementam.",
    "A atenção de produto escalar escalonada é a base de quase todos os modelos de linguagem modernos.",
    "O dropout é uma técnica de regularização essencial para evitar o sobreajuste em redes neurais."
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
    print("CAFUNE (Composite Architecture for Fast Universal Noise-reduction Engine)")
    print("   Treino Fase 7 -- Tokenização BPE & Escala")
    print("="*50)

    # ── Configuração ──
    SEQ_LEN = 128
    D_MODEL = 256
    N_HEADS = 8
    N_LAYERS = 6
    D_FF = 1024
    
    EPOCHS = 20
    MAX_LR = 4e-4
    WARMUP_RATIO = 0.15

    # ── Tokenizador BPE ──
    print(f"\n[1/4] Forjando Tokenizer BPE (Sub-palavras)...")
    tok = BPETokenizer(vocab_size=512)
    tok.build_vocab(EXTENDED_CORPUS)
    tok.save("vocab.json")

    # ── Dataset ──
    print(f"\n[2/4] Preparando dataset neuro-codificado...")
    dataset = prepare_dataset(EXTENDED_CORPUS, SEQ_LEN, tok)

    # ── Bridge Julia ──
    print(f"\n[3/4] Inicializando motor Julia (Zygote/Pure-AD)...")
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
        print(f"\n⚠️  Bridge Julia não disponível: {e}")
        return

    # ── Treino ──
    print(f"\n[4/4] Treinando com BPE Ativado...")
    bridge.train_on_batch(
        dataset, 
        epochs=EPOCHS, 
        max_lr=MAX_LR, 
        warmup_ratio=WARMUP_RATIO
    )

    # ── Avaliação ──
    evaluate(bridge, tok)

    print("\n🚀 TREINO FASE 7 CONCLUÍDO!")


if __name__ == "__main__":
    main()
