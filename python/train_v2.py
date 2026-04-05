import sys
from pathlib import Path
import os

# Add julia/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "julia"))

from bridge import CAFUNEBridge
from tokenizer import CharTokenizer

SAMPLE_CORPUS = [
    "O gato dormiu no telhado.",
    "A IA vai dominar o mundo.",
    "Julia e rapida e Haskell e segura.",
    "O segredo da difusao esta no ruido.",
]

def main():
    print("CAFUNE - Treino Fase 2 (Zygote + Optimisers)")
    print("="*50)

    # Config
    SEQ_LEN = 32
    EPOCHS = 5

    # Tokenizer
    tok = CharTokenizer()
    tok.build_vocab(SAMPLE_CORPUS)
    
    # Dataset
    dataset = [tok.pad(tok.encode(text, add_special=True)[:SEQ_LEN], SEQ_LEN) for text in SAMPLE_CORPUS]

    # Bridge Julia
    bridge = CAFUNEBridge()
    bridge.build_model(
        vocab_size=tok.vocab_size,
        seq_len=SEQ_LEN,
        d_model=64, # Small for CPU
        n_heads=4,
        n_layers=2,
        num_diff_steps=10,
    )

    print(f"\nIniciando Treino (Autodiff)...")
    bridge.train_on_batch(dataset, epochs=EPOCHS)

    print("\nTestando geracao...")
    generated = bridge.generate(seq_len=20)
    print(f"Gerado: '{tok.decode(generated)}'")

if __name__ == "__main__":
    main()
