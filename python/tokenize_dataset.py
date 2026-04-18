"""
tokenize_dataset.py — Tokeniza bercario_data.jsonl com SentencePiece e salva
como dataset_tokens.json para o Julia consumir diretamente.

Uso:
    python python/tokenize_dataset.py

Gera:
    python/dataset_tokens.json — lista de sequências já tokenizadas
    python/spm_config.json     — config do SPM para o Julia (vocab_size, mask_id, etc.)
"""

import io
import sys
import json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import sentencepiece as spm

SCRIPT_DIR    = Path(__file__).parent
DATA_FILE     = SCRIPT_DIR / "bercario_data.jsonl"
MODEL_FILE    = SCRIPT_DIR / "cafune_spm.model"
TOKENS_OUT    = SCRIPT_DIR / "dataset_tokens.json"
CONFIG_OUT    = SCRIPT_DIR / "spm_config.json"

SEQ_LEN = 128   # deve ser igual ao SEQ_LEN do Julia


def load_texts() -> list[str]:
    texts = []
    seen = set()
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                prompt = obj.get("prompt", "").strip()
                target = obj.get("target", obj.get("response", "")).strip()
                # Concatena prompt + target como sequência única
                combined = (prompt + " " + target).strip() if target else prompt
                if combined and combined not in seen:
                    seen.add(combined)
                    texts.append(combined)
                # Também adiciona separados para mais diversidade
                if prompt and prompt not in seen:
                    seen.add(prompt)
                    texts.append(prompt)
                if target and target not in seen:
                    seen.add(target)
                    texts.append(target)
            except json.JSONDecodeError:
                pass
    return texts


def pad_or_truncate(ids: list[int], seq_len: int, pad_id: int = 0) -> list[int]:
    if len(ids) >= seq_len:
        return ids[:seq_len]
    return ids + [pad_id] * (seq_len - len(ids))


def main():
    print("=== Tokenizando dataset com SentencePiece ===")

    if not MODEL_FILE.exists():
        print(f"ERRO: {MODEL_FILE} não encontrado. Execute train_spm.py primeiro.")
        sys.exit(1)

    sp = spm.SentencePieceProcessor()
    sp.load(str(MODEL_FILE))

    vocab_size   = sp.get_piece_size()
    mask_id      = sp.piece_to_id("[MASK]")
    pad_id       = sp.piece_to_id("[PAD]")
    bos_id       = sp.piece_to_id("[BOS]")
    eos_id       = sp.piece_to_id("[EOS]")

    print(f"Modelo: {MODEL_FILE.name}")
    print(f"vocab_size={vocab_size}, mask_id={mask_id}, pad_id={pad_id}")

    texts = load_texts()
    print(f"Sequências únicas: {len(texts)}")

    # Tokeniza tudo
    sequences = []
    for text in texts:
        ids = [bos_id] + sp.encode(text) + [eos_id]
        padded = pad_or_truncate(ids, SEQ_LEN, pad_id)
        sequences.append(padded)

    print(f"Sequências tokenizadas: {len(sequences)}")

    # Verifica qualidade
    sample_text = "Ultimamente tenho me sentido meio perdido, sabe?"
    sample_ids = sp.encode(sample_text)
    sample_tokens = sp.encode(sample_text, out_type=str)
    print(f"\nExemplo: '{sample_text}'")
    print(f"  Tokens: {sample_tokens}")
    print(f"  IDs:    {sample_ids}")

    # Salva dataset tokenizado
    TOKENS_OUT.write_text(json.dumps(sequences), encoding="utf-8")
    print(f"\nDataset salvo: {TOKENS_OUT} ({len(sequences)} sequências × {SEQ_LEN} tokens)")

    # Salva config para o Julia
    config = {
        "vocab_size":   vocab_size,
        "mask_id":      mask_id,
        "pad_id":       pad_id,
        "bos_id":       bos_id,
        "eos_id":       eos_id,
        "seq_len":      SEQ_LEN,
        "model_file":   str(MODEL_FILE),
        "n_sequences":  len(sequences),
        "tokenizer":    "sentencepiece",
    }
    CONFIG_OUT.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Config salva: {CONFIG_OUT}")


if __name__ == "__main__":
    main()
