"""
rebuild_vocab.py — Reconstrói vocab.json a partir de bercario_data.jsonl

Salva em python/vocab_new.json para comparação antes de substituir.
Usa o mesmo formato do CharTokenizer (char2id + BPE merges).

Diferença do vocab_builder.py antigo:
  - Lê bercario_data.jsonl (dataset atual, ~4500+ pares PT-BR)
  - Inclui acentos e caracteres PT-BR completos
  - Salva como vocab_new.json (não sobrescreve vocab.json automaticamente)

Uso:
    python python/rebuild_vocab.py
    python python/rebuild_vocab.py --apply   # substitui vocab.json
"""

import io
import json
import sys
import argparse

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import collections
from pathlib import Path

SCRIPT_DIR  = Path(__file__).parent
DATA_FILE   = SCRIPT_DIR / "bercario_data.jsonl"
VOCAB_OUT   = SCRIPT_DIR / "vocab_new.json"
VOCAB_FINAL = SCRIPT_DIR / "vocab.json"

VOCAB_SIZE  = 500       # alvo de tokens totais
MAX_TOKEN_LEN = 4       # filtro valid_ids no Julia

SPECIAL_TOKENS = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "[MASK]": 4}


def load_texts() -> list[str]:
    texts = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                for field in ("user", "response", "input", "output", "prompt", "target", "text"):
                    if field in obj and obj[field]:
                        texts.append(str(obj[field]))
            except json.JSONDecodeError:
                pass
    return texts


def build_bpe(texts: list[str], vocab_size: int) -> dict[str, int]:
    # 1. Base chars (sorted for determinism)
    all_chars = set()
    for t in texts:
        all_chars.update(t)
    char2id = dict(SPECIAL_TOKENS)
    for ch in sorted(all_chars):
        if ch not in char2id:
            char2id[ch] = len(char2id)

    print(f"  Base chars: {len(all_chars)} | IDs atribuídos: {len(char2id)}")

    # 2. BPE merges
    corpus = [list(t) for t in texts]
    num_merges = vocab_size - len(char2id)
    if num_merges <= 0:
        print(f"  Já temos {len(char2id)} tokens ≥ vocab_size={vocab_size}. Pulando merges.")
        return char2id

    print(f"  Fazendo {num_merges} merges BPE...")
    merges = {}
    for i in range(num_merges):
        pairs = collections.Counter()
        for seq in corpus:
            for a, b in zip(seq, seq[1:]):
                pairs[(a, b)] += 1
        if not pairs:
            break

        best = pairs.most_common(1)[0][0]
        merged = best[0] + best[1]
        if merged in char2id:
            # já existe — skip
            continue
        char2id[merged] = len(char2id)
        merges[best] = merged

        new_corpus = []
        for seq in corpus:
            new_seq, j = [], 0
            while j < len(seq):
                if j < len(seq) - 1 and (seq[j], seq[j+1]) == best:
                    new_seq.append(merged)
                    j += 2
                else:
                    new_seq.append(seq[j])
                    j += 1
            new_corpus.append(new_seq)
        corpus = new_corpus

        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{num_merges}] '{best[0]}' + '{best[1]}' → '{merged}'")

    return char2id


def stats(char2id: dict[str, int], label: str):
    valid = [t for t in char2id if t not in SPECIAL_TOKENS and len(t) <= MAX_TOKEN_LEN]
    print(f"\n{label}:")
    print(f"  Total tokens : {len(char2id)}")
    print(f"  valid_ids (len≤{MAX_TOKEN_LEN}): {len(valid)}")
    accented = [t for t in valid if any(c in t for c in "áàâãéêíóôõúüç")]
    print(f"  Com acentos  : {len(accented)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Substitui vocab.json pelo novo (requer retrain do checkpoint!)")
    args = parser.parse_args()

    print(f"=== Rebuild Vocab CAFUNE ===")
    print(f"Dataset: {DATA_FILE}")

    texts = load_texts()
    print(f"Textos carregados: {len(texts)}")
    total_chars = sum(len(t) for t in texts)
    print(f"Total de caracteres: {total_chars:,}")

    # Vocab atual
    old_vocab = json.loads(VOCAB_FINAL.read_text(encoding="utf-8"))
    old_char2id = old_vocab.get("char2id", {})
    stats(old_char2id, "Vocab ATUAL (vocab.json)")

    # Novo vocab
    print(f"\nConstruindo novo vocab (target={VOCAB_SIZE})...")
    new_char2id = build_bpe(texts, VOCAB_SIZE)
    stats(new_char2id, "Vocab NOVO")

    # Salva vocab_new.json
    out = {"char2id": new_char2id, "vocab_size": len(new_char2id)}
    VOCAB_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSalvo em: {VOCAB_OUT}")

    if args.apply:
        # Backup do antigo
        backup = VOCAB_FINAL.with_suffix(".json.bak")
        backup.write_text(VOCAB_FINAL.read_text(encoding="utf-8"), encoding="utf-8")
        VOCAB_FINAL.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"vocab.json substituído! Backup em: {backup}")
        print(f"\n⚠️  O checkpoint atual (julia/checkpoints/) é INCOMPATÍVEL com o novo vocab.")
        print(f"   Delete os checkpoints e reinicie o treino do zero para usar o novo vocab.")
    else:
        print(f"\nPara aplicar e substituir vocab.json:")
        print(f"  python python/rebuild_vocab.py --apply")
        print(f"\n⚠️  Isso invalida o checkpoint atual — requer retrain do zero.")


if __name__ == "__main__":
    main()
