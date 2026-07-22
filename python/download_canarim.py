"""Baixa e integra uma amostra do Canarim ao berçário do CAFUNE.

O modo padrão é somente leitura. Use ``--apply`` para salvar o resultado e
``--output`` para não substituir o dataset atual.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset


DEFAULT_DATASET = "dominguesm/Canarim-Instruct-PTBR-Dataset"
DEFAULT_SAMPLE = 50_000
BERCARIO = Path(__file__).with_name("bercario_data.jsonl")


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_existing(path: Path) -> tuple[list[dict], set[str]]:
    entries: list[dict] = []
    seen: set[str] = set()
    if not path.exists():
        return entries, seen

    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = str(entry.get("prompt", "")).strip()
            if prompt:
                entries.append(entry)
                seen.add(fingerprint(prompt))
    return entries, seen


def valid(prompt: str, target: str) -> bool:
    return 5 <= len(prompt) and 10 <= len(target) <= 800 and target.count("\n") <= 15


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="salva o dataset combinado")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=BERCARIO)
    args = parser.parse_args()

    existing, seen = load_existing(BERCARIO)
    source = load_dataset(DEFAULT_DATASET, split="train")
    candidates: list[dict] = []

    for row in source:
        instruction = str(row.get("instruction", "")).strip()
        context = str(row.get("input", "") or "").strip()
        prompt = f"{instruction}\n{context}" if context else instruction
        target = str(row.get("output", "") or "").strip()
        key = fingerprint(prompt)
        if valid(prompt, target) and key not in seen:
            seen.add(key)
            candidates.append({"prompt": prompt, "target": target, "source": "canarim"})

    random.Random(args.seed).shuffle(candidates)
    selected = candidates[: args.sample]
    combined = existing + selected
    print(f"Existentes: {len(existing)} | Canarim: {len(selected)} | Total: {len(combined)}")

    if not args.apply:
        print("Dry run: use --apply para salvar.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for entry in combined:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Salvo em: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

