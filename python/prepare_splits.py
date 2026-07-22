"""Cria splits determinísticos e deduplicados para o dataset tokenizado."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "research.toml"


def canonical_bytes(sequence: list[int]) -> bytes:
    return ",".join(map(str, sequence)).encode("ascii")


def build_manifest(dataset_path: Path, seed: str, train_cutoff: int, validation_cutoff: int) -> dict:
    raw = dataset_path.read_bytes()
    sequences: list[list[int]] = json.loads(raw)
    seen: dict[str, int] = {}
    splits = {"train": [], "validation": [], "test": []}

    for index, sequence in enumerate(sequences):
        digest = hashlib.sha256(canonical_bytes(sequence)).hexdigest()
        if digest in seen:
            continue
        seen[digest] = index
        bucket = int.from_bytes(hashlib.sha256(f"{seed}:{digest}".encode()).digest()[:8], "big") % 10_000
        if bucket < train_cutoff:
            split = "train"
        elif bucket < validation_cutoff:
            split = "validation"
        else:
            split = "test"
        splits[split].append(index)

    return {
        "schema_version": 1,
        "dataset": str(dataset_path.relative_to(ROOT)).replace("\\", "/"),
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "split_seed": seed,
        "total_sequences": len(sequences),
        "unique_sequences": len(seen),
        "duplicates_removed": len(sequences) - len(seen),
        "splits": splits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    tokenizer = config["tokenizer"]
    data = config["data"]
    dataset_path = ROOT / tokenizer["dataset"]
    output_path = ROOT / data["splits"]
    train_cutoff = round(float(data["train_ratio"]) * 10_000)
    validation_cutoff = train_cutoff + round(float(data["validation_ratio"]) * 10_000)
    manifest = build_manifest(dataset_path, data["split_seed"], train_cutoff, validation_cutoff)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    counts = {name: len(indices) for name, indices in manifest["splits"].items()}
    print(f"Splits salvos em {output_path}: {counts}")
    print(f"Duplicatas removidas: {manifest['duplicates_removed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

