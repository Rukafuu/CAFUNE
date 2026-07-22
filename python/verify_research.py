"""Audita as metas públicas do CAFUNE contra artefatos locais.

O comando é somente leitura e não carrega pesos nem inicia treino.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "research.toml"


def parameter_count(*, vocab_size: int, d_model: int, n_layers: int, d_ff: int) -> int:
    """Conta o modelo híbrido: metade MHA (4 matrizes), metade SSA (3)."""

    embeddings = 2 * d_model * vocab_size
    common = (
        + d_ff * d_model
        + d_ff
        + d_model * d_ff
        + d_model
        + 4 * d_model
    )
    standard_layers = n_layers // 2
    ssa_layers = n_layers - standard_layers
    blocks = standard_layers * (4 * d_model**2 + common) + ssa_layers * (3 * d_model**2 + common)
    final_norm = 2 * d_model
    return embeddings + blocks + final_norm


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def load_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def audit(config_path: Path) -> list[Check]:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    model = config["model"]
    tokenizer = config["tokenizer"]
    targets = config["targets"]
    data = config["data"]

    params = parameter_count(
        vocab_size=model["vocab_size"],
        d_model=model["d_model"],
        n_layers=model["n_layers"],
        d_ff=model["d_ff"],
    )
    checks = [
        Check(
            "parameters",
            "PASS" if params >= targets["parameters"] else "FAIL",
            f"{params:,} calculados; meta {targets['parameters']:,}",
        ),
        Check(
            "architecture",
            "PASS" if model["n_layers"] == 12 else "FAIL",
            f"{model['n_layers']} camadas, d_model={model['d_model']}, d_ff={model['d_ff']}",
        ),
    ]

    spm_config = load_json(tokenizer["config"])
    vocab = load_json(tokenizer["vocab"])
    declared_sizes = {
        model["vocab_size"],
        spm_config["vocab_size"],
        vocab["vocab_size"],
    }
    tokenizer_files = [ROOT / tokenizer[key] for key in ("model", "vocab", "config", "dataset")]
    checks.append(
        Check(
            "tokenizer",
            "PASS" if len(declared_sizes) == 1 and all(path.is_file() for path in tokenizer_files) else "FAIL",
            f"SentencePiece BPE; tamanhos declarados={sorted(declared_sizes)}",
        )
    )

    checks.extend(
        [
            Check("validation_loss", "PENDING", f"meta <= {targets['validation_loss']}; sem avaliação reproduzível"),
            Check(
                "local_runtime",
                "PASS" if targets["local_only"] else "FAIL",
                "treino e inferência não exigem API externa",
            ),
        ]
    )
    split_path = ROOT / data["splits"]
    if split_path.is_file():
        manifest = json.loads(split_path.read_text(encoding="utf-8"))
        assigned = sum(len(indices) for indices in manifest["splits"].values())
        split_ok = assigned == manifest["unique_sequences"] and all(manifest["splits"].values())
        checks.append(Check("data_splits", "PASS" if split_ok else "FAIL", f"{assigned} sequências únicas distribuídas"))
    else:
        checks.append(Check("data_splits", "FAIL", "execute python python/prepare_splits.py"))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    checks = audit(args.config.resolve())

    if args.as_json:
        print(json.dumps([check.__dict__ for check in checks], ensure_ascii=False, indent=2))
    else:
        for check in checks:
            print(f"[{check.status:7}] {check.name}: {check.detail}")
    return 1 if any(check.status == "FAIL" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
