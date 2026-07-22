"""CLI operacional mínima do CAFUNE."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

from cafune_config import MEM_FILE, MEM_SIZE, PROJECT_ROOT, ensure_mmap


def _state(ok: bool) -> str:
    return "OK" if ok else "AUSENTE"


def doctor() -> int:
    """Mostra dependências e artefatos sem modificar o ambiente."""

    print(f"CAFUNE root: {PROJECT_ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"mmap: {MEM_FILE} [{_state(MEM_FILE.is_file())}]")
    if MEM_FILE.is_file():
        size = MEM_FILE.stat().st_size
        print(f"mmap size: {size} bytes ({'OK' if size == MEM_SIZE else 'INVÁLIDO'})")

    model_candidates = (
        PROJECT_ROOT / "julia" / "cafune_model.bson",
        PROJECT_ROOT / "cafune_model.bson",
    )
    for model in model_candidates:
        if model.is_file():
            print(f"checkpoint transformer: {model} [OK]")
            break
    else:
        print("checkpoint transformer: [AUSENTE]")

    snn = PROJECT_ROOT / "snn_weights.bson"
    print(f"checkpoint SNN: {snn} [{_state(snn.is_file())}]")

    required = ("flask", "filelock", "requests")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    print(f"dependências Python: {'OK' if not missing else 'faltando ' + ', '.join(missing)}")
    print(f"OpenRouter: {'configurado' if os.getenv('OPENROUTER_API_KEY') else 'desativado'}")
    return 1 if missing else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cafune", description="Ferramentas operacionais do CAFUNE")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="verifica ambiente e artefatos sem treinar")
    commands.add_parser("init-runtime", help="cria ou migra o mmap preservando seu conteúdo")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "doctor":
        return doctor()
    if args.command == "init-runtime":
        ensure_mmap()
        print(f"mmap pronto: {MEM_FILE} ({MEM_FILE.stat().st_size} bytes)")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
