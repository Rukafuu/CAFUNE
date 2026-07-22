"""Configuração compartilhada dos serviços CAFUNE."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = Path(os.getenv("CAFUNE_RUNTIME_DIR", PROJECT_ROOT)).resolve()

MEM_FILE = Path(os.getenv("CAFUNE_MEM_FILE", RUNTIME_DIR / "cafune_brain.mem")).resolve()
MEM_SIZE = 2048
LOCK_FILE = Path(f"{MEM_FILE}.lock")

CMD_OFFSET = 0
RESPONSE_START = 200
RESPONSE_END = 600
PROMPT_START = 600
PROMPT_END = 1000


def ensure_runtime_dir() -> None:
    """Cria somente o diretório necessário para artefatos de execução."""

    MEM_FILE.parent.mkdir(parents=True, exist_ok=True)


def ensure_mmap() -> None:
    """Cria ou expande o mmap canônico sem sobrescrever dados existentes."""

    ensure_runtime_dir()
    current_size = MEM_FILE.stat().st_size if MEM_FILE.exists() else 0
    if current_size < MEM_SIZE:
        with MEM_FILE.open("ab") as stream:
            stream.write(b"\0" * (MEM_SIZE - current_size))
