"""Ponte JSON/stdin para o tokenizador SentencePiece local."""

from __future__ import annotations

import argparse
import json
import sys

from tokenizer import BPETokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("encode", "decode"))
    args = parser.parse_args()
    tokenizer = BPETokenizer()

    if args.operation == "encode":
        print(json.dumps(tokenizer.encode(sys.stdin.read(), add_special=False)))
    else:
        token_ids = json.loads(sys.stdin.read())
        print(tokenizer.decode(token_ids, skip_special=True), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

