"""
tokenizer.py — Tokenizador Character-Level para o CAFUNE

Por que character-level na Fase 1?
- Zero dependências externas
- Fácil de entender e depurar
- Vocabulário pequeno (~100 chars) → modelo aprende mais rápido
- Pode trocar por BPE depois sem mudar a arquitetura

Vocabulário especial:
    [PAD] = 0   → padding de sequências
    [UNK] = 1   → caractere desconhecido
    [BOS] = 2   → início de sequência (begin of sequence)
    [EOS] = 3   → fim de sequência (end of sequence)
    [MASK] = 4  → token mascarado (o ruído do DLLM)
"""

import json
from pathlib import Path
from typing import Union

SPECIAL_TOKENS = {
    "[PAD]":  0,
    "[UNK]":  1,
    "[BOS]":  2,
    "[EOS]":  3,
    "[MASK]": 4,
}
MASK_TOKEN_ID = 4
N_SPECIAL = len(SPECIAL_TOKENS)


class CharTokenizer:
    """
    Tokenizador em nível de caractere.
    Constrói o vocabulário a partir de um corpus de texto.
    """

    def __init__(self):
        self.char2id: dict[str, int] = {}
        self.id2char: dict[int, str] = {}
        self.vocab_size: int = 0

        # Adicionar tokens especiais
        for token, idx in SPECIAL_TOKENS.items():
            self.char2id[token] = idx
            self.id2char[idx] = token

    def build_vocab(self, texts: list[str]) -> None:
        """
        Constrói o vocabulário a partir de uma lista de textos.
        Adiciona todos os caracteres únicos encontrados.
        """
        unique_chars = set()
        for text in texts:
            unique_chars.update(text)

        # Ordenar para determinismo
        for char in sorted(unique_chars):
            if char not in self.char2id:
                idx = N_SPECIAL + len(self.char2id) - N_SPECIAL
                idx = len(self.char2id)
                self.char2id[char] = idx
                self.id2char[idx] = char

        self.vocab_size = len(self.char2id)
        print(f"✅ Vocabulário construído: {self.vocab_size} tokens")
        print(f"   Especiais: {N_SPECIAL} | Chars: {self.vocab_size - N_SPECIAL}")

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        """
        Converte texto → lista de IDs.

        Args:
            text: texto para tokenizar
            add_special: se True, adiciona [BOS] e [EOS]

        Returns:
            Lista de IDs de tokens
        """
        ids = [self.char2id.get(c, SPECIAL_TOKENS["[UNK]"]) for c in text]

        if add_special:
            ids = [SPECIAL_TOKENS["[BOS]"]] + ids + [SPECIAL_TOKENS["[EOS]"]]

        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        Converte lista de IDs → texto.

        Args:
            ids: lista de IDs de tokens
            skip_special: se True, ignora tokens especiais

        Returns:
            Texto decodificado
        """
        special_ids = set(SPECIAL_TOKENS.values())
        chars = []

        for id_ in ids:
            if skip_special and id_ in special_ids:
                continue
            char = self.id2char.get(id_, "[UNK]")
            if char == "[MASK]":
                chars.append("▓")  # Visual para tokens mascarados
            else:
                chars.append(char)

        return "".join(chars)

    def pad(self, ids: list[int], max_len: int, pad_right: bool = True) -> list[int]:
        """Padding de sequência para max_len."""
        if len(ids) >= max_len:
            return ids[:max_len]

        pad = [SPECIAL_TOKENS["[PAD]"]] * (max_len - len(ids))
        return ids + pad if pad_right else pad + ids

    def save(self, path: Union[str, Path]) -> None:
        """Salva o vocabulário em JSON."""
        path = Path(path)
        data = {
            "char2id": self.char2id,
            "vocab_size": self.vocab_size,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"💾 Tokenizador salvo em {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CharTokenizer":
        """Carrega vocabulário de JSON."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        tok = cls()
        tok.char2id = data["char2id"]
        tok.id2char = {int(v): k for k, v in data["char2id"].items()}
        tok.vocab_size = data["vocab_size"]

        print(f"✅ Tokenizador carregado: {tok.vocab_size} tokens")
        return tok


# ──────────────────────────────────────────────────────────────
#  Teste rápido
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    corpus = [
        "O gato preto pulou sobre o muro.",
        "A inteligência artificial vai mudar o mundo.",
        "Frankenstein construiu um monstro de partes.",
        "Difusão discreta é o futuro dos LLMs.",
        "Haskell, Julia, C e Python — quatro linguagens, um modelo.",
    ]

    tok = CharTokenizer()
    tok.build_vocab(corpus)

    test_text = "O gato preto"
    ids = tok.encode(test_text)
    decoded = tok.decode(ids)

    print(f"\n📝 Texto:    '{test_text}'")
    print(f"🔢 IDs:      {ids}")
    print(f"📖 Decoded:  '{decoded}'")

    # Simular mascaramento
    import random
    masked = [MASK_TOKEN_ID if random.random() < 0.5 else id_ for id_ in ids]
    print(f"\n🎭 Mascarado 50%: {tok.decode(masked, skip_special=False)}")
