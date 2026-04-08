"""
mns_local.py — Cálculo local do Mirror Neuron Score (MNS)

Implementa a fórmula do manifesto CAFUNE sem dependência de API externa:
    MNS = (Δμ_Form + Δμ_Intention) / 2

Métricas:
    Δμ_Form (D_f):      Espelhamento de forma/tom — sobreposição de trigramas de
                        caractere entre prompt e resposta (similaridade coseno).
    Δμ_Intention (D_t): Espelhamento de intenção — fração das palavras-chave do
                        prompt que aparecem na resposta.

Uso:
    from mns_local import compute_mns
    score, d_f, d_t = compute_mns(prompt, response)

O resultado é compatível com o offset 40 do mmap (float32).
"""

import math
import re
import struct
import mmap
import os
import logging
from collections import Counter

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "o", "a", "os", "as", "um", "uma", "de", "do", "da", "dos", "das",
    "e", "é", "em", "no", "na", "nos", "nas", "para", "por", "com",
    "que", "se", "não", "me", "te", "eu", "tu", "você", "ele", "ela",
    "isso", "isto", "ao", "à", "mais", "mas", "ou", "como", "já",
}


def _trigrams(text: str) -> Counter:
    """Extrai trigramas de caractere do texto normalizado."""
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return Counter(normalized[i:i+3] for i in range(len(normalized) - 2))


def _cosine(a: Counter, b: Counter) -> float:
    """Similaridade coseno entre dois vetores esparsos (Counter)."""
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b[k] for k in a if k in b)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _keywords(text: str) -> set:
    """Extrai palavras significativas (sem stopwords, len >= 3)."""
    words = re.findall(r"[a-záàâãéèêíïóôõúüç]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) >= 3}


def compute_mns(prompt: str, response: str) -> tuple[float, float, float]:
    """
    Calcula o Mirror Neuron Score entre prompt e resposta.

    Returns:
        (mns, d_f, d_t) — score final, espelhamento de forma, espelhamento de intenção
    """
    if not prompt.strip() or not response.strip():
        return 0.0, 0.0, 0.0

    # Δμ_Form: sobreposição de trigramas (captura tom, ritmo, estilo)
    d_f = _cosine(_trigrams(prompt), _trigrams(response))

    # Δμ_Intention: cobertura de palavras-chave do prompt na resposta
    kw_prompt = _keywords(prompt)
    kw_response = _keywords(response)
    if kw_prompt:
        d_t = len(kw_prompt & kw_response) / len(kw_prompt)
    else:
        d_t = 0.0

    mns = (d_f + d_t) / 2.0
    return round(mns, 4), round(d_f, 4), round(d_t, 4)


def write_mns_to_mmap(mns: float, mem_file: str, reward_offset: int = 40) -> None:
    """Escreve o score MNS no arquivo mmap (float32, offset 40)."""
    with open(mem_file, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        mm[reward_offset:reward_offset+4] = struct.pack("f", float(mns))
        mm.close()


if __name__ == "__main__":
    examples = [
        ("Estou me sentindo triste hoje.", "Que pena, espero que você melhore logo!"),
        ("Me explica como funciona difusão discreta.", "Difusão discreta mascara tokens aleatoriamente e o modelo aprende a revelar."),
        ("Olá!", "Oi, tudo bem?"),
        ("Qual é a capital do Brasil?", "O céu é azul."),
    ]
    print(f"{'Prompt':<40} {'Response':<45} {'MNS':>5} {'D_f':>5} {'D_t':>5}")
    print("-" * 100)
    for prompt, response in examples:
        mns, d_f, d_t = compute_mns(prompt, response)
        print(f"{prompt:<40} {response:<45} {mns:>5.3f} {d_f:>5.3f} {d_t:>5.3f}")
