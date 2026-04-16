"""
mns_flair.py — Mirror Neuron Score v2 com análise linguística via Flair

Substitui a heurística de trigramas do mns_local.py por análise linguística real:

    MNS_v2 = w_sent * D_sentiment
           + w_gram * D_grammar
           + w_cov  * D_coverage

Componentes:
    D_sentiment (0-1): tom empático/positivo da resposta (Flair sentiment)
    D_grammar   (0-1): riqueza gramatical — proporção de verbos + substantivos
                       detectados via POS tagger multilingual
    D_coverage  (0-1): cobertura de palavras-chave do prompt na resposta
                       (herdado do mns_local como sinal de intenção)

Modelos usados (baixados automaticamente na primeira execução):
    - sentiment:           'sentiment' (English, aceita PT-BR com degradação leve)
    - pos-multi-fast:      'flair/pos-multi-fast' (multilingual, inclui PT)

Fallback automático para mns_local se Flair não estiver disponível.
"""

from __future__ import annotations
import logging
import re

logger = logging.getLogger(__name__)

# Pesos dos componentes
W_SENTIMENT = 0.40
W_GRAMMAR   = 0.35
W_COVERAGE  = 0.25

# POS tags que indicam conteúdo semântico real
CONTENT_TAGS = {"NN", "NNS", "NNP", "NNPS",   # substantivos
                "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  # verbos
                "JJ", "JJR", "JJS",             # adjetivos
                "RB", "RBR", "RBS"}             # advérbios

_sentiment_model = None
_pos_model       = None
_flair_ok        = False


def _load_models() -> bool:
    """Carrega modelos Flair na primeira chamada. Retorna True se OK."""
    global _sentiment_model, _pos_model, _flair_ok
    if _flair_ok:
        return True
    try:
        from flair.models import TextClassifier, SequenceTagger
        logger.info("[Flair] Carregando modelos (primeira vez pode demorar)...")
        _sentiment_model = TextClassifier.load("sentiment")
        _pos_model       = SequenceTagger.load("flair/pos-multi-fast")
        _flair_ok = True
        logger.info("[Flair] Modelos prontos — sentiment + pos-multi-fast")
        return True
    except Exception as e:
        logger.warning("[Flair] Não disponível: %s — usando mns_local como fallback", e)
        return False


def _sentiment_score(text: str) -> float:
    """Retorna score de sentimento positivo/empático (0-1)."""
    from flair.data import Sentence
    sentence = Sentence(text)
    _sentiment_model.predict(sentence)
    label = sentence.labels[0]
    # POSITIVE → score direto; NEGATIVE → invertido
    if label.value == "POSITIVE":
        return round(label.score, 4)
    else:
        return round(1.0 - label.score, 4)


def _grammar_score(text: str) -> float:
    """
    Riqueza gramatical: proporção de tokens com POS tags de conteúdo.
    Uma resposta com verbos + substantivos reais pontua mais alto.
    """
    from flair.data import Sentence
    sentence = Sentence(text)
    _pos_model.predict(sentence)
    if not sentence.tokens:
        return 0.0
    content_count = sum(
        1 for token in sentence.tokens
        if token.get_label("pos").value in CONTENT_TAGS
    )
    return round(content_count / len(sentence.tokens), 4)


def _coverage_score(prompt: str, response: str) -> float:
    """Cobertura de palavras-chave do prompt na resposta (herdado do mns_local)."""
    stopwords = {
        "o", "a", "os", "as", "um", "uma", "de", "do", "da", "dos", "das",
        "e", "é", "em", "no", "na", "nos", "nas", "para", "por", "com",
        "que", "se", "não", "me", "te", "eu", "tu", "você", "ele", "ela",
        "isso", "isto", "ao", "à", "mais", "mas", "ou", "como", "já",
    }
    def keywords(t):
        words = re.findall(r"[a-záàâãéèêíïóôõúüç]+", t.lower())
        return {w for w in words if w not in stopwords and len(w) >= 3}

    kw_p = keywords(prompt)
    kw_r = keywords(response)
    if not kw_p:
        return 0.0
    return round(len(kw_p & kw_r) / len(kw_p), 4)


def compute_mns_flair(prompt: str, response: str) -> tuple[float, float, float, float]:
    """
    Calcula MNS v2 com análise Flair.

    Returns:
        (mns, d_sentiment, d_grammar, d_coverage)

    Fallback automático para mns_local se Flair não estiver disponível.
    """
    if not prompt.strip() or not response.strip():
        return 0.0, 0.0, 0.0, 0.0

    if not _load_models():
        # Fallback para mns_local
        from mns_local import compute_mns
        mns, d_f, d_t = compute_mns(prompt, response)
        return mns, d_f, 0.0, d_t

    try:
        d_sent = _sentiment_score(response)
        d_gram = _grammar_score(response)
        d_cov  = _coverage_score(prompt, response)

        mns = (W_SENTIMENT * d_sent
             + W_GRAMMAR   * d_gram
             + W_COVERAGE  * d_cov)

        return round(mns, 4), d_sent, d_gram, d_cov

    except Exception as e:
        logger.error("[Flair] Erro no cálculo: %s — fallback mns_local", e)
        from mns_local import compute_mns
        mns, d_f, d_t = compute_mns(prompt, response)
        return mns, d_f, 0.0, d_t


if __name__ == "__main__":
    examples = [
        ("Estou me sentindo triste hoje.", "Que pena, espero que você melhore logo! Às vezes um dia difícil passa mais rápido do que parece."),
        ("Me explica difusão discreta.", "Difusão discreta mascara tokens aleatoriamente e o modelo aprende a revelar os originais gradualmente."),
        ("Olá!", "Oi!"),
        ("Como economizar dinheiro?", "O céu é verde e azul por causa dos pássaros voando alto."),
    ]
    print(f"{'Prompt':<35} {'MNS':>5} {'Sent':>5} {'Gram':>5} {'Cov':>5}")
    print("-" * 65)
    for prompt, response in examples:
        mns, d_s, d_g, d_c = compute_mns_flair(prompt, response)
        print(f"{prompt:<35} {mns:>5.3f} {d_s:>5.3f} {d_g:>5.3f} {d_c:>5.3f}")
