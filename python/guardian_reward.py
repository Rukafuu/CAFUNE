"""
guardian_reward.py — Integração do Guardian do Raegis com o loop CAFUNE

Treina o Guardian (autoencoder TF ou IsolationForest sklearn) com
neural_history.jsonl — respostas não-flagadas = "normal" — e usa-o
para penalizar respostas anômalas.

Fluxo dentro do loop de treinamento:
    1. Carrega neural_history.jsonl (escrito pelo raegis_sentinel)
    2. Treina o Guardian com entradas não flagadas (baseline do modelo)
    3. Avalia a resposta atual → anomaly_score ∈ [0, 1]
    4. Escreve penalty em mmap offset 52 (float32)

Offset 52 é lido pelo main_training.jl e somado às penalidades existentes.

Uso standalone:
    python python/guardian_reward.py
"""

import os
import sys
import json
import mmap
import struct
import time
import logging

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger(__name__)

# Caminhos corretos — raiz do projeto
MEM_FILE        = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "cafune_brain.mem"))
HISTORY_FILE    = os.path.normpath(os.path.join(os.path.dirname(__file__), "neural_history.jsonl"))
GUARDIAN_OFFSET = 52   # float32 — anomaly penalty
MIN_HISTORY     = 10   # mínimo para treinar


def load_history() -> list[dict]:
    """Carrega neural_history.jsonl como lista de dicts."""
    if not os.path.exists(HISTORY_FILE):
        return []
    rows = []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


class LocalGuardian:
    """
    Detector de anomalias local usando IsolationForest + TF-IDF.
    Substitui raegis.core.guardian sem dependência externa.
    """

    def __init__(self):
        from sklearn.ensemble import IsolationForest
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=500)
        self.model      = IsolationForest(contamination=0.1, random_state=42)
        self.fitted     = False

    def fit(self, texts: list[str]):
        if len(texts) < MIN_HISTORY:
            return
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X)
        self.fitted = True

    def anomaly_score(self, text: str) -> float:
        """Retorna penalidade ∈ [0.0, 0.5]. 0 se normal."""
        if not self.fitted or not text.strip():
            return 0.0
        X = self.vectorizer.transform([text])
        # score_samples: mais negativo = mais anômalo
        raw = self.model.score_samples(X)[0]
        is_anomaly = self.model.predict(X)[0] == -1
        if not is_anomaly:
            return 0.0
        # Normaliza raw score para [0, 0.5]
        penalty = min(abs(raw) / (abs(raw) + 1.0), 0.5)
        return round(penalty, 4)


def guardian_loop():
    if not os.path.exists(MEM_FILE):
        logger.error("[Guardian] cafune_brain.mem nao encontrado em: %s", MEM_FILE)
        return

    logger.info("=== [GUARDIAN: DETECTOR DE ANOMALIAS — IsolationForest local] ===")

    guardian      = LocalGuardian()
    retrain_every = 20
    eval_count    = 0
    last_seen_ts  = 0.0

    with open(MEM_FILE, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 2048)
        try:
            while True:
                # Re-treinar periodicamente com histórico do Raegis
                if eval_count % retrain_every == 0:
                    history = load_history()
                    normal_texts = [
                        h["response"] for h in history
                        if not h.get("flagged", False) and h.get("response", "").strip()
                    ]
                    if len(normal_texts) >= MIN_HISTORY:
                        guardian.fit(normal_texts)
                        logger.info("[Guardian] Treinado com %d respostas normais.", len(normal_texts))
                    else:
                        logger.info("[Guardian] Historico insuficiente (%d/%d) — aguardando...",
                                    len(normal_texts), MIN_HISTORY)

                # Detecta novo output via timestamp (não conflita com handshake mm[0])
                current_ts = struct.unpack('d', mm[20:28])[0]
                response   = mm[200:600].split(b'\x00')[0].decode("utf-8", errors="ignore")

                if response.strip() and current_ts != last_seen_ts:
                    last_seen_ts = current_ts
                    penalty = guardian.anomaly_score(response)
                    mm[GUARDIAN_OFFSET:GUARDIAN_OFFSET+4] = struct.pack('f', penalty)

                    if penalty > 0.0:
                        logger.info("[Guardian] ANOMALIA | penalty=%.3f | \"%s\"",
                                    penalty, response[:60])

                    eval_count += 1

                time.sleep(2.0)

        except KeyboardInterrupt:
            logger.info("[Guardian] Offline.")
        finally:
            mm.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    guardian_loop()
