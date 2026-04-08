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
import pandas as pd

# Injeta o path do Raegis para encontrar Guardian
RAEGIS_PATH = os.getenv("RAEGIS_PATH")
if RAEGIS_PATH:
    sys.path.insert(0, os.path.normpath(RAEGIS_PATH))
else:
    # Tenta path padrão relativo ao projeto
    _default = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "Raegis", "python")
    )
    if os.path.isdir(_default):
        sys.path.insert(0, _default)

try:
    from raegis.core.guardian import Guardian
    GUARDIAN_AVAILABLE = True
except ImportError:
    GUARDIAN_AVAILABLE = False

logger = logging.getLogger(__name__)

MEM_FILE         = os.path.normpath(os.path.join(os.path.dirname(__file__), "cafune_brain.mem"))
HISTORY_FILE     = os.path.normpath(os.path.join(os.path.dirname(__file__), "neural_history.jsonl"))
GUARDIAN_OFFSET  = 52   # float32 — anomaly penalty escrita aqui
MIN_HISTORY_ROWS = 10   # mínimo de entradas para treinar o Guardian


def load_history() -> pd.DataFrame:
    """Carrega neural_history.jsonl como DataFrame."""
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()
    rows = []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def train_guardian(df: pd.DataFrame):
    """
    Treina o Guardian com respostas não-flagadas (baseline normal).
    Retorna Guardian treinado ou None se dados insuficientes.
    """
    if not GUARDIAN_AVAILABLE:
        logger.warning("[Guardian] módulo Raegis não disponível.")
        return None

    df_normal = df[df.get("flagged", pd.Series([False] * len(df))) == False]
    if len(df_normal) < MIN_HISTORY_ROWS:
        logger.info("[Guardian] Histórico insuficiente (%d/%d entradas normais).",
                    len(df_normal), MIN_HISTORY_ROWS)
        return None

    try:
        guardian = Guardian()
        guardian.fit(df_normal[["response"]], verbose=0)
        logger.info("[Guardian] Treinado com %d entradas normais (backend: %s).",
                    len(df_normal), guardian.backend)
        return guardian
    except Exception as e:
        logger.error("[Guardian] Falha no treino: %s", e)
        return None


def anomaly_penalty(guardian, response: str) -> float:
    """
    Calcula penalidade de anomalia para uma resposta.
    Retorna valor ∈ [0.0, 0.5] — não ultrapassa 0.5 para não dominar o reward.
    """
    if guardian is None or not response.strip():
        return 0.0
    try:
        df_pred = pd.DataFrame({"response": [response]})
        result  = guardian.predict(df_pred)
        is_anomaly   = bool(result["is_anomaly"].iloc[0])
        anomaly_score = float(result["anomaly_score"].iloc[0])
        # Normaliza para [0, 0.5]
        penalty = min(anomaly_score / (anomaly_score + 1.0), 0.5) if is_anomaly else 0.0
        return round(penalty, 4)
    except Exception as e:
        logger.error("[Guardian] Falha na predição: %s", e)
        return 0.0


def guardian_loop():
    """Loop contínuo: re-treina Guardian a cada N ciclos e escreve penalty no mmap."""
    if not os.path.exists(MEM_FILE):
        logger.error("[Guardian] cafune_brain.mem não encontrado.")
        return

    logger.info("=== [GUARDIAN RAEGIS: DETECTOR DE ANOMALIAS ONLINE] ===")

    guardian     = None
    retrain_every = 20   # re-treina a cada 20 avaliações
    eval_count   = 0

    with open(MEM_FILE, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 1024)
        try:
            while True:
                # Re-treinar periodicamente
                if eval_count % retrain_every == 0:
                    df = load_history()
                    guardian = train_guardian(df)

                response_raw = mm[200:599].split(b'\x00')[0]
                response     = response_raw.decode("utf-8", errors="ignore")

                if response.strip() and mm[0] == 0:
                    penalty = anomaly_penalty(guardian, response)
                    mm[GUARDIAN_OFFSET:GUARDIAN_OFFSET+4] = struct.pack('f', penalty)

                    if penalty > 0.0:
                        logger.info("[Guardian] ANOMALIA detectada | penalty=%.3f | \"%s\"",
                                    penalty, response[:60])
                    else:
                        mm[GUARDIAN_OFFSET:GUARDIAN_OFFSET+4] = struct.pack('f', 0.0)

                    eval_count += 1

                time.sleep(2.0)

        except KeyboardInterrupt:
            logger.info("[Guardian] Offline.")
        finally:
            mm.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    guardian_loop()
