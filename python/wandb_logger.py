"""
wandb_logger.py — Logger do treino CAFUNE para Weights & Biases

Lê training_log.jsonl (escrito pelo Julia a cada epoch) e o mmap
(scores RLAIF em tempo real) e faz push para o wandb.

Métricas logadas por epoch:
    loss            — perda supervisionada
    best_loss       — melhor loss histórico
    mns_score       — score BitNet + Flair (offset 40)
    mns_local       — score secundário (offset 44)
    raegis_penalty  — penalidade ética (offset 48)
    guardian_penalty— penalidade anomalia (offset 52)
    ethics_flag     — flag sycophancy (offset 60)
    reward_final    — reward combinado após penalidades
    dataset_n       — tamanho do dataset
    output_sample   — amostra do texto gerado (texto livre)

Uso:
    python python/wandb_logger.py
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR   = os.path.dirname(__file__)
MEM_FILE     = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "cafune_brain.mem"))
TRAIN_LOG    = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "julia", "training_log.jsonl"))
POLL_INTERVAL = 10  # segundos entre verificações


def read_mmap_metrics(mm) -> dict:
    """Lê métricas do mmap e retorna como dict."""
    try:
        mns_score        = struct.unpack('f', mm[40:44])[0]
        mns_local        = struct.unpack('f', mm[44:48])[0]
        raegis_penalty   = struct.unpack('f', mm[48:52])[0]
        guardian_penalty = struct.unpack('f', mm[52:56])[0]
        ethics_flag      = mm[60]
        loss_val         = struct.unpack('d', mm[32:40])[0]

        # Calcula reward final igual ao Julia
        mns_score        = max(0.0, min(1.0, mns_score        if not (mns_score != mns_score)        else 0.0))
        mns_local        = max(0.0, min(1.0, mns_local        if not (mns_local != mns_local)        else 0.0))
        raegis_penalty   = max(0.0, min(1.0, raegis_penalty   if not (raegis_penalty != raegis_penalty) else 0.0))
        guardian_penalty = max(0.0, min(0.5, guardian_penalty if not (guardian_penalty != guardian_penalty) else 0.0))
        effective_raegis = raegis_penalty * 2.0 if ethics_flag == 0x01 else raegis_penalty
        alpha            = 0.7 if mns_score > 0 else 0.0
        combined         = alpha * mns_score + (1.0 - alpha) * mns_local
        reward_final     = max(0.0, combined - effective_raegis - guardian_penalty)

        output = mm[200:600].split(b'\x00')[0].decode("utf-8", errors="ignore").strip()

        return {
            "mns_score":        round(mns_score, 4),
            "mns_local":        round(mns_local, 4),
            "raegis_penalty":   round(raegis_penalty, 4),
            "guardian_penalty": round(guardian_penalty, 4),
            "ethics_flag":      int(ethics_flag),
            "reward_final":     round(reward_final, 4),
            "loss_mmap":        round(loss_val, 4) if abs(loss_val) < 100 else None,
            "output_sample":    output[:200] if output else None,
        }
    except Exception as e:
        logger.warning("Erro ao ler mmap: %s", e)
        return {}


def tail_jsonl(path: str, last_line: int) -> tuple[list[dict], int]:
    """Retorna linhas novas do JSONL desde last_line. Retorna (novas_linhas, novo_offset)."""
    if not os.path.exists(path):
        return [], last_line
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < last_line:
                continue
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries, last_line + len(entries)


def run_logger():
    import wandb

    # Carrega API key do .env
    try:
        from dotenv import load_dotenv
        dotenv_path = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".env"))
        load_dotenv(dotenv_path, override=True)
    except ImportError:
        pass

    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        logger.warning("[wandb] WANDB_API_KEY não encontrada no .env — tentando login salvo")

    # Inicializa run wandb
    run = wandb.init(
        project="cafune",
        name=f"cafune-run-{int(time.time())}",
        config={
            "model":       "BidirectionalTransformer",
            "vocab_size":  461,
            "d_model":     256,
            "n_heads":     8,
            "n_layers":    6,
            "diffusion":   "LLaDA masked discrete",
            "optimizer":   "Adam",
            "rlaif":       "BitNet 60% + Flair 40%",
            "sentinel":    "Raegis heurístico + IsolationForest",
        },
        resume="allow",
    )
    logger.info("[wandb] Run iniciado: %s", run.url)

    if not os.path.exists(MEM_FILE):
        logger.error("[wandb] cafune_brain.mem não encontrado: %s", MEM_FILE)
        return

    # Começa do fim do arquivo — ignora histórico já logado
    if os.path.exists(TRAIN_LOG):
        with open(TRAIN_LOG, "r", encoding="utf-8") as f:
            last_line = sum(1 for line in f if line.strip())
    else:
        last_line = 0
    logger.info("[wandb] Iniciando a partir da linha %d do training_log", last_line)

    with open(MEM_FILE, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 2048)
        logger.info("[wandb] Monitorando training_log.jsonl + mmap...")

        try:
            while True:
                new_entries, last_line = tail_jsonl(TRAIN_LOG, last_line)

                for entry in new_entries:
                    mmap_metrics = read_mmap_metrics(mm)

                    log_data = {
                        "epoch":            entry.get("epoch"),
                        "loss":             entry.get("loss"),
                        "best_loss":        entry.get("best_loss"),
                        "dataset_n":        entry.get("dataset_n"),
                        **mmap_metrics,
                    }

                    # Remove Nones
                    log_data = {k: v for k, v in log_data.items() if v is not None}

                    # Output como texto livre (wandb.Html ou só string)
                    if "output_sample" in log_data:
                        sample = log_data.pop("output_sample")
                        log_data["output_sample"] = wandb.Html(
                            f"<pre style='font-family:monospace'>{sample}</pre>"
                        )

                    wandb.log(log_data, step=entry.get("epoch"))
                    logger.info("[wandb] Epoch %d logado | loss=%.4f reward=%.3f",
                                entry.get("epoch", 0),
                                entry.get("loss", 0),
                                log_data.get("reward_final", 0))

                time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            logger.info("[wandb] Encerrando...")
        finally:
            mm.close()
            wandb.finish()


if __name__ == "__main__":
    run_logger()
