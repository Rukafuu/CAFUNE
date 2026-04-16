"""
gemini_teacher.py — Professor RLAIF do CAFUNE

Avalia outputs do CAFUNE usando dois sistemas complementares:

  BitNet (60%): avaliação semântica e de coerência via LLM 1-bit local
                retorna mns + suggestion + reason em JSON
                requer llama-server.exe rodando na porta 8080

  Flair  (40%): análise linguística estrutural via modelos NLP
                sentiment (tom empático) + POS grammar + keyword coverage

Score final = 0.6 * bitnet_score + 0.4 * flair_score
Fallback automático: se BitNet offline → 100% Flair; se Flair falhar → mns_local
"""

import mmap
import time
import os
import struct
import sys
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Pesos da combinação final
W_BITNET = 0.6
W_FLAIR  = 0.4


def _init_flair():
    """Carrega modelos Flair. Retorna True se ao menos sentiment carregou."""
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from mns_flair import _load_models
        return _load_models()
    except Exception as e:
        logger.warning("[Flair] Indisponível: %s", e)
        return False


def _score_bitnet(prompt: str, output: str) -> tuple[float, str, str]:
    """
    Avalia output com BitNet Teacher.
    Retorna (score, suggestion, reason) ou (None, '', '') se offline.
    """
    from bitnet_client import is_server_alive, generate_content
    if not is_server_alive():
        return None, "", ""

    prompt_eval = f"""Mentor do CAFUNE.
Prompt: "{prompt}"
Resposta atual: "{output}"
Forneça Sugestão (Gabarito) e Razão em JSON. Score 0-1 (MNS).
{{ "mns": float, "suggestion": "string", "reason": "string" }}
REPORTE APENAS O JSON."""

    resp = generate_content(
        prompt_eval,
        system_instruction="Você é um professor avaliador rigoroso. Retorne SOMENTE JSON válido.",
        as_json=True,
        temperature=0.3,
    )
    if not resp:
        return None, "", ""

    try:
        resp = resp.replace("```json", "").replace("```", "").strip()
        data = json.loads(resp)
        return float(data.get("mns", 0.5)), str(data.get("suggestion", "")), str(data.get("reason", ""))
    except Exception as e:
        logger.error("[BitNet] Parse error: %s | resp: %s", e, resp[:100])
        return None, "", ""


def _score_flair(prompt: str, output: str) -> float:
    """Avalia output com Flair. Retorna score ou None se indisponível."""
    try:
        from mns_flair import compute_mns_flair
        score, d_sent, d_gram, d_cov = compute_mns_flair(prompt, output)
        logger.info(" [Flair] MNS=%.3f | sent=%.3f gram=%.3f cov=%.3f", score, d_sent, d_gram, d_cov)
        return score
    except Exception as e:
        logger.warning("[Flair] Erro: %s — fallback mns_local", e)
        try:
            from mns_local import compute_mns
            score, d_f, d_t = compute_mns(prompt, output)
            logger.info(" [local] MNS=%.3f | D_f=%.3f D_t=%.3f", score, d_f, d_t)
            return score
        except Exception:
            return 0.5


def gemini_teacher_loop():
    mem_file = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "cafune_brain.mem"))

    print("\n=== [PROFESSOR RLAIF — BitNet + Flair] ===")
    print("Carregando modelos Flair...")
    use_flair = _init_flair()
    if use_flair:
        print("[OK] Flair pronto — sentiment + POS multilingual")
    else:
        print("[WARN] Flair offline — usando mns_local como fallback")

    print("Verificando BitNet server (127.0.0.1:8080)...")
    from bitnet_client import is_server_alive
    if is_server_alive():
        print("[OK] BitNet server ativo")
    else:
        print("[INFO] BitNet offline — será usado se subir durante o treino")

    with open(mem_file, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 2048)
        print("\nAguardando handshake do Julia (0x03)...\n")

        # Layout de offsets (0-based):
        #   0        → handshake: 0x03=novo output, 0x04=avaliação pronta
        #   40-43    → MNS score principal (float32)
        #   44-47    → MNS score secundário (float32)
        #   200-599  → output gerado pelo CAFUNE
        #   600-999  → contexto/prompt do epoch
        #   1001-1399→ suggestion do BitNet (utf-8)
        #   1401-1799→ reason do BitNet (utf-8)
        MNS_OFFSET  = 40
        MNS2_OFFSET = 44
        SUG_OFFSET  = 1001
        REA_OFFSET  = 1401

        try:
            while True:
                if mm[0] != 0x03:
                    time.sleep(0.5)
                    continue

                output     = mm[200:600].split(b'\x00')[0].decode("utf-8", errors="ignore")
                prompt_ctx = mm[600:1000].split(b'\x00')[0].decode("utf-8", errors="ignore")

                if not output.strip():
                    mm[0] = 0x04
                    continue

                print(f"\n[TEXTO DO ALUNO]: \"{output[:80]}\"")

                try:
                    # ── BitNet ────────────────────────────────────────────────
                    bitnet_score, suggestion, reason = _score_bitnet(prompt_ctx, output)
                    if bitnet_score is not None:
                        print(f" [BitNet] MNS={bitnet_score:.3f} | {reason[:60]}")
                    else:
                        print(" [BitNet] Offline — usando só Flair")

                    # ── Flair ─────────────────────────────────────────────────
                    flair_score = _score_flair(prompt_ctx, output) if use_flair else 0.5

                    # ── Combinação ────────────────────────────────────────────
                    if bitnet_score is not None:
                        final = W_BITNET * bitnet_score + W_FLAIR * flair_score
                        print(f" [FINAL] {W_BITNET}*{bitnet_score:.3f} + {W_FLAIR}*{flair_score:.3f} = {final:.3f}")
                    else:
                        final = flair_score
                        print(f" [FINAL] Flair-only = {final:.3f}")

                    final = max(0.0, min(1.0, final))
                    mm[MNS_OFFSET:MNS_OFFSET+4]   = struct.pack('f', float(final))
                    mm[MNS2_OFFSET:MNS2_OFFSET+4] = struct.pack('f', float(flair_score))

                    # Escreve suggestion e reason no mmap
                    if suggestion:
                        enc = suggestion.encode("utf-8")[:398]
                        mm[SUG_OFFSET:SUG_OFFSET+len(enc)] = enc
                        mm[SUG_OFFSET+len(enc)]            = 0x00
                    if reason:
                        enc = reason.encode("utf-8")[:398]
                        mm[REA_OFFSET:REA_OFFSET+len(enc)] = enc
                        mm[REA_OFFSET+len(enc)]            = 0x00

                except Exception as e:
                    logger.error("Erro na avaliação: %s", e)

                finally:
                    mm[0] = 0x04
                    print(" [v] Handshake 0x04 → Julia liberado")

        except KeyboardInterrupt:
            print("\n Mentoria encerrada.")
        finally:
            mm.close()


if __name__ == "__main__":
    gemini_teacher_loop()
