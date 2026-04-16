import mmap
import time
import os
import struct
import sys

# Força stdout UTF-8 no Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def gemini_teacher_loop():
    mem_file = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "cafune_brain.mem"))

    # Pré-carrega modelos Flair antes de abrir o mmap (pode demorar na 1ª vez)
    print("\n=== [PROFESSOR RLAIF — MNS v2 com Flair] ===")
    print("Carregando modelos linguísticos...")
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(__file__))
        from mns_flair import compute_mns_flair, _load_models
        _load_models()  # dispara download/cache dos modelos Flair
        use_flair = True
        print("[OK] Flair pronto — sentiment + POS multilingual")
    except Exception as e:
        use_flair = False
        print(f"[WARN] Flair indisponível ({e}) — usando mns_local")

    with open(mem_file, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 2048)
        print("Aguardando handshake do Julia (0x03)...\n")

        # Layout de offsets (0-based):
        #   0      → CmdID handshake: 0x03=aguarda avaliação, 0x04=avaliação pronta
        #   40-43  → MNS score principal (float32)  ← teacher escreve aqui
        #   44-47  → MNS score secundário (float32)
        #   200-599→ output gerado pelo CAFUNE
        #   600-999→ contexto/prompt do epoch
        #  1001-1399→ suggestion do teacher (str)
        #  1401-1799→ reason do teacher (str)
        MNS_OFFSET       = 40
        MNS_LOCAL_OFFSET = 44
        HANDSHAKE_OFFSET = 0

        try:
            while True:
                # Espera sinal 0x03 do Julia (novo output pronto)
                if mm[HANDSHAKE_OFFSET] != 0x03:
                    time.sleep(0.5)
                    continue

                output = mm[200:600].split(b'\x00')[0].decode("utf-8", errors="ignore")
                prompt_text_raw = mm[600:1000].split(b'\x00')[0].decode("utf-8", errors="ignore")

                if not output.strip():
                    mm[HANDSHAKE_OFFSET] = 0x04  # libera mesmo sem texto
                    continue

                print(f"\n[TEXTO DO ALUNO]: \"{output[:80]}\"")

                try:
                    score = 0.5
                    if use_flair:
                        from mns_flair import compute_mns_flair
                        score, d_sent, d_gram, d_cov = compute_mns_flair(prompt_text_raw, output)
                        print(f" [Flair] MNS={score:.3f} | sent={d_sent:.3f} gram={d_gram:.3f} cov={d_cov:.3f}")
                    else:
                        from mns_local import compute_mns
                        score, d_f, d_t = compute_mns(prompt_text_raw, output)
                        print(f" [local] MNS={score:.3f} | D_f={d_f:.3f} D_t={d_t:.3f}")

                    score_clamped = max(0.0, min(1.0, score))
                    mm[MNS_OFFSET:MNS_OFFSET+4]             = struct.pack('f', float(score_clamped))
                    mm[MNS_LOCAL_OFFSET:MNS_LOCAL_OFFSET+4] = struct.pack('f', float(score_clamped))

                except Exception as e:
                    print(f" Erro na avaliação: {e}")

                finally:
                    # Sinaliza Julia que avaliação foi concluída
                    mm[HANDSHAKE_OFFSET] = 0x04
                    print(f" [v] Handshake 0x04 → Julia liberado")

        except KeyboardInterrupt:
            print("\n Mentoria encerrada.")
        finally:
            mm.close()


if __name__ == "__main__":
    gemini_teacher_loop()
