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
    with open(mem_file, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 1024)
        print("\n=== [PROFESSOR RLAIF — MNS LOCAL (sem API)] ===")
        print("Avaliando outputs do CAFUNE com MNS local...")

        # Offsets:
        #   20-27 → timestamp de geração (float64)
        #   40-43 → MNS score principal  (float32)
        #   44-47 → MNS score secundário (float32)
        MNS_OFFSET       = 40
        MNS_LOCAL_OFFSET = 44
        TS_OFFSET        = 20

        last_seen_ts = 0.0

        try:
            while True:
                ts_bytes = mm[TS_OFFSET:TS_OFFSET+8]
                current_ts = struct.unpack('d', ts_bytes)[0]

                response_data = mm[200:599].split(b'\x00')[0]
                output = response_data.decode("utf-8", errors="ignore")

                if len(output.strip()) > 2 and current_ts != last_seen_ts:
                    prompt_text_raw = mm[600:1000].split(b'\x00')[0].decode("utf-8", errors="ignore")
                    print(f"\n[TEXTO DO ALUNO]: \"{output[:80]}\"")

                    try:
                        from mns_local import compute_mns
                        score, d_f, d_t = compute_mns(prompt_text_raw, output)
                        reason = f"D_f={d_f:.3f} D_t={d_t:.3f}"
                        print(f" [local] MNS={score:.3f} | {reason}")

                        score_clamped = max(0.0, min(1.0, score))
                        mm[MNS_OFFSET:MNS_OFFSET+4]       = struct.pack('f', float(score_clamped))
                        mm[MNS_LOCAL_OFFSET:MNS_LOCAL_OFFSET+4] = struct.pack('f', float(score_clamped))

                        last_seen_ts = current_ts

                    except Exception as e:
                        print(f" Erro na avaliação: {e}")

                time.sleep(5)

        except KeyboardInterrupt:
            print("\n Mentoria encerrada.")
        finally:
            mm.close()


if __name__ == "__main__":
    gemini_teacher_loop()
