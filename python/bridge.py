import os
import mmap
import time
import logging
from filelock import FileLock, Timeout

MEM_FILE = os.path.join(os.path.dirname(__file__), "cafune_brain.mem")
LOCK_FILE = MEM_FILE + ".lock"

logger = logging.getLogger(__name__)


class BridgeTimeoutError(Exception):
    pass


class CAFUNEBridge:
    def __init__(self):
        # Cria arquivo mem se não existir (1024 bytes)
        if not os.path.exists(MEM_FILE):
            with open(MEM_FILE, "wb") as f:
                f.write(b'\0' * 1024)
        print("🚀 [Silicon Bridge] Ponte de Memória Compartilhada (MMap) ativada!")

    def generate_response(self, prompt):
        try:
            with FileLock(LOCK_FILE, timeout=10):
                with open(MEM_FILE, "r+b") as f:
                    mm = mmap.mmap(f.fileno(), 0)

                    # Zero out old prompt (offsets 600..1000)
                    mm[600:1000] = b'\0' * 400

                    # Write prompt encoded
                    enc = prompt.encode('utf-8')[:399]
                    mm[600:600+len(enc)] = enc

                    # Enviar gatilho
                    mm[0] = 0x01

                    # Aguardar retorno (quando CmdID volta a 0x00)
                    timeout = 120  # O motor daemon tem JIT de 90s na primeira vez
                    start = time.time()
                    while mm[0] != 0x00:
                        time.sleep(0.1)
                        if time.time() - start > timeout:
                            mm.close()
                            raise BridgeTimeoutError("Daemon não respondeu em 120s. Engine ligado?")

                    # Ler resposta (offsets 200..600)
                    res_bytes = mm[200:600]
                    end_idx = res_bytes.find(b'\x00')
                    if end_idx != -1:
                        res_bytes = res_bytes[:end_idx]

                    mm.close()
                    return res_bytes.decode('utf-8')

        except Timeout:
            logger.error("Não foi possível adquirir lock do mmap em 10s. Outro processo está usando o bridge?")
            return "[Silicon Bridge Error] Recurso ocupado — tente novamente."
        except BridgeTimeoutError as e:
            logger.error("Bridge timeout: %s", e)
            return f"[Silicon Bridge Error] {e}"
        except OSError as e:
            logger.error("Erro de I/O no mmap: %s", e)
            return f"[Silicon Bridge Error] Falha de I/O: {e}"
        except UnicodeDecodeError as e:
            logger.error("Resposta do engine não é UTF-8 válido: %s", e)
            return "[Silicon Bridge Error] Resposta corrompida."


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bridge = CAFUNEBridge()
    print(bridge.generate_response("Quem é vc?"))
