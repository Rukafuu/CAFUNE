import os
import mmap
import time

MEM_FILE = "cafune_brain.mem"

class CAFUNEBridge:
    def __init__(self):
        # Cria arquivo mem se não existir (1024 bytes)
        if not os.path.exists(MEM_FILE):
            with open(MEM_FILE, "wb") as f:
                f.write(b'\0' * 1024)
        print("🚀 [Silicon Bridge] Ponte de Memória Compartilhada (MMap) ativada!")

    def generate_response(self, prompt):
        try:
            with open(MEM_FILE, "r+b") as f:
                mm = mmap.mmap(f.fileno(), 0)
                
                # Zero out old prompt (offsets 600..1000)
                mm[600:1000] = b'\0' * 400
                
                # Write string encoded
                enc = prompt.encode('utf-8')[:399]
                mm[600:600+len(enc)] = enc
                
                # Enviar gatilho
                mm[0] = 0x01
                
                # Aguardar retorno (quando CmdID volta a 0x00)
                timeout = 120 # O motor daemon tem JIT de 90s na primeira vez
                start = time.time()
                while mm[0] != 0x00:
                    time.sleep(0.1)
                    if time.time() - start > timeout:
                        return "[CAFUNE] Timeout de ressonância (Daemon off?)"
                
                # Ler resposta (offsets 200..600)
                res_bytes = mm[200:600]
                end_idx = res_bytes.find(b'\x00')
                if end_idx != -1:
                    res_bytes = res_bytes[:end_idx]
                
                mm.close()
                return res_bytes.decode('utf-8')
        except Exception as e:
            return f"[Silicon Bridge Error] Falha de acoplamento: {e}"



if __name__ == "__main__":
    bridge = CAFUNEBridge()
    print(bridge.generate_response("Quem é vc?"))
