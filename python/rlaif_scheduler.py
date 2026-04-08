import mmap
import time
import os
import json
import random
import struct
from filelock import FileLock, Timeout

# Caminhos ajustados para o CWD do script (python/)
MEM_FILE     = os.path.normpath(os.path.join(os.path.dirname(__file__), "cafune_brain.mem"))
LOCK_FILE    = MEM_FILE + ".lock"
DATASET_FILE = os.path.normpath(os.path.join(os.path.dirname(__file__), "bercario_data.jsonl"))

def load_dataset():
    dataset = []
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            for line in f:
                dataset.append(json.loads(line))
    return dataset

def run_scheduler():
    dataset = load_dataset()
    if not dataset:
        print("[ERROR] Dataset 'bercario_data.jsonl' não encontrado!")
        return

    mem_file = os.path.normpath(os.path.join(os.path.dirname(__file__), "cafune_brain.mem"))
    
    if not os.path.exists(mem_file):
        print(f"[ERROR] Memoria nao encontrada. O Engine esta ligado?")
        return

    print("--- [RLAIF SCHEDULER: O BERÇÁRIO ESTÁ ONLINE] ---")
    print("Iniciando ciclo de 10 interações por hora (1 a cada 6 min)...")
    
    with open(mem_file, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 1024)

        try:
            while True:
                cmd_id = mm[0]
                if cmd_id == 0:
                    entry  = random.choice(dataset)
                    prompt = entry["prompt"]
                    intent = entry["intent"]

                    print(f"\n[RLAIF PULSE] Enviando prompt: \"{prompt}\" (Intenção: {intent})")

                    try:
                        with FileLock(LOCK_FILE, timeout=10):
                            prompt_bytes = prompt.encode("utf-8")[:399]
                            mm[600:600+len(prompt_bytes)] = prompt_bytes
                            mm[600+len(prompt_bytes):1000] = b'\x00' * (400 - len(prompt_bytes))
                            mm[0] = 0x01
                        print("[✓] Pulso enviado com lock.")
                    except Timeout:
                        print("[!] Lock ocupado — pulso adiado para o próximo ciclo.")

                    time.sleep(360)
                else:
                    print(f" [.] Engine ocupado (CmdID: {cmd_id}), aguardando...")
                    time.sleep(5)

        except KeyboardInterrupt:
            print("\nScheduler encerrado.")
        finally:
            mm.close()

if __name__ == "__main__":
    run_scheduler()
