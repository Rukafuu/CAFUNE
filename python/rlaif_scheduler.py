import mmap
import time
import os
import json
import random
import struct

# Caminhos ajustados para o CWD do script (python/)
MEM_FILE = os.path.normpath(os.path.join(os.path.dirname(__file__), "cafune_brain.mem"))
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
                # 1. Esperar o Engine estar livre (CmdID == 0)
                cmd_id = mm[0]
                if cmd_id == 0:
                    entry = random.choice(dataset)
                    prompt = entry["prompt"]
                    intent = entry["intent"]
                    
                    print(f"\n[RLAIF PULSE] Enviando prompt: \"{prompt}\" (Intenção: {intent})")
                    
                    # 2. Escrever o prompt na memória (Offset 600 em 0-based index)
                    prompt_bytes = prompt.encode("utf-8")[:399]
                    mm[600:600+len(prompt_bytes)] = prompt_bytes
                    mm[600+len(prompt_bytes):1000] = b'\x00' * (400 - len(prompt_bytes))
                    
                    # 3. Disparar o pulso (CmdID = 1)
                    mm[0] = 0x01
                    print("[✓] Pulso enviado. O Engine deve começar a processar...")
                    
                    # Esperar o ciclo de 6 minutos
                    time.sleep(360) 
                else:
                    # Engine ocupado (CmdID 1 ou 2), aguarde o próximo ciclo
                    print(f" [.] Engine ocupado ou não-zero (CmdID: {cmd_id}), aguardando...")
                    time.sleep(5)
                    
        except KeyboardInterrupt:
            print("\nScheduler encerrado.")
        finally:
            mm.close()

if __name__ == "__main__":
    run_scheduler()
