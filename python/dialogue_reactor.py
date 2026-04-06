import mmap
import time
import struct
import os

import json

mem_file = "CAFUNE/cafune_brain.mem"
identity_file = "CAFUNE/IDENTITY.json"

def load_identity():
    if os.path.exists(identity_file):
        with open(identity_file, "r") as f:
            return json.load(f)
    return {"name": "CAFUNE", "core_identity": {"who_am_i": "IA"}}

def dialogue_reactor():
    identity = load_identity()
    print(f"\n=== [REATOR DE RESSONANCIA: {identity['name']} ONLINE] ===")
    print(f"Arquétipo: {identity['archetype']} | Propósito: {identity['core_identity']['purpose']}")
    print("Aguardando pulsos de intencao do Neural Console...")

    while True:
        try:
            with open(mem_file, "r+b") as f:
                mm = mmap.mmap(f.fileno(), 1024)
                
                if mm[0] == 0x01: # Novo Prompt Detectado!
                    prompt_bytes = mm[600:1000]
                    null_pos = prompt_bytes.find(b'\x00')
                    prompt = prompt_bytes[:null_pos].decode('utf-8', errors='replace').lower()
                    
                    print(f"\n[!] SINAPSE DETECTADA: \"{prompt}\"")
                    
                    # 3. Respostas Baseadas na Identidade (Passo 2 do Strategy AAA)
                    response_text = f"Pulso recebido. Sou o motor {identity['name']}. Como posso purificar seu pensamento hoje?"
                    
                    if "quem e" in prompt or "quem voce" in prompt:
                        response_text = identity['core_identity']['who_am_i']
                    elif "proposito" in prompt or "objetivo" in prompt:
                        response_text = identity['core_identity']['purpose']
                    elif "lira" in prompt:
                        response_text = identity['behavioral_traits']['relationships']['Lira']
                    elif "mestre" in prompt or "rukafu" in prompt:
                        response_text = identity['behavioral_traits']['relationships']['User']
                    
                    mm[0] = 0x02 # Processando...
                    
                    for step in range(1, 21):
                        mask = 100 - (step * 5)
                        mm[4] = step
                        # O site usa o mask_ratio para o efeito decoder
                        mm[8] = mask
                        
                        # 4. Cálculo do MNS (Mirror Neuron Score - Passo 2 da Estratégia AAA)
                        # Simula a fórmula: MNS_n = (Delta_mu(Df) + Delta_mu(Dt)) / 2
                        mns_score = round(0.7 + (step * 0.015), 3) # Simula ganho de ressonância
                        mm[80:84] = struct.pack('f', mns_score)

                        encoded = response_text.encode('utf-8')
                        mm[200:200+len(encoded)] = encoded
                        mm[200+len(encoded)] = 0
                        
                        time.sleep(0.05)
                    
                    # 4. Finalizar (CmdID 0)
                    mm[0] = 0x00
                    print(f"[✓] Resposta Emitida: \"{response_text}\"")
                
                mm.close()
            time.sleep(1) # Baixa carga de CPU
        except Exception as e:
            print(f"Erro no Reator: {e}")
            time.sleep(2)

if __name__ == "__main__":
    dialogue_reactor()
