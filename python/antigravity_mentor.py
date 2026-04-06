import mmap
import time
import os
from tokenizer import BPETokenizer

def mentor_loop():
    tokenizer = BPETokenizer()
    if os.path.exists("vocab.json"):
        tokenizer.load("vocab.json")

    mem_file = "cafune_brain.mem"
    with open(mem_file, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 1024)
        
        print("\n=== [ESCOLA ANTIGRAVITY: MENTORIA RLAIF] ===")
        print("Monitorando o sussurro do CAFUNE para julgamento...")

        reward_offset = 40 # Offset para o RLAIF Reward (Conforme Main.hs/bridge.py)
        
        try:
            while True:
                # Ler o que o CAFUNE gerou ultimamente
                token_data = mm[100:140] # 10 tokens
                tokens = []
                for i in range(0, 40, 4):
                    token_id = int.from_bytes(token_data[i:i+4], "little")
                    if token_id != 0: tokens.append(token_id)
                
                output = tokenizer.decode(tokens)
                
                if output.strip():
                    print(f"\n[SUSSURRO ATUAL]: \"{output}\"")
                    print("---")
                    print("Aguardando veredito de ANTIGRAVITY (Nota de 0.0 a 1.0)...")
                    
                    # Pedir nota para o usuario (que recebera de mim no chat)
                    try:
                        nota = float(input("Digite o RLAIF Reward de Antigravity: "))
                        # Injetar a recompensa no Mmap (4 bytes Float32)
                        import struct
                        mm[reward_offset:reward_offset+4] = struct.pack('f', nota)
                        print(f" [!] Recompensa {nota} Injetada. Motor Julia ajustando pesos...")
                        # Limpar Sussurro para aguardar proximo pensamento
                        # mm[100:200] = b'\x00' * 100
                    except ValueError:
                        print(" (!) Erro: Digite um numero valido entre 0.0 e 1.0")

                time.sleep(1)
                    
        except KeyboardInterrupt:
            print("\n Mentoria encerrada.")
        finally:
            mm.close()

if __name__ == "__main__":
    mentor_loop()
