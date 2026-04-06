import mmap
import time
import os
from tokenizer import BPETokenizer

def demo_voice():
    # 1. Carregar o Tradutor (BPE)
    tokenizer = BPETokenizer()
    if os.path.exists("vocab.json"):
        tokenizer.load("vocab.json")
        print(" Tradutor BPE Carregado. Pronto para ouvir o CAFUNE.")
    else:
        print(" Vocabulario nao encontrado. O CAFUNE falara em numeros por enquanto.")

    # 2. Conectar ao Córtex (Mmap)
    mem_file = "cafune_brain.mem"
    if not os.path.exists(mem_file):
        # Criar se nao existir, com 1KB
        with open(mem_file, "wb") as f:
            f.write(b'\x00' * 1024)

    with open(mem_file, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 1024)
        
        print("\n=== [CAFUNE: DEMO DE GERACAO POR DIFUSAO] ===")
        print("Aguardando o orquestrador Haskell iniciar a difusao...")
        
        last_step = -1
        try:
            while True:
                # Ler metadados (offsets baseados no Orchestrator.hs)
                cmd_id = mm[0]
                step = mm[4]
                mask_ratio = int(mm[8]) / 100.0  # Simulado
                
                # Ler os primeiros 10 tokens (exemplo) do offset 100
                token_data = mm[100:200]
                tokens = []
                for i in range(0, 40, 4): # IDs de 4 bytes
                    token_id = int.from_bytes(token_data[i:i+4], "little")
                    if token_id != 0:
                        tokens.append(token_id)

                # Traduzir e Mostrar Evolucao
                if step != last_step:
                    decoded = tokenizer.decode(tokens)
                    progress_bar = "█" * (10 - int(mask_ratio * 10)) + "░" * int(mask_ratio * 10)
                    print(f"[{progress_bar}] Passo {step:02d} | CAFUNE diz: \"{decoded}\"")
                    last_step = step

                time.sleep(0.1)
                
                # Sair se completar (Passo 20 ou Mask 0)
                if step >= 20 and cmd_id == 0:
                    print("\n Geracao concluida com sucesso. O motor atingiu a pureza total.")
                    break
                    
        except KeyboardInterrupt:
            print("\n Interrompido pelo usuario. Silenciando o CAFUNE.")
        finally:
            mm.close()

if __name__ == "__main__":
    demo_voice()
