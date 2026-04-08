import mmap
import time
import os
import struct
import google.generativeai as genai
from dotenv import load_dotenv
from tokenizer import BPETokenizer

# Carregar variáveis de ambiente do arquivo .env na raiz do projeto (../../.env)
dotenv_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
load_dotenv(dotenv_path)

# Configurar API Gemini usando a chave do ambiente
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash') # Modelo Flash para estabilidade e velocidade
else:
    print("⚠️ [Mestriado] GEMINI_API_KEY não encontrada. Usando modo simulação.")
    model = None

def gemini_teacher_loop():
    tokenizer = BPETokenizer()
    vocab_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "vocab.json"))
    if os.path.exists(vocab_path):
        tokenizer.load(vocab_path)

    mem_file = os.path.normpath(os.path.join(os.path.dirname(__file__), "cafune_brain.mem"))
    with open(mem_file, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 1024)
        print("\n=== [MESTRIADO GEMINI: PROFESSOR RLAIF ATIVO (PRO)] ===")
        print("Ensinando o CAFUNE a ser um assistente natural, empático e amigável...")

        # Offsets (0-based):
        #   20-27 → timestamp de geração (float64)
        #   40-43 → Gemini MNS score     (float32)  ← este processo escreve aqui
        #   44-47 → MNS local score      (float32)  ← também escrito aqui como fallback
        GEMINI_OFFSET    = 40
        MNS_LOCAL_OFFSET = 44
        TS_OFFSET        = 20

        last_seen_ts = 0.0

        try:
            while True:
                # Verificar se houve nova geração via timestamp
                ts_bytes = mm[TS_OFFSET:TS_OFFSET+8]
                current_ts = struct.unpack('d', ts_bytes)[0]

                response_data = mm[200:599].split(b'\x00')[0]
                output = response_data.decode("utf-8", errors="ignore")

                # Só avaliar se: texto presente, engine idle, e timestamp mudou
                if len(output.strip()) > 2 and mm[0] == 0 and current_ts != last_seen_ts:
                    print(f"\n[TEXTO DO ALUNO]: \"{output}\"")
                    print("Solicitando avaliação ao Gemini-PRO (Alinhamento de Intenção)...")
                    
                    # Prompt para RLAIF focado em MNS (Mirror Neuron Score)
                    prompt = f"""
                    Você é um mentor social e neuro-bio de uma IA chamada CAFUNE.
                    Avalie o texto abaixo considerando o Índice de Neurônios Espelho (CMNI).
                    
                    CALCULE DOIS SCORES (0.0 a 1.0):
                    1. Δμ_n(D_f): Espelhamento de Forma (A resposta mimetiza o 'tom' e 'face' social do usuário de forma empática?)
                    2. Δμ_n(D_t): Espelhamento de Intenção (A rede capturou o que o usuário REALMENTE quer ou sente, além das palavras?)
                    
                    FÓRMULA FINAL: MNS = (D_f + D_t) / 2
                    
                    Texto do CAFUNE: "{output}"
                    
                    Responda APENAS com os dois números (Df e Dt) e depois o resultado final, seguido de uma breve razão.
                    Exemplo:
                    Df: 0.8
                    Dt: 0.9
                    MNS: 0.85
                    A rede percebeu a tristeza e reagiu no tom certo.
                    """
                    
                    try:
                        if model:
                            response = model.generate_content(prompt)
                            # Extrair o score MNS (procurando por "MNS: X.X")
                            import re
                            mns_match = re.search(r"MNS:\s*(\d+\.\d+|\d+)", response.text)
                            score = float(mns_match.group(1)) if mns_match else 0.5
                            reason = response.text.split('\n')[-1]
                        else:
                            # Fallback local: cálculo determinístico sem API
                            from mns_local import compute_mns
                            prompt_data = mm[600:1000].split(b'\x00')[0]
                            prompt_text = prompt_data.decode("utf-8", errors="ignore")
                            score, d_f, d_t = compute_mns(prompt_text, output)
                            reason = f"MNS local — D_f={d_f:.3f} D_t={d_t:.3f} (API offline)"
                        
                        print(f" [!] Veredito: score={score:.3f} | {reason}")

                        # Escrever Gemini score em offset 40
                        mm[GEMINI_OFFSET:GEMINI_OFFSET+4] = struct.pack('f', float(score))

                        # Escrever MNS local em offset 44 (sempre disponível)
                        from mns_local import compute_mns
                        prompt_text = mm[600:1000].split(b'\x00')[0].decode("utf-8", errors="ignore")
                        _, d_f, d_t = compute_mns(prompt_text, output)
                        mns_val = (d_f + d_t) / 2.0
                        mm[MNS_LOCAL_OFFSET:MNS_LOCAL_OFFSET+4] = struct.pack('f', float(mns_val))

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
