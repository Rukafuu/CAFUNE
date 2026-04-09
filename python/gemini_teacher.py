import mmap
import time
import os
import struct
import sys
import io
from dotenv import load_dotenv
from tokenizer import BPETokenizer

# Força stdout UTF-8 no Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Carregar variáveis de ambiente — tenta ../../.env e depois ../.env
dotenv_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
if not os.path.exists(dotenv_path):
    dotenv_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path)

# Configurar API Gemini — usa google.genai (novo SDK)
api_key = os.getenv("GEMINI_API_KEY")
_client = None
_types  = None
_legacy_model = None

if api_key:
    try:
        from google import genai as google_genai
        from google.genai import types as google_types
        _client = google_genai.Client(api_key=api_key)
        _types  = google_types
        print("[OK] Gemini SDK (google.genai) com web grounding pronto.")
    except ImportError:
        import google.generativeai as genai_legacy
        genai_legacy.configure(api_key=api_key)
        _legacy_model = genai_legacy.GenerativeModel('gemini-1.5-flash')
        print("[OK] Gemini SDK legado (google.generativeai) configurado.")
else:
    print("[WARN] GEMINI_API_KEY nao encontrada. Usando MNS local como fallback.")


def call_gemini(prompt_text: str, use_grounding: bool = True) -> str:
    """Chama Gemini com ou sem web grounding. Retorna texto da resposta."""
    if _client and _types:
        config_kwargs = {"temperature": 0.2}
        if use_grounding:
            config_kwargs["tools"] = [_types.Tool(google_search=_types.GoogleSearch())]
        config = _types.GenerateContentConfig(**config_kwargs)
        resp = _client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt_text,
            config=config,
        )
        return resp.text
    elif _legacy_model:
        resp = _legacy_model.generate_content(prompt_text)
        return resp.text
    return ""

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
                    prompt_text_raw = mm[600:1000].split(b'\x00')[0].decode("utf-8", errors="ignore")
                    print(f"\n[TEXTO DO ALUNO]: \"{output[:80]}\"")
                    print("Solicitando avaliacao ao Gemini (web grounding ativo)...")

                    # Prompt de avaliação MNS com contexto web
                    eval_prompt = f"""Você é um mentor de uma IA chamada CAFUNE. Use a web para verificar se a resposta abaixo é factualmente correta e empática.

Prompt original do usuario: "{prompt_text_raw}"
Resposta do CAFUNE: "{output}"

CALCULE DOIS SCORES (0.0 a 1.0):
1. Df: Espelhamento de Forma — a resposta tem tom empático e natural?
2. Dt: Espelhamento de Intencao — a resposta captura o que o usuario realmente quer?

Se a resposta contiver informacao factualmente ERRADA (verificada via web), penalize Dt em -0.3.

Responda APENAS neste formato:
Df: 0.X
Dt: 0.X
MNS: 0.X
Razao: uma linha explicando"""

                    try:
                        import re
                        if _client or _legacy_model:
                            # Usa grounding se texto tiver mais de 15 chars (vale a pena buscar)
                            use_grounding = len(output.strip()) > 15
                            resp_text = call_gemini(eval_prompt, use_grounding=use_grounding)
                            mns_match = re.search(r"MNS:\s*(\d+\.\d+|\d+)", resp_text)
                            score = float(mns_match.group(1)) if mns_match else 0.5
                            reason_match = re.search(r"Razao:\s*(.+)", resp_text)
                            reason = reason_match.group(1).strip() if reason_match else resp_text.split('\n')[-1]
                            grounded_tag = "[WEB]" if use_grounding else "[local]"
                            print(f" {grounded_tag} MNS={score:.3f} | {reason}")
                        else:
                            # Fallback MNS local sem API
                            from mns_local import compute_mns
                            score, d_f, d_t = compute_mns(prompt_text_raw, output)
                            reason = f"MNS local D_f={d_f:.3f} D_t={d_t:.3f} (API offline)"
                            print(f" [offline] MNS={score:.3f} | {reason}")

                        # Escrever Gemini score (com grounding) em offset 40
                        score_clamped = max(0.0, min(1.0, score))
                        mm[GEMINI_OFFSET:GEMINI_OFFSET+4] = struct.pack('f', float(score_clamped))

                        # Escrever MNS local em offset 44 (sempre disponível como fallback)
                        from mns_local import compute_mns
                        _, d_f, d_t = compute_mns(prompt_text_raw, output)
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
