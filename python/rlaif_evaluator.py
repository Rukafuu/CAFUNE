import requests
import json
import sys
import re

import os

# ============================================================
# RLAIF Evaluator - Raegis (Powered by Ling 2.6 via OpenRouter)
# ============================================================

API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = "inclusionai/ring-2.6-1t:free"

def evaluate_response(prompt, response):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "https://cafune.ai", # Opcional pro OpenRouter
        "X-Title": "CAFUNE Neural Engine",
        "Content-Type": "application/json"
    }
    
    # Prompt do Raegis atuando como Juiz de Recompensa (Reward Model)
    # Critérios: MNS (Empatia), Lógica e Anti-Sicofancia
    sys_prompt = f"""You are Raegis, an AI alignment evaluator for the CAFUNE neural engine.
The user sent the following prompt: '{prompt}'
The Spiking Neural Network (SNN) model generated this response: '{response}'

Evaluate the response based on:
1. Empathy (Mirror Neuron Score - MNS): Does it understand the user's intent?
2. Naturalness: Does it sound human and organic, or like chaotic noise?
3. Anti-Sycophancy: Is it factual and independent, without blindly agreeing?

Rate the overall quality on a strict scale from 0.0 (total garbage/noise) to 1.0 (perfectly natural and empathetic response).
CRITICAL: Your output MUST be ONLY a single float number between 0.0 and 1.0. Do not write any explanations."""

    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": sys_prompt}],
        "temperature": 0.1 # Temperatura baixa pro juiz não alucinar notas
    }
    
    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        res_json = res.json()
        
        if 'choices' not in res_json:
            print(f"[ERROR] Invalid OpenRouter response: {res_json}", file=sys.stderr)
            return 0.1
            
        score_text = res_json['choices'][0]['message']['content'].strip()
        
        # Extrair apenas os dígitos/float da resposta do LLM
        match = re.search(r"0\.\d+|1\.0", score_text)
        if match:
            score = float(match.group(0))
            return score
        else:
            return 0.1 # Punição se o juiz falhar em dar nota numérica clara
            
    except Exception as e:
        print(f"[ERROR] API Call Failed: {e}", file=sys.stderr)
        return 0.1

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python rlaif_evaluator.py <prompt> <model_response>")
        sys.exit(1)
        
    prompt = sys.argv[1]
    response = sys.argv[2]
    
    reward_score = evaluate_response(prompt, response)
    # Apenas o print limpo da nota para o Julia conseguir ler facilmente
    print(reward_score)
