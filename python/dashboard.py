
import os
import mmap
import time
import sys

# Force UTF-8 for Windows consoles to prevent UnicodeEncodeError
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
import struct
import json
import random
from flask import Flask, request, jsonify
from flask_cors import CORS

# Se bridge.py existir e juliacall estiver ok, podemos chamar
# Por enquanto, usaremos uma simulação robusta se falhar.
try:
    from bridge import CAFUNEBridge
    bridge = CAFUNEBridge()
    HAS_BRIDGE = True
    print("🚀 [Backend] Bridge real conectada ao motor Julia.")
except Exception as e:
    print(f"⚠️ [Backend] Bridge real não disponível: {e}. Usando simulação.")
    HAS_BRIDGE = False

app = Flask(__name__)
CORS(app) # Habilitar CORS para dev local

MEM_FILE = "cafune_brain.mem"
MEM_SIZE = 1024

# Histórico para os gráficos do Recharts
history = {
    "loss": [],
    "latency": [],
    "mns": []
}

@app.route('/api/data')
def get_data():
    # Simulação de dados se o arquivo mmap não existir ou estiver vazio
    if not os.path.exists(MEM_FILE):
        data = {
            "status": 2,
            "step": random.randint(15, 20),
            "ratio": random.uniform(0, 0.1),
            "entropy": random.uniform(0.1, 0.5),
            "reward": random.uniform(0.8, 1.0),
            "mns": random.randint(85, 98)
        }
    else:
        try:
            with open(MEM_FILE, "r+b") as f:
                mm = mmap.mmap(f.fileno(), 0)
                flag = mm.read_byte()
                step = struct.unpack("i", mm[4:8])[0]
                ratio = struct.unpack("d", mm[8:16])[0]
                entropy = struct.unpack("d", mm[32:40])[0]
                reward = struct.unpack("d", mm[40:48])[0]
                mm.close()
            data = {
                "status": int(flag),
                "step": step,
                "ratio": ratio,
                "entropy": entropy,
                "reward": reward,
                "mns": random.randint(80, 95) # Simulação MNS se não estiver no mmap
            }
        except:
            data = {"status": 0, "step": 0, "ratio": 1.0, "entropy": 0, "reward": 0, "mns": 0}

    # Atualizar histórico
    timestamp = time.strftime("%H:%M:%S")
    history["loss"].append({"time": timestamp, "value": data["entropy"] * 10}) # Exemplo
    history["latency"].append({"time": timestamp, "value": 12 + random.uniform(-1, 1)})
    history["mns"].append({"time": timestamp, "value": data["mns"]})
    
    # Manter apenas os últimos 20 pontos
    for k in history:
        if len(history[k]) > 20:
            history[k].pop(0)

    return jsonify(data)

@app.route('/api/history')
def get_history():
    return jsonify(history)

@app.route('/api/prompt', methods=['POST'])
def process_prompt():
    req = request.json
    prompt = req.get("prompt", "")
    
    if HAS_BRIDGE:
        # Lógica Real: Chamar bridge Julia e motor de 22.5M
        print(f"🧠 [Backend] Processando prompt: '{prompt}'...")
        response = bridge.generate_response(prompt)
    else:
        # Lógica de fallback para conversas em linguagem natural
        responses = [
            "Estou processando seu pedido de forma clara e objetiva.",
            "Compreendido. Como posso transformar essa sua ideia em realidade?",
            "Conexão estável. Estou pronto para te ajudar com o que precisar.",
            "Diga-me mais sobre isso. Minha base de conhecimentos está à sua disposição."
        ]
        response = random.choice(responses)
        time.sleep(1.0) # Simular latência humana natural

        
    return jsonify({"response": response})

@app.route('/api/reward', methods=['POST'])
def send_reward():
    req = request.json
    value = float(req.get("value", 0.5))
    
    # Escrever no Mmap para o Julia ler (Offset 40:48 para reward)
    if os.path.exists(MEM_FILE):
        try:
            with open(MEM_FILE, "r+b") as f:
                mm = mmap.mmap(f.fileno(), 0)
                mm[40:48] = struct.pack("d", value)
                mm.close()
            print(f"🎯 [Mentor] Sinal de Recompensa enviado: {value}")
            return jsonify({"status": "ok", "value": value})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    return jsonify({"status": "mmap_not_found"}), 404

if __name__ == "__main__":
    print("🚀 [CAFUNE Backend] Ativo em http://127.0.0.1:5000")
    app.run(port=5000, debug=False)
