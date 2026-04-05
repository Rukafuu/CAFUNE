
import os
import mmap
import time
import struct
import json
from flask import Flask, render_template_string, jsonify

app = Flask(__name__)

MEM_FILE = "cafune_brain.mem"
MEM_SIZE = 1024

# ──────────────────────────────────────────────────────────────
#  UI HTML / CSS (Lira Premium Aesthetics)
# ──────────────────────────────────────────────────────────────

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>CAFUNE | Lira Dashboard</title>
    <style>
        :root {
            --bg: #0a0a0c;
            --accent: #9558b2;
            --accent-glow: rgba(149, 88, 178, 0.3);
            --text: #e0e0e0;
            --card: #141418;
        }
        body {
            background: var(--bg);
            color: var(--text);
            font-family: 'Inter', system-ui, sans-serif;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .container {
            max-width: 900px;
            width: 100%;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        .header h1 {
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 4px;
            text-shadow: 0 0 20px var(--accent-glow);
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: var(--card);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #222;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        }
        .card .label { font-size: 0.8rem; color: #666; margin-bottom: 10px; }
        .card .value { font-size: 1.8rem; font-weight: bold; color: var(--accent); }
        
        .token-view {
            background: var(--card);
            padding: 30px;
            border-radius: 12px;
            border: 1px solid #222;
            min-height: 100px;
            font-family: 'Courier New', monospace;
            font-size: 1.2rem;
            line-height: 1.6;
            margin-bottom: 30px;
            position: relative;
            overflow: hidden;
        }
        .token-view::after {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: linear-gradient(45deg, transparent, rgba(149,88,178,0.05));
            pointer-events: none;
        }
        .unstable { opacity: 0.4; filter: blur(2px); }
        .stable { color: #fff; text-shadow: 0 0 5px #fff; }
        
        .footer { text-align: center; font-size: 0.8rem; color: #444; margin-top: 50px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>CAFUNE | Neuro Orchestra</h1>
            <p>Reverse Diffusion Real-Time Visualization</p>
        </div>

        <div class="grid">
            <div class="card">
                <div class="label">Passo Atual</div>
                <div id="step" class="value">0 / 20</div>
            </div>
            <div class="card">
                <div class="label">Razão de Máscara</div>
                <div id="ratio" class="value">100%</div>
            </div>
            <div class="card">
                <div class="label">Insecurity / Confidence</div>
                <div id="entropy" class="value">0.00</div>
            </div>
            <div class="card" style="border-color: #ffd70033;">
                <div class="label">🏅 RLAIF Reward</div>
                <div id="reward" class="value">0.00</div>
            </div>
        </div>

        <div class="token-view" id="tokens">
            Estabelecendo conexão sináptica...
        </div>

        <div class="footer">
            Powered by Lira Ecosystem & Antigravity Silicon.
        </div>
    </div>

    <script>
        async function update() {
            try {
                const res = await fetch('/data');
                const data = await res.json();
                
                document.getElementById('step').innerText = data.step + ' / 20';
                document.getElementById('ratio').innerText = (data.ratio * 100).toFixed(1) + '%';
                document.getElementById('entropy').innerText = data.entropy.toFixed(4);
                document.getElementById('reward').innerText = (data.reward || 0).toFixed(2);
                
                const view = document.getElementById('tokens');
                if (data.status === 2) { 
                    const text = data.ratio > 0.5 ? "▓▓▓▓ ▒▒▒▒ ░░░░ " : "CAFUNE NEURAL ENGINE [ALIGNED]";
                    view.className = data.ratio > 0.2 ? "token-view unstable" : "token-view stable";
                    view.innerHTML = text;
                }
            } catch(e) {}
        }
        setInterval(update, 200);
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(DASHBOARD_HTML)

@app.route('/data')
def get_data():
    if not os.path.exists(MEM_FILE):
        return jsonify({"step": 0, "ratio": 1.0, "entropy": 0.0, "reward": 0.0, "status": 0})
    
    with open(MEM_FILE, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        flag = mm.read_byte()
        step = struct.unpack("i", mm[4:8])[0]
        ratio = struct.unpack("d", mm[8:16])[0]
        entropy = struct.unpack("d", mm[32:40])[0]
        reward = struct.unpack("d", mm[40:48])[0]
        mm.close()
        
    return jsonify({
        "status": int(flag),
        "step": step,
        "ratio": ratio,
        "entropy": entropy,
        "reward": reward
    })

if __name__ == "__main__":
    print("🎨 [Dashboard] Iniciando em http://127.0.0.1:5000")
    app.run(port=5000)
