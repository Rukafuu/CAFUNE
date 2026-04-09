
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

MEM_FILE  = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "cafune_brain.mem"))
MEM_SIZE  = 1024
TRAIN_LOG = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "julia", "training_log.jsonl"))

# Histórico para os gráficos do Recharts
history = {
    "loss": [],
    "latency": [],
    "mns": []
}

@app.route('/')
def index():
    return """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>CAFUNE Monitor</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0d0d0d; color: #e0e0e0; font-family: 'Courier New', monospace; padding: 20px; }
    h1 { color: #9b59b6; margin-bottom: 4px; font-size: 1.4em; }
    .subtitle { color: #555; font-size: 0.8em; margin-bottom: 20px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 20px; }
    .card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; padding: 14px; }
    .card .label { font-size: 0.7em; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .card .value { font-size: 1.8em; font-weight: bold; margin-top: 4px; }
    .card .value.green  { color: #2ecc71; }
    .card .value.purple { color: #9b59b6; }
    .card .value.yellow { color: #f1c40f; }
    .card .value.red    { color: #e74c3c; }
    .card .value.cyan   { color: #1abc9c; }
    .card .value.blue   { color: #3498db; }
    .chart-box { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; padding: 14px; margin-bottom: 12px; }
    .chart-box h3 { font-size: 0.75em; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
    canvas { width: 100% !important; height: 100px !important; }
    .progress-bar-bg { background: #1a1a1a; border-radius: 4px; height: 8px; margin: 6px 0 10px; }
    .progress-bar-fill { background: #9b59b6; height: 8px; border-radius: 4px; transition: width 0.5s; }
    .epoch-table { width: 100%; border-collapse: collapse; font-size: 0.75em; }
    .epoch-table th { color: #555; text-align: left; padding: 4px 8px; border-bottom: 1px solid #222; }
    .epoch-table td { padding: 4px 8px; border-bottom: 1px solid #1a1a1a; }
    .epoch-table tr:first-child td { color: #2ecc71; }
    .log-box { background: #111; border: 1px solid #222; border-radius: 8px; padding: 12px; height: 160px; overflow-y: auto; font-size: 0.75em; }
    .log-box .entry { padding: 2px 0; border-bottom: 1px solid #1a1a1a; }
    .log-box .entry .ts { color: #555; margin-right: 8px; }
    .log-box .entry.rlaif { color: #9b59b6; }
    .log-box .entry.warn  { color: #f1c40f; }
    .log-box .entry.ok    { color: #2ecc71; }
    .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; background: #e74c3c; }
    .status-dot.alive { background: #2ecc71; animation: pulse 1.5s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
    .section-title { color: #555; font-size: 0.7em; text-transform: uppercase; letter-spacing: 2px; margin: 16px 0 8px; }
  </style>
</head>
<body>
  <h1>&#x1F9DF; CAFUNE Monitor</h1>
  <p class="subtitle">Live RLAIF Training Dashboard &mdash; Raegis + Gemini + Guardian</p>

  <p class="section-title">Estado do Motor</p>
  <div class="grid">
    <div class="card">
      <div class="label">Status</div>
      <div class="value" id="status-val" style="font-size:1em;margin-top:6px;">
        <span class="status-dot" id="dot"></span><span id="status-txt">--</span>
      </div>
    </div>
    <div class="card"><div class="label">Loss Atual</div><div class="value yellow" id="loss-val">--</div></div>
    <div class="card"><div class="label">Gemini MNS</div><div class="value purple" id="gemini-val">--</div></div>
    <div class="card"><div class="label">MNS Local</div><div class="value cyan" id="mns-val">--</div></div>
    <div class="card"><div class="label">Raegis Penalty</div><div class="value red" id="raegis-val">--</div></div>
    <div class="card"><div class="label">Guardian Penalty</div><div class="value red" id="guardian-val">--</div></div>
    <div class="card"><div class="label">Ethics Flag</div><div class="value" id="ethics-val">--</div></div>
    <div class="card"><div class="label">Reward Final</div><div class="value green" id="reward-val">--</div></div>
  </div>

  <p class="section-title">Histórico de Loss</p>
  <div class="chart-box">
    <h3>Loss (difusão mascarada)</h3>
    <canvas id="lossChart"></canvas>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
    <div class="chart-box">
      <h3>Gemini MNS Score</h3>
      <canvas id="geminiChart"></canvas>
    </div>
    <div class="chart-box">
      <h3>Penalidades (Raegis + Guardian)</h3>
      <canvas id="penaltyChart"></canvas>
    </div>
  </div>

  <p class="section-title">Progresso do Treino</p>
  <div class="chart-box">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
      <span id="epoch-label" style="font-size:0.85em;color:#9b59b6;">Epoch --/--</span>
      <span id="epoch-loss" style="font-size:0.85em;color:#f1c40f;">Loss: --</span>
      <span id="epoch-best" style="font-size:0.75em;color:#555;">Best: --</span>
      <span id="epoch-dataset" style="font-size:0.75em;color:#555;">Dataset: -- seqs</span>
    </div>
    <div class="progress-bar-bg"><div class="progress-bar-fill" id="epoch-bar" style="width:0%"></div></div>
    <div style="overflow-x:auto;">
      <table class="epoch-table">
        <thead><tr><th>Epoch</th><th>Loss</th><th>Best</th><th>Dataset</th><th>Hora</th></tr></thead>
        <tbody id="epoch-tbody"></tbody>
      </table>
    </div>
  </div>

  <p class="section-title">Resposta Atual do Modelo</p>
  <div class="log-box" id="response-box">
    <div class="entry"><span class="ts">--:--:--</span><span>Aguardando output do modelo...</span></div>
  </div>

<script>
let lastResponse = '';
// Mini chart renderer (canvas, sem dependência externa)
function makeChart(canvasId, color) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const data = [];
  return {
    push(v) {
      data.push(v);
      if (data.length > 40) data.shift();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (data.length < 2) return;
      const min = Math.min(...data), max = Math.max(...data);
      const range = max - min || 1;
      ctx.strokeStyle = color; ctx.lineWidth = 2;
      ctx.beginPath();
      data.forEach((d, i) => {
        const x = (i / (data.length - 1)) * canvas.width;
        const y = canvas.height - ((d - min) / range) * (canvas.height - 4) - 2;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
  };
}

const lossChart    = makeChart('lossChart',    '#f1c40f');
const geminiChart  = makeChart('geminiChart',  '#9b59b6');
const penaltyChart = makeChart('penaltyChart', '#e74c3c');

const STATUS = { 0: 'IDLE', 1: 'PROCESSANDO', 2: 'CONCLUIDO', 3: 'ERRO' };

function fmt(v, digits=3) { return isNaN(v) ? '--' : (+v).toFixed(digits); }

async function tick() {
  try {
    const r = await fetch('/api/mmap');
    const d = await r.json();

    const dot = document.getElementById('dot');
    const txt = document.getElementById('status-txt');
    dot.className = 'status-dot' + (d.cmd_id <= 2 ? ' alive' : '');
    txt.textContent = STATUS[d.cmd_id] || 'ERRO';

    document.getElementById('loss-val').textContent     = fmt(d.loss);
    document.getElementById('gemini-val').textContent   = fmt(d.gemini_score);
    document.getElementById('mns-val').textContent      = fmt(d.mns_local);
    document.getElementById('raegis-val').textContent   = fmt(d.raegis_penalty);
    document.getElementById('guardian-val').textContent = fmt(d.guardian_penalty);
    document.getElementById('ethics-val').textContent   = d.ethics_flag ? 'ALERTA' : 'OK';
    document.getElementById('ethics-val').className     = 'value ' + (d.ethics_flag ? 'red' : 'green');

    const alpha  = d.gemini_score > 0 ? 0.7 : 0.0;
    const combined = alpha * d.gemini_score + (1 - alpha) * d.mns_local;
    const pen = d.raegis_penalty * (d.ethics_flag ? 2 : 1) + d.guardian_penalty;
    const reward = Math.max(0, combined - pen);
    document.getElementById('reward-val').textContent = fmt(reward);

    lossChart.push(d.loss || 0);
    geminiChart.push(d.gemini_score || 0);
    penaltyChart.push((d.raegis_penalty || 0) + (d.guardian_penalty || 0));

    if (d.response && d.response.trim() && d.response !== lastResponse) {
      lastResponse = d.response;
      const box = document.getElementById('response-box');
      const ts = new Date().toTimeString().slice(0,8);
      const div = document.createElement('div');
      div.className = 'entry ok';
      div.innerHTML = '<span class="ts">' + ts + '</span>' + d.response.slice(0, 160);
      box.insertBefore(div, box.firstChild);
      if (box.children.length > 30) box.removeChild(box.lastChild);
    }
  } catch(e) { /* aguarda motor */ }
}

async function tickTraining() {
  try {
    const r = await fetch('/api/training_log');
    const rows = await r.json();
    if (!rows.length) return;
    const last = rows[rows.length - 1];
    const pct  = last.total > 0 ? Math.round((last.epoch / last.total) * 100) : 0;
    document.getElementById('epoch-label').textContent   = `Epoch ${last.epoch}/${last.total}`;
    document.getElementById('epoch-loss').textContent    = `Loss: ${last.loss}`;
    document.getElementById('epoch-best').textContent    = `Best: ${last.best_loss}`;
    document.getElementById('epoch-dataset').textContent = `Dataset: ${last.dataset_n} seqs`;
    document.getElementById('epoch-bar').style.width     = pct + '%';
    const tbody = document.getElementById('epoch-tbody');
    tbody.innerHTML = '';
    for (let i = rows.length - 1; i >= Math.max(0, rows.length - 10); i--) {
      const e = rows[i];
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${e.epoch}</td><td>${e.loss}</td><td>${e.best_loss}</td><td>${e.dataset_n}</td><td>${e.timestamp}</td>`;
      tbody.appendChild(tr);
    }
    lossChart.push(parseFloat(last.loss) || 0);
  } catch(e) { /* aguarda log */ }
}

setInterval(tick, 1500);
setInterval(tickTraining, 3000);
tick();
tickTraining();
</script>
</body>
</html>"""

@app.route('/api/mmap')
def get_mmap():
    mem_path = MEM_FILE
    if not os.path.exists(mem_path):
        return jsonify({"error": "mmap not found"}), 404
    try:
        with open(mem_path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 1024)
            cmd_id          = mm[0]
            loss            = struct.unpack('d', mm[32:40])[0]
            gemini_score    = struct.unpack('f', mm[40:44])[0]
            mns_local       = struct.unpack('f', mm[44:48])[0]
            raegis_penalty  = struct.unpack('f', mm[48:52])[0]
            guardian_penalty= struct.unpack('f', mm[52:56])[0]
            ethics_flag     = bool(mm[60])
            response        = mm[200:600].split(b'\x00')[0].decode('utf-8', errors='replace')
            mm.close()
        import math
        def safe(v): return 0.0 if (math.isnan(v) or math.isinf(v)) else round(v, 4)
        return jsonify({
            "cmd_id":           cmd_id,
            "loss":             safe(loss),
            "gemini_score":     safe(gemini_score),
            "mns_local":        safe(mns_local),
            "raegis_penalty":   safe(raegis_penalty),
            "guardian_penalty": safe(guardian_penalty),
            "ethics_flag":      ethics_flag,
            "response":         response,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

@app.route('/api/training_log')
def get_training_log():
    if not os.path.exists(TRAIN_LOG):
        return jsonify([])
    rows = []
    try:
        with open(TRAIN_LOG, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(rows[-50:])  # últimas 50 epochs

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
