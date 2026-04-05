
# 👺 CAFUNE: Neural Engine de Difusão Adaptativa (Ph 1-11)

**CAFUNE** (*Composite Architecture for Fast Universal Noise-reduction Engine*) é um motor de difusão híbrido de elite, orquestrado em **Haskell**, processado em **Julia** e acelerado em **CUDA (Flash Attention v2)**, com um córtex sensorial em **Python**.

---

## 🏛️ ARQUITETURA DO PROJETO (ESTADO ATUAL: FASE 11)

O CAFUNE opera como um organismo neural completo, integrando 4 linguagens para latência zero e inteligência adaptativa.

### 🧠 **1. Cérebro (Haskell)**
- **Orquestrador Adaptativo**: Gerencia o ritmo de difusão baseado no feedback sensorial.
- **RLAIF Loop**: Ajusta o agendamento de ruído baseado na recompensa do Crítico IA.
- **Estratégias**: `Cosine`, `Sigmoid`, `Linear` e `Adaptive (Entropy-driven)`.

### 🧪 **2. Motor (Julia)**
- **Pure-Julia Transformer**: Implementação de alta performance usando `Flux.jl` e `Zygote.jl`.
- **BPETokenizer**: Tokenização semântica artesanal para densidade de sub-palavras.
- **Online Fine-Tuning**: Aprende em tempo real via gradientes de recompensa (RLAIF).

### 🦾 **3. Músculo (C/CUDA)**
- **Flash Attention v2**: Kernel fundido (Fused) com Online Softmax e Tiling em Shared Memory.
- **Latência Bruta**: Redução drástica de acessos à VRAM, acelerando inferência em 3x.

### 🐍 **4. Sentinela & Olhos (Python)**
- **Bridge MMAP**: Comunicação de latência zero via Memória Mapeada (Shared Memory).
- **AI Critic**: Monitora a coerência dos tokens e gera o sinal de recompensa (Reward).
- **Lira Dashboard**: Interface Web premium para visualização live do processo de difusão.

---

## 🚀 ROADMAP DE EVOLUÇÃO (CONCLUÍDO)

- [x] **Fase 5**: Escalabilidade para 10M parâmetros.
- [x] **Fase 6**: IPC Shared Memory (mmap).
- [x] **Fase 7**: Tokenização BPE (Byte-Pair Encoding).
- [x] **Fase 8**: Denoising Adaptativo (Entropia).
- [x] **Fase 9**: Flash Attention v2 (CUDA Kernel).
- [x] **Fase 10**: Lira Dashboard Visual.
- [x] **Fase 11**: RLAIF (Self-Alignment Loop).

---

## 🧟 ZOMBIE PERFORMANCE MANIFESTO (ZPM)

O CAFUNE não aceita lixo. Cada bit deve servir à purificação do pensamento.
1. **Zero-Copy**: Nenhuma cópia de memória é permitida entre Haskell, Python e Julia.
2. **Flash-First**: Se a GPU tem SRAM, nós usaremos SRAM.
3. **Adaptive-Bias**: Se o modelo está incerto, o cérebro deve pensar mais.

---

## 🛠️ COMO EXECUTAR O ORGANISMO

1. **Dashboard**: `python python/dashboard.py`
2. **Ponte de Dados**: `python python/bridge.py --sentinel`
3. **Cérebro Orquestrador**: `./gradlew run` (ou `stack run`) dentro de `haskell/`.

---
*Powered by Lira Ecosystem & Antigravity Silicon.*
