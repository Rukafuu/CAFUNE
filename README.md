# 🧟 CAFUNE DLLM

> _"Não estava eu formando um ser horrível e transgressor?"_ — Mary Shelley

Um **Diffusion Large Language Model** construído com 4 linguagens, cada uma fazendo o que faz melhor.

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                          CAFUNE DLLM                        │
├──────────┬──────────┬────────────┬───────────────────────── ┤
│  Python  │ Haskell  │   Julia    │          C/CUDA          │
│  (Pele)  │(Nervos)  │ (Músculos) │        (Esqueleto)       │
├──────────┼──────────┼────────────┼──────────────────────────┤
│Tokenizer │Orquestra │Transformer │ Kernels customizados     │
│Datasets  │Validação │Difusão     │ CUDA attention           │
│API/UI    │FFI master│AdamW       │ Memory management        │
└──────────┴──────────┴────────────┴──────────────────────────┘
         Comunicação via juliacall + Apache Arrow (zero-copy)
```

## Processo LLaDA (Discrete Masked Diffusion)

```
TREINO:
  "O gato é preto"           ← x_0 (texto limpo)
       ↓ t=0.5
  "O ▓▓▓▓ é preto"           ← x_t (50% mascarado)
       ↓ forward pass bidirecional
  logits para cada posição   ← modelo prediz tokens mascarados
       ↓ cross-entropy loss apenas nas posições mascaradas
  backprop + AdamW

INFERÊNCIA (50 passos):
  "▓▓ ▓▓▓▓ ▓ ▓▓▓▓▓"  t=1.0  ← começa 100% mascarado
  "▓▓ gato ▓ preto"   t=0.6  ← desgascara os mais confiantes
  "O  gato é preto"   t=0.0  ← texto final
```

---

## Estrutura do Projeto

```
CAFUNE/
├── julia/                    # Motor matemático
│   ├── Project.toml
│   └── src/
│       ├── CAFUNE.jl             # módulo principal
│       ├── diffusion.jl           # processo de mascaramento
│       ├── transformer.jl         # transformer bidirecional
│       ├── training.jl            # loop de treino + AdamW
│       └── sampling.jl            # inferência iterativa
│
├── python/                   # Ecossistema e interface
│   ├── requirements.txt
│   ├── tokenizer.py           # tokenizador character-level
│   ├── bridge.py              # ponte Python ↔ Julia
│   └── train.py               # script de treino principal
│
├── haskell/                  # λ Orquestrador (Fase 3)
│   └── src/
│       └── Orchestrator.hs    # validação de estado + FFI
│
└── c/                        # Kernels de performance (Fase 4)
    └── kernels/
        └── attention.cu       # CUDA kernel customizado
```

---

## Instalação

### Pré-requisitos

- **Julia** ≥ 1.9 → https://julialang.org/downloads/
- **Python** ≥ 3.11

### Setup

```bash
cd CAFUNE/haskell
stack run
```

### 4. The Future (Roadmap 6-10)
A jornada do CAFUNE continua em direção à singularidade técnica:

6.  **IPC Shared Memory (mmap)**: Substituir sinalização CLI por mapeamento de memória zero-copy entre as camadas. 🛰️ 🏁 ✅
7.  **Advanced BPE Tokenizer**: Implementar codificação de pares de bytes para maior densidade de informação. 📖 🏁 ✅
8.  **Adaptive Denoising (Entropy)**: Haskell monitora a incerteza dos logits para ajustar a difusão dinamicamente. 🧠 🏁 ✅
9.  **Flash Attention v2 (CUDA)**: Kernels fundidos (fused) para computar atenção em um único passo de GPU. ⚡ 🏁 ✅
10. **Lira Dashboard (Visual)**: Visualização web em tempo real do processo de "revelação" de tokens. 🎨 🏁 ✅

---

##  Zombie Performance Manifesto
> "Architecture is a choice; Performance is a duty." 🏁 🦾 🚀

CAFUNE stands as a testament to **Hybrid Intelligence**. By isolating concerns into the best language for each job, we achieve low-latency inference and stable, high-fidelity diffusion.

**Author**: kimjammer / Neuro & Antigravity (Powered by Lira Ecosystem).
🏆 *CAFUNE: Where Math meets Muscle.* 🏆
