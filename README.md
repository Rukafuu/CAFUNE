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
# 1. Instalar dependências Python
pip install -r python/requirements.txt

# 2. Instalar pacotes Julia
julia --project=julia -e "using Pkg; Pkg.instantiate()"

# 3. Rodar o treino
python python/train.py
```

---

## Roadmap

### Fase 1 — Python + Julia (atual)

- [x] Tokenizador character-level
- [x] Processo de difusão discreta (mascaramento)
- [x] Transformer bidirecional implementado do zero
- [x] Loop de treino com AdamW
- [x] Inferência iterativa (confidence-based unmasking)
- [x] Bridge Python ↔ Julia via juliacall

### Fase 2 — Autodiferenciação com Zygote.jl

- [ ] Substituir gradiente manual por Zygote.jl
- [ ] Treinar todos os parâmetros (atual treina só lm_head)
- [ ] Adicionar learning rate scheduling (cosine warmup)
- [ ] Checkpoint/save do modelo

### Fase 3 — Haskell como Orquestrador

- [ ] FFI Haskell → Julia (via C-ABI)
- [ ] Validação de invariantes com tipos algebraicos
- [ ] Controle do grafo de execução

### Fase 4 — C/CUDA Kernels

- [ ] Flash Attention kernel customizado
- [ ] Kernel de mascaramento em batch
- [ ] Integração com CUDA.jl

### Fase 5 — Escala

- [ ] Apache Arrow para zero-copy entre linguagens
- [ ] Dataset real (Wikipedia PT-BR, livros)
- [ ] Treinamento distribuído

---

## Conceitos Chave

| Conceito       | LLaDA                      | GPT                       |
| -------------- | -------------------------- | ------------------------- |
| **Atenção**    | Bidirecional (vê tudo)     | Causal (só passado)       |
| **Ruído**      | Mascaramento discreto      | N/A                       |
| **Geração**    | Paralela, iterativa        | Sequencial, token a token |
| **Loss**       | CE nas posições mascaradas | CE em todos os tokens     |
| **Inferência** | 50 passos de denoising     | 1 forward por token       |

---

## Referências

- **LLaDA** (2025): [Large Language Diffusion with mAsking](https://arxiv.org/abs/2502.09992)
- **MDLM** (2024): [Masked Diffusion Language Model](https://arxiv.org/abs/2406.07524)
- **D3PM** (2021): [Structured Denoising Diffusion for Discrete State Spaces](https://arxiv.org/abs/2107.03006)
- **BERT** (2018): [Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
