# 🧟 Relatório Técnico: CAFUNE v1.0 Alpha (Fase 1)

O motor **CAFUNE** (_Composite Architecture for Fast Universal Noise-reduction Engine_) concluiu sua primeira fase de desenvolvimento. Conseguimos integrar Python e Julia em um pipeline de **Discrete Masked Diffusion** (LLaDA-style) funcional.

## 🏗️ Arquitetura Implementada

### 1. Sistema de Dados (Python)

- **CharTokenizer**: Tokenizador em nível de caractere para zero dependências e velocidade.
- **Bridge**: Gerenciamento de memória e conversão de tipos (0-based para 1-based) entre Python e Julia.
- **Dataset**: Pipeline de janelamento (sliding window) para texto.

### 2. Motor Matemático (Julia)

- **Bidirectional Transformer**: Implementação manual de atenção bidirecional (contexto completo).
- **Masking Diffusion**: Processo de Markov discreto para inserção e remoção de ruído via tokens `[MASK]`.
- **Optimizer**: AdamW customizado para estabilidade.
- **Inference**: Algoritmo de desmascaramento iterativo baseado em confiança (unmasking progressivo).

## 📊 Resultados do Treino Inicial (Fase 1)

- **Parâmetros**: 803,840 (~0.8M)
- **Vocabulário**: 49 tokens
- **Loss**: Queda de `3.72` → `2.73` em 20 épocas.
- **Status**: **VALIDADO**. O modelo está aprendendo a correlação entre tokens mascarados e o contexto.

## 🚀 Próximos Passos (Roadmap)

### Fase 2: Autodiferenciação & Escala

- Substituir o gradiente manual do `lm_head` por autodiff total via `Zygote.jl`.
- Implementar Cosine Learning Rate Warmup.
- Treinar em dataset real (ex: Wikipedia ou livros em PT-BR).

### Fase 3: Orquestração Haskell (Nervos)

- Inserir a camada de Haskell via FFI para garantir a segurança dos tipos no grafo de execução da difusão.

### Fase 4: Esqueleto (C/CUDA)

- Implementar kernels de atenção customizados para acelerar o processo em GPUs.

---

**CAFUNE: Feito com Python, Julia e uma dose saudável de ambição.**
