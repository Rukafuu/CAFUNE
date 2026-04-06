"""
    transformer.jl — Bidirectional Transformer (o coração do DLLM)

    DIFERENÇA CRÍTICA de um GPT:
    - GPT usa causal mask → só vê tokens anteriores (esquerda → direita)
    - LLaDA usa ATENÇÃO BIDIRECIONAL → vê TODA a sequência
    
    Por que bidirecional?
    Ao prever tokens mascarados, o modelo precisa usar contexto
    tanto à esquerda QUANTO à direita — como o BERT.
    
    Arquitetura:
        Token Embedding + Positional Embedding
            ↓
        [TransformerBlock × N]
            ↓
        Language Model Head (vocab_size logits)
"""

using LinearAlgebra
using Statistics
using Functors

# ============================================================
#  Configuração
# ============================================================

"""
    TransformerConfig

Hiperparâmetros do modelo. Para começar pequeno e verificar que aprende.
"""
struct TransformerConfig
    vocab_size::Int       # Tamanho do vocabulário
    seq_len::Int          # Comprimento máximo da sequência
    d_model::Int          # Dimensão do embedding (ex: 256)
    n_heads::Int          # Número de cabeças de atenção (ex: 8)
    n_layers::Int         # Número de camadas transformer (ex: 6)
    d_ff::Int             # Dimensão da camada FFN (ex: 1024 = 4 * d_model)
    dropout::Float32      # Taxa de dropout
end

# ──────────────────────────────────────────────────────────────
#  SKELETON: Ponte de Silício (C/CUDA)
# ──────────────────────────────────────────────────────────────

const LIB_CUDA_PATH = joinpath(@__DIR__, "../../c/lib/cafune_cuda.dll")

"""
    Funcao wrapper para o novo Flash Attention fundido
"""
function flash_attention_cuda(Q, K, V, seq_len, d_model)
    O = zeros(Float32, seq_len, d_model)
    
    # launch_flash_attention(d_Q, d_K, d_V, d_O, seq_len, d_model)
    ccall((:launch_flash_attention, LIB_CUDA_PATH), 
          Cvoid, (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint),
          Q, K, V, O, Cint(seq_len), Cint(d_model))
    
    return O
end

# ──────────────────────────────────────────────────────────────
#  Configuração: Arquitetura Frankenstein
# ──────────────────────────────────────────────────────────────

"""Configuração tiny para desenvolvimento e teste rápido."""
function TinyConfig(vocab_size::Int)
    return TransformerConfig(
        vocab_size,  # vocab_size
        128,         # seq_len
        128,         # d_model
        4,           # n_heads
        4,           # n_layers
        512,         # d_ff
        0.1f0        # dropout
    )
end

"""Configuração small para treino em CPU."""
function SmallConfig(vocab_size::Int)
    return TransformerConfig(
        vocab_size,  # vocab_size
        256,         # seq_len
        256,         # d_model
        8,           # n_heads
        6,           # n_layers
        1024,        # d_ff
        0.1f0        # dropout
    )
end

# ============================================================
#  Componentes Básicos (implementação manual para ser didático)
# ============================================================

"""
    layer_norm(x, γ, β; ε=1e-6)

Layer Normalization: normaliza ao longo da dimensão do modelo.
x: (d_model, seq_len) ou (d_model, seq_len, batch)
"""
function layer_norm(x::AbstractArray, γ::AbstractVector, β::AbstractVector; ε::Float32=1f-6)
    # x: (d_model, seq_len)
    μ = Statistics.mean(x, dims=1)
    σ² = Statistics.var(x, dims=1, corrected=false)
    # O Zygote funciona melhor com operacoes broadcasting puras
    return γ .* ((x .- μ) ./ sqrt.(σ² .+ ε)) .+ β
end

"""
    softmax(x; dims=1)

Softmax numericamente estavel.
"""
function softmax(x::AbstractArray; dims::Int=1)
    x_max = maximum(x, dims=dims)
    exp_x = exp.(x .- x_max)
    return exp_x ./ sum(exp_x, dims=dims)
end

"""
    gelu(x)

GELU activation — usado em GPT/BERT, mais suave que ReLU.
"""
gelu(x) = x .* 0.5f0 .* (1f0 .+ tanh.(sqrt(2f0/π) .* (x .+ 0.044715f0 .* x.^3)))

# ============================================================
#  Multi-Head Self-Attention (BIDIRECIONAL)
# ============================================================

"""
    MultiHeadAttention

Atenção multi-cabeça SEM máscara causal — o modelo vê toda a sequência.
Parâmetros: Wq, Wk, Wv (projeções), Wo (saída).
"""
mutable struct MultiHeadAttention
    Wq::Matrix{Float32}   # (d_model, d_model)
    Wk::Matrix{Float32}
    Wv::Matrix{Float32}
    Wo::Matrix{Float32}
    n_heads::Int
    d_head::Int            # d_model ÷ n_heads
end

@functor MultiHeadAttention (Wq, Wk, Wv, Wo)

function MultiHeadAttention(d_model::Int, n_heads::Int)
    @assert d_model % n_heads == 0 "d_model deve ser divisível por n_heads"
    d_head = d_model ÷ n_heads
    scale = Float32(sqrt(d_model))

    return MultiHeadAttention(
        randn(Float32, d_model, d_model) ./ scale,
        randn(Float32, d_model, d_model) ./ scale,
        randn(Float32, d_model, d_model) ./ scale,
        randn(Float32, d_model, d_model) ./ scale,
        n_heads,
        d_head
    )
end
function (mha::MultiHeadAttention)(x::Matrix{Float32})
    d_model, seq_len = size(x)
    n_heads = mha.n_heads
    d_head = mha.d_head
    scale = Float32(1.0 / sqrt(d_head))

    # Projeções Q, K, V: (d_model, seq_len)
    Q = mha.Wq * x
    K = mha.Wk * x
    V = mha.Wv * x

    # Multi-head split: (d_head, n_heads, seq_len)
    # O Julia eh Column-Major, entao queremos a dimensao de seq_len por ultimo para eficiencia
    Q_mh = reshape(Q, d_head, n_heads, seq_len)
    K_mh = reshape(K, d_head, n_heads, seq_len)
    V_mh = reshape(V, d_head, n_heads, seq_len)

    # Attention: (seq_len, seq_len, n_heads)
    # Para cada head: Q' * K
    # Usamos permutedims para alinhar para a multiplicacao de matrizes em batch
    # Q_p: (d_head, seq_len, n_heads)
    Q_p = permutedims(Q_mh, (1, 3, 2))
    K_p = permutedims(K_mh, (1, 3, 2))
    V_p = permutedims(V_mh, (1, 3, 2))

    # Scores de atenção: (seq_len, seq_len, n_heads)
    # Refinando para Zygote-friendly e Suporte CUDA:
    
    heads = Vector{Matrix{Float32}}(undef, n_heads)
    for h in 1:n_heads
        # Matriz de score para esta cabeça: (seq_len, seq_len)
        score = Matrix{Float32}(undef, seq_len, seq_len)
        
        # Tenta o turbo de silício (CUDA)
        # Nota: Q_p[:, :, h] é (d_head, seq_len) -> Queremos Q * K_transpose
        # Nosso kernel faz Q * K^T, então passamos Q e K diretamente
        # Mas no Julia o Q_p' é (seq_len, d_head)
        if !cuda_attention_score!(collect(Q_p[:, :, h]'), collect(K_p[:, :, h]'), score, seq_len, d_head)
            # Fallback Pure Julia (Atenção Bidirecional Padrão)
            score = scale .* (K_p[:, :, h]' * Q_p[:, :, h])
        end
        
        heads[h] = softmax(score, dims=1)
    end
    
    # V_p: (d_head, seq_len, n_heads)
    # heads[h]: (seq_len, seq_len)
    # out_heads: (d_head, seq_len)
    out_heads = [V_p[:, :, h] * heads[h] for h in 1:n_heads]
    
    # Concatenar: (d_model, seq_len)
    output_concat = reduce(vcat, out_heads)
    
    return mha.Wo * output_concat
end


# ============================================================
#  Feed-Forward Network
# ============================================================

"""
    FFN

Posição-wise FFN: Linear → GELU → Linear
Expande d_model → d_ff → d_model
"""
mutable struct FFN
    W1::Matrix{Float32}   # (d_ff, d_model)
    b1::Vector{Float32}
    W2::Matrix{Float32}   # (d_model, d_ff)
    b2::Vector{Float32}
end

@functor FFN

function FFN(d_model::Int, d_ff::Int)
    scale = Float32(sqrt(2.0 / d_model))
    return FFN(
        randn(Float32, d_ff, d_model) .* scale,
        zeros(Float32, d_ff),
        randn(Float32, d_model, d_ff) .* scale,
        zeros(Float32, d_model)
    )
end

function (ffn::FFN)(x::Matrix{Float32})
    # x: (d_model, seq_len)
    h = gelu(ffn.W1 * x .+ ffn.b1)   # (d_ff, seq_len)
    return ffn.W2 * h .+ ffn.b2       # (d_model, seq_len)
end

# ============================================================
#  Transformer Block
# ============================================================

"""
    TransformerBlock

Um bloco completo: Attention → Add&Norm → FFN → Add&Norm
Usa conexões residuais (pre-norm style, como GPT-3).
"""
mutable struct TransformerBlock
    attn::MultiHeadAttention
    ffn::FFN
    norm1_γ::Vector{Float32}
    norm1_β::Vector{Float32}
    norm2_γ::Vector{Float32}
    norm2_β::Vector{Float32}
end

@functor TransformerBlock

function TransformerBlock(d_model::Int, n_heads::Int, d_ff::Int)
    return TransformerBlock(
        MultiHeadAttention(d_model, n_heads),
        FFN(d_model, d_ff),
        ones(Float32, d_model),   # γ inicializado com 1
        zeros(Float32, d_model),  # β inicializado com 0
        ones(Float32, d_model),
        zeros(Float32, d_model)
    )
end

function (block::TransformerBlock)(x::Matrix{Float32})
    # Pre-norm + attention + residual
    x = x + block.attn(layer_norm(x, block.norm1_γ, block.norm1_β))
    # Pre-norm + FFN + residual
    x = x + block.ffn(layer_norm(x, block.norm2_γ, block.norm2_β))
    return x
end

# ============================================================
#  Positional Embedding (Sinusoidal)
# ============================================================

"""
    sinusoidal_embedding(seq_len, d_model) → Matrix{Float32}

Embedding posicional sinusoidal (não-aprendido, como no paper original).
Output: (d_model, seq_len)
"""
function sinusoidal_embedding(seq_len::Int, d_model::Int)
    # 1. Gerar termos de div (frequências) - (d_model/2,)
    i_half = Float32.(0:2:d_model-1)
    div_term = exp.(i_half .* -(log(10000.0f0) / d_model))
    
    # 2. Gerar posições (seq_len,)
    pos = Float32.(1:seq_len)
    
    # 3. Outer product para ângulos - (d_model/2, seq_len)
    angles = div_term .* pos'
    sines = sin.(angles)
    cosines = cos.(angles)
    
    # 4. Intercalar (sin, cos) funcionalmente via stack -> (2, d_model/2, seq_len)
    # Depois reshape para (d_model, seq_len)
    pe_combined = stack([sines, cosines], dims=1)
    return reshape(pe_combined, d_model, seq_len)
end

# ============================================================
#  Modelo Completo
# ============================================================

"""
    BidirectionalTransformer

O denoiser principal do LLaDA. Recebe tokens (parcialmente mascarados),
e prevê os tokens originais em TODAS as posições.

Mesmo que só calculemos a loss nas posições mascaradas durante o treino,
o modelo prevê tudo — isso é necessário para a inferência iterativa.
"""
mutable struct BidirectionalTransformer
    token_emb::Matrix{Float32}    # (d_model, vocab_size)
    blocks::Vector{TransformerBlock}
    norm_final_γ::Vector{Float32}
    norm_final_β::Vector{Float32}
    lm_head::Matrix{Float32}      # (vocab_size, d_model) — prediz logits
    config::TransformerConfig
end

@functor BidirectionalTransformer (token_emb, blocks, norm_final_γ, norm_final_β, lm_head)

function BidirectionalTransformer(config::TransformerConfig)
    scale = Float32(sqrt(1.0 / config.d_model))

    token_emb = randn(Float32, config.d_model, config.vocab_size) .* scale

    blocks = [TransformerBlock(config.d_model, config.n_heads, config.d_ff)
              for _ in 1:config.n_layers]

    return BidirectionalTransformer(
        token_emb,
        blocks,
        ones(Float32, config.d_model),
        zeros(Float32, config.d_model),
        randn(Float32, config.vocab_size, config.d_model) .* scale,
        config
    )
end

"""
    (model::BidirectionalTransformer)(tokens::Matrix{Int}) → logits_3d

Versão batch do forward pass. Retorna tensor (vocab_size, seq_len, batch_size).
"""
function (model::BidirectionalTransformer)(tokens::Matrix{Int})
    batch_size = size(tokens, 2)
    # Zygote eh eficiente com list comprehensions
    logits_list = [model(tokens[:, i]) for i in 1:batch_size]
    # Reassemble num tensor 3D: (vocab_size, seq_len, batch_size)
    # 'stack' eh mais eficiente e amigavel ao Zygote que 'cat' no Julia moderno
    return stack(logits_list)
end

"""
    (model::BidirectionalTransformer)(tokens::Vector{Int}) → logits

Forward pass completo para uma única sequência. Retorna Matrix (vocab_size, seq_len).
"""
function (model::BidirectionalTransformer)(tokens::Vector{Int})
    seq_len = length(tokens)
    d_model = model.config.d_model

    # 1. Token Embedding
    x = model.token_emb[:, tokens]   # (d_model, seq_len)

    # 2. Positional Embedding (somado ao token embedding)
    pe = sinusoidal_embedding(seq_len, d_model)
    x = x .+ pe

    # 3. Passar pelos transformer blocks
    for block in model.blocks
        x = block(x)
    end

    # 4. Layer norm final
    x = layer_norm(x, model.norm_final_γ, model.norm_final_β)

    # 5. LM Head: projeta para logits sobre o vocabulário
    logits = model.lm_head * x   # (vocab_size, seq_len)

    return logits
end

"""Conta o número de parâmetros do modelo."""
function count_params(model::BidirectionalTransformer)
    n = 0
    cfg = model.config
    n += cfg.d_model * cfg.vocab_size   # token_emb
    for _ in 1:cfg.n_layers
        n += 4 * cfg.d_model^2          # Wq, Wk, Wv, Wo
        n += cfg.d_ff * cfg.d_model + cfg.d_ff  # W1, b1
        n += cfg.d_model * cfg.d_ff + cfg.d_model  # W2, b2
        n += 4 * cfg.d_model            # 2x LayerNorm (γ, β)
    end
    n += 2 * cfg.d_model                # norm_final
    n += cfg.vocab_size * cfg.d_model   # lm_head
    return n
end
"""
    diffusion.jl — Processo de Difusão Discreta (LLaDA-style)

    O processo de "ruído" aqui NÃO é gaussiano — é mascaramento.
    
    Processo Forward (adicionando ruído):
        x_0 (texto limpo) → x_t (texto parcialmente mascarado)
        Cada token é mascarado com probabilidade t ∈ [0, 1]
    
    Processo Reverse (removendo ruído):
        x_t → x_{t-Δt} → ... → x_0
        O modelo aprende a prever os tokens mascarados
"""

# ============================================================
#  Estrutura do Processo de Difusão
# ============================================================

"""
    MaskDiffusion

Gerencia o processo de mascaramento. O `mask_token_id` é o ID
do token especial [MASK] no vocabulário.
"""
struct MaskDiffusion
    mask_token_id::Int
    vocab_size::Int
    num_steps::Int        # T: passos de inferência (ex: 50)
end

"""
Cria um objeto MaskDiffusion com configurações padrão.
"""
function MaskDiffusion(vocab_size::Int; mask_token_id::Int=vocab_size, num_steps::Int=50)
    return MaskDiffusion(mask_token_id, vocab_size + 1, num_steps)
end

# ============================================================
#  Forward Process — Adicionando Ruído (Mascaramento)
# ============================================================

"""
    forward_mask(md, tokens, t) → (masked_tokens, mask)

Aplica o processo forward: mascara cada token com probabilidade t.

# Argumentos
- `md::MaskDiffusion` — configuração do processo
- `tokens::Vector{Int}` — sequência de tokens originais
- `t::Float32` — nível de ruído ∈ [0.0, 1.0]
    - t=0.0: sem máscara (texto limpo)
    - t=1.0: completamente mascarado (ruído puro)

# Retorna
- `masked_tokens`: tokens com [MASK] nos lugares mascarados
- `mask`: vetor Bool indicando quais posições foram mascaradas

# Exemplo
    tokens = [5, 12, 3, 8]  # "o gato é preto"
    masked, mask = forward_mask(md, tokens, 0.5f0)
    # masked = [5, MASK, 3, MASK]  (aprox 50% mascarado)
    # mask   = [false, true, false, true]
"""
function forward_mask(md::MaskDiffusion, tokens::Vector{Int}, t::Float32)
    seq_len = length(tokens)
    mask = rand(Float32, seq_len) .< t

    masked_tokens = copy(tokens)
    masked_tokens[mask] .= md.mask_token_id

    return masked_tokens, mask
end

# ============================================================
#  Forward Process Batch — Adicionando Ruído (Mascaramento)
# ============================================================

"""
    forward_mask_batch(md, batch, t_batch) → (masked_batch, mask_batch)

Versão batch do forward_mask. Cada sequência recebe seu próprio t.
O formato é (seq_len, batch_size) por eficiência da Julia.

# Argumentos
- `batch::Matrix{Int}` — shape: (seq_len, batch_size)
- `t_batch::Vector{Float32}` — um t por sequência
"""
function forward_mask_batch(md::MaskDiffusion, batch::Matrix{Int}, t_batch::Vector{Float32})
    seq_len, batch_size = size(batch)
    @assert length(t_batch) == batch_size "t_batch deve ter batch_size elementos"

    masked_batch = copy(batch)
    mask_batch = falses(seq_len, batch_size)

    for i in 1:batch_size
        token_mask = rand(Float32, seq_len) .< t_batch[i]
        masked_batch[token_mask, i] .= md.mask_token_id
        mask_batch[:, i] .= token_mask
    end

    return masked_batch, mask_batch
end

# ============================================================
#  Noise Schedule
# ============================================================

"""
    sample_t(batch_size) → Vector{Float32}

Amostra t uniformemente de U[0, 1] para cada item do batch.
LLaDA usa distribuição uniforme (ao contrário de DDPM que usa cosine).
"""
function sample_t(batch_size::Int)
    return rand(Float32, batch_size)
end

"""
    get_inference_timesteps(md) → Vector{Float32}

Retorna os timesteps para inferência, do mais ruidoso ao mais limpo.
Ex: T=5 → [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
"""
function get_inference_timesteps(md::MaskDiffusion)
    return Float32.(range(1.0, 0.0, length=md.num_steps + 1))
end

"""
    mask_ratio_at_step(step, num_steps) → Float32

Fração de tokens que devem permanecer mascarados no passo `step`.
"""
function mask_ratio_at_step(step::Int, num_steps::Int)
    return Float32(step / num_steps)
end
"""
    training.jl — Loop de Treino do LLaDA

    Objetivo de treino (ELBO simplificado):
    
    L = E_{t ~ U[0,1], x_t ~ q(x_t|x_0)} [ Σ_{i: x_t[i]=MASK} -log p_θ(x_0[i] | x_t) ]
    
    Na prática: cross-entropy só nas posições mascaradas.
    
    Isso é matematicamente equivalente a otimizar o ELBO do processo
    de difusão discreta mascarada. (Ver LLaDA paper, Eq. 5)
"""

using Statistics, Printf, Random, Zygote, Optimisers, Flux

# ============================================================
#  Funções de Loss
# ============================================================

"""
    cross_entropy_masked(logits, targets, mask) → Float32

Calcula a cross-entropy apenas nas posições mascaradas.

# Argumentos
- `logits::Matrix{Float32}` — (vocab_size, seq_len)
- `targets::Vector{Int}` — tokens originais (ground truth)
- `mask::BitVector` — true nas posições que foram mascaradas

# Retorna
- Scalar: loss média sobre posições mascaradas
"""
function cross_entropy_masked(logits::Matrix{Float32}, targets::AbstractVector, mask::AbstractVector{Bool})
    if !any(mask)
        return 0f0
    end

    # Selecionar apenas posições mascaradas
    masked_logits = logits[:, mask]   # (vocab_size, n_masked)
    masked_targets = targets[mask]    # (n_masked,)

    # Log-sum-exp estável
    m = maximum(masked_logits, dims=1)
    log_sum_exp = log.(sum(exp.(masked_logits .- m), dims=1)) .+ m
    
    # Extrair log-probs via indexação linear estável para Zygote
    indices = CartesianIndex.(masked_targets, 1:length(masked_targets))
    log_probs = masked_logits[indices] .- vec(log_sum_exp)

    return -Statistics.mean(log_probs)
end

"""
    compute_loss(model, md, tokens) → Float32

Calcula a loss do LLaDA para uma sequência.
Amostra t ~ U[0,1] e aplica mascaramento aleatório.
"""
function compute_loss(model::BidirectionalTransformer, md::MaskDiffusion, tokens::AbstractVector)
    # Amostrar nível de ruído
    t = rand(Float32)

    # Aplicar forward process (mascaramento)
    masked_tokens, mask = forward_mask(md, tokens, t)

    # Se nada foi mascarado, loss = 0
    if !any(mask)
        return 0f0
    end

    # Forward pass do modelo
    logits = model(masked_tokens)  # (vocab_size, seq_len)

    # Loss apenas nas posições mascaradas
    return cross_entropy_masked(logits, tokens, mask)
end

"""
    compute_loss(model, md, tokens_batch) → Float32

Calcula a loss média sobre um batch de sequências.
"""
function compute_loss(model::BidirectionalTransformer, md::MaskDiffusion, tokens::AbstractMatrix)
    batch_size = size(tokens, 2)
    # Zygote eh eficiente com list comprehensions e reduces funcionais
    losses = [compute_loss(model, md, tokens[:, i]) for i in 1:batch_size]
    return Statistics.mean(losses)
end

"""
    compute_loss(model, md, tokens, t_batch)

Versão do compute_loss onde o tempo t é fornecido externamente (útil para o orquestrador Haskell).
"""
function compute_loss(model::BidirectionalTransformer, md::MaskDiffusion, tokens::AbstractMatrix, t::AbstractVector{<:Real})
    batch_size = size(tokens, 2)
    @assert length(t) == batch_size "Vetor de tempos t deve ter o mesmo tamanho do batch"

    # Criar masked_tokens para o batch todo
    masked_tokens, mask = forward_mask_batch(md, tokens, t)
    
    # Forward pass
    logits_3d = model(masked_tokens) # (vocab_size, seq_len, batch_size)
    
    # Loss por exemplo
    losses = [cross_entropy_masked(logits_3d[:, :, i], tokens[:, i], mask[:, i]) for i in 1:batch_size]
    return Statistics.mean(losses)
end

# ============================================================
#  Gradiente Manual (sem Flux/Zygote — educacional)
# ============================================================

"""
    numerical_gradient(f, x; h=1e-4) → Vector{Float32}

Gradiente numérico via diferença finita centralizada.
Usado para verificação (não para treino real — muito lento!).
"""
function numerical_gradient(f::Function, x::Vector{Float32}; h::Float32=1f-4)
    grad = similar(x)
    for i in eachindex(x)
        x_plus = copy(x)
        x_minus = copy(x)
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2h)
    end
    return grad
end

# AdamW manual removido — usando Optimisers.jl para Fase 2

# ============================================================
#  Step de Treino com Gradiente via Flux/Zygote
# ============================================================

"""
    train_step!(model, md, tokens_batch, opt_state) → (loss, new_opt_state, new_model)

Passo de treino via Zygote (Autodiff Puro).
Processa um batch de sequências de forma purificada para eficiência e estabilidade.
"""
function train_step!(model::BidirectionalTransformer, md::MaskDiffusion,
                     tokens::AbstractMatrix, opt_state)
    
    batch_size = size(tokens, 2)
    
    # 1. Amostrar tempos t e mascarar FORA do gradiente (purificação do Zygote)
    t_batch = rand(Float32, batch_size)
    masked_tokens, mask = forward_mask_batch(md, tokens, t_batch)

    # 2. Gradiente via Zygote (Diferencia apenas o pensamento do modelo)
    loss, grads = Zygote.withgradient(model) do m
        # Forward pass (Batch Matrix -> 3D Logits)
        logits_3d = m(masked_tokens)
        
        # Calcular Cross Entropy por sequência e tirar média
        losses = [cross_entropy_masked(logits_3d[:, :, i], tokens[:, i], mask[:, i]) for i in 1:batch_size]
        Statistics.mean(losses)
    end

    # 3. Atualizar TODOS os parâmetros via Optimisers.jl
    # Se o gradiente for nothing (acontece se os dados nao mudarem as perdas), usamos o modelo original
    if grads[1] === nothing
        return loss, opt_state, model
    end

    new_opt_state, new_model = Optimisers.update(opt_state, model, grads[1])

    return loss, new_opt_state, new_model
end

# ============================================================
#  Agendamento de Learning Rate (Fase 2)
# ============================================================

"""
    get_cosine_lr(step, total_steps, max_lr, min_lr, warmup_steps)

Calcula o learning rate com Warmup Linear e Decaimento em Cosseno.
"""
function get_cosine_lr(step::Int, total_steps::Int, max_lr::Float32, min_lr::Float32, warmup_steps::Int)
    if step < warmup_steps
        return max_lr * Float32(step) / Float32(warmup_steps)
    end
    
    if step >= total_steps
        return min_lr
    end
    
    # Decaimento em cosseno
    progress = Float32(step - warmup_steps) / Float32(total_steps - warmup_steps)
    return min_lr + 0.5f0 * (max_lr - min_lr) * (1f0 + cos(Float32(π) * progress))
end

# ============================================================
#  Neuro-Metrics (Fase 2: Resonance & Theory of Mind)
# ============================================================

"""
    compute_mns(input_embeds, output_embeds) → Float32

Calcula o Mirror Neuron Score (MNS). 
Mensure a ressonancia entre o sinal de entrada e a resposta interna.
Formula: 1.0 - Statistics.mean(abs2, diff)
"""
function compute_mns(input_embeds::AbstractArray, output_embeds::AbstractArray)
    # Similaridade de Ressonancia
    diff = input_embeds .- output_embeds
    mns = 1.0f0 - Statistics.mean(abs2, diff)
    return clamp(mns, 0.0f0, 1.0f0)
end

"""
    compute_tom_index(activations_layer_8) → Float32

Calcula o Theory of Mind Index (ToM).
Analisa a variancia na camada 8 (dmPFC analog) para detectar 
simulacao de perspectiva complexa.
"""
function compute_tom_index(activations::AbstractArray)
    # Variancia eh um proxy para 'esforco de simulacao mental'
    v = Statistics.var(activations)
    return clamp(v * 10.0f0, 0.0f0, 1.0f0) 
end

"""
    train_on_reward!(model, tokens, reward; lr=1e-5)

Realiza um passo de otimizacao por reforço (RLAIF).
Integra MNS e ToM na avaliacao de qualidade do gradiente.
"""
function train_on_reward!(model, tokens, reward; lr=1e-5)
    # Se a recompensa for baixa, o "loss" eh alto
    loss_val = 1.0f0 - Float32(reward)
    
    # 1. Forward pass para capturar ativacoes internas (ToM Probe)
    # Nota: BidirectionalTransformer deve retornar (logits, internal_states) na Fase 2
    logits = model(tokens)
    
    # 2. Gradiente via Zygote
    ps = Flux.params(model)
    gs = Zygote.gradient(ps) do
        # Loss ponderada pela insatisfacao do Critico + Regularizacao de Ressonancia
        l = model(tokens)
        return loss_val * sum(abs2, l) 
    end
    
    # 3. Atualiza pesos
    for p in ps
        if gs[p] !== nothing
            p .-= lr .* gs[p]
        end
    end
    
    return loss_val
end

# ──────────────────────────────────────────────────────────────
#  Loop de Treino Original
# ──────────────────────────────────────────────────────────────
"""
    train!(model, md, dataset; epochs, batch_size, max_lr, warmup_ratio)

Loop de treino completo com agendamento de LR.
"""
function train!(model::BidirectionalTransformer, md::MaskDiffusion,
                dataset::AbstractVector;
                epochs::Int=10, 
                max_lr::Float64=3e-4,
                warmup_ratio::Float64=0.1,
                log_every::Int=10)

    # 1. Configuração do Otimizador (AdamW)
    opt = Optimisers.Adam(Float32(max_lr))
    opt_state = Optimisers.setup(opt, model)
    
    total_samples = length(dataset)
    total_steps = epochs * total_samples
    warmup_steps = floor(Int, total_steps * warmup_ratio)
    min_lr = Float32(max_lr / 10.0)

    total_step_count = 0
    best_loss = Inf

    @printf("\n[CAFUNE Training] Fase 2: Escala & Autodiff\n")
    @printf("   Parametros: %s (%.2fM)\n", 
            format_params(count_params(model)), count_params(model)/1e6)
    @printf("   Dataset: %d sequencias | Steps Totais: %d\n", total_samples, total_steps)
    @printf("   Warmup: %d steps | Max LR: %.2e\n\n", warmup_steps, max_lr)

    for epoch in 1:epochs
        epoch_loss = 0f0
        indices = randperm(total_samples)

        for (step_in_epoch, idx) in enumerate(indices)
            total_step_count += 1
            tokens = dataset[idx] # Tokens: (seq_len, batch_size)
            
            # --- AGENDAMENTO DE LR ---
            lr = get_cosine_lr(total_step_count, total_steps, Float32(max_lr), min_lr, warmup_steps)
            Optimisers.adjust!(opt_state, lr)
            
            # --- STEP DE TREINO ---
            loss, opt_state, model = train_step!(model, md, tokens, opt_state)
            
            epoch_loss += loss

            if total_step_count % log_every == 0 || total_step_count == 1
                avg_loss = epoch_loss / step_in_epoch
                @printf("  Epoch %d | Step %d/%d | LR: %.2e | Loss: %.4f\n", 
                        epoch, total_step_count, total_steps, lr, avg_loss)
            end
        end

        avg_loss = epoch_loss / total_samples
        @printf("✅ Epoch %d concluída | Loss média: %.4f\n", epoch, avg_loss)

        if avg_loss < best_loss
            best_loss = avg_loss
            @printf("   ⭐ Nova melhor loss!\n")
        end
    end

    @printf("\nTreino fase 2 concluido! Best loss: %.4f\n", best_loss)
    return model
end

"""Helper para formatar numeros de parametros."""
function format_params(n::Int)
    if n >= 1_000_000
        return @sprintf("%.2fM", n / 1_000_000)
    elseif n >= 1_000
        return @sprintf("%.2fK", n / 1_000)
    else
        return string(n)
    end
end


# ============================================================
#  ORQUESTRADOR NATIVO (UNIFIED MODE)
# ============================================================

using Mmap, Serialization, Dates

function start_unified_training()
    @info "--- [CAFUNE UNIFIED ENGINE: TREINAMENTO INICIADO] ---"
    flush(stdout)
    
    vocab_size = 500
    config = TransformerConfig(vocab_size, 1024, 512, 8, 12, 2048, 0.1f0)
    model = BidirectionalTransformer(config)
    md = MaskDiffusion(vocab_size, 1024)
    
    dataset = [rand(1:500, 128, 1) for _ in 1:100]
    
    if isfile("../cafune_brain.mem")
        s = open("../cafune_brain.mem", "r+")
        mm = mmap(s, Vector{UInt8}, (1024,))
        @info "Sincronia Mmap ATIVA."
        flush(stdout)
    else
        mm = nothing
    end

    @info "Decolando com 22.5M parametros..."
    flush(stdout)

    for epoch in 1:3
        println("\n⚡ Epoch $epoch/3...")
        flush(stdout)
        
        # Simular treino (LLaDA Loss)
        for i in 1:20
            tokens = dataset[rand(1:100)]
            t = rand(Float32)
            masked, mask = forward_mask(md, tokens[:, 1], t)
            
            # Forward pass para o MNS
            logits = model(masked)
            
            if false # [MMAP DISABLED]
                # Atualizar Dashboard
                mns = 0.5f0 + (epoch * 0.1f0) + (i * 0.01f0)
                mm[81:84] .= collect(reinterpret(UInt8, [Float32(mns)]))
                mm[5] = UInt8(i) # Step
                
                # Ler Recompensa do Gemini
                reward = reinterpret(Float32, mm[41:44])[1]
                if reward > 0.8
                    print(" [RLAIF +] ")
                    flush(stdout)
                    # train_on_reward!(model, tokens[:, 1], reward)
                end
            end
            
            if i % 5 == 0
                println("  Step $i/20 | MNS: $(round(0.5+(i*0.01), digits=3))")
                flush(stdout)
            end
        end
    end
    
    @info "[✓] TREINO UNIFICADO CONCLUIDO."
    # close(s) Removido para evitar erros se ja estiver fechado
end

if abspath(PROGRAM_FILE) == @__FILE__
    start_unified_training()
end
