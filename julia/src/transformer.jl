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
    
    if isfile(LIB_CUDA_PATH)
        # launch_flash_attention(d_Q, d_K, d_V, d_O, seq_len, d_model)
        ccall((:launch_flash_attention, LIB_CUDA_PATH), 
              Cvoid, (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint),
              Q, K, V, O, Cint(seq_len), Cint(d_model))
    end
    
    return O
end

"""
    cuda_attention_score!(Q, K, score, seq_len, d_head)
    
Tenta usar GPU para calcular o produto escalar da atenção.
Retorna true se disparado com sucesso, false caso contrário.
"""
function cuda_attention_score!(Q, K, score, seq_len, d_head)
    if isfile(LIB_CUDA_PATH)
        try
            # launch_attention_score(d_Q, d_K, d_S, seq_len, d_head)
            ccall((:launch_attention_score, LIB_CUDA_PATH), 
                  Cint, (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint),
                  Q, K, score, Cint(seq_len), Cint(d_head))
            return true
        catch
            return false
        end
    end
    return false
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

# ──────────────────────────────────────────────────────────────
#  RoPE: Rotary Positional Embeddings
# ──────────────────────────────────────────────────────────────

"""
    apply_rope(x, cos_pos, sin_pos)

Aplica rotação rotacional (RoPE) em x. 
x: (d_head, seq_len) ou (d_head, seq_len, batch)
"""
function apply_rope(x::AbstractMatrix{Float32}, cos_pos::Matrix{Float32}, sin_pos::Matrix{Float32})
    d_head, seq_len = size(x)
    half = d_head ÷ 2
    
    # x_left: (half, seq_len), x_right: (half, seq_len)
    x_left = x[1:half, :]
    x_right = x[half+1:end, :]
    
    # Rotação: [x1*cos - x2*sin, x1*sin + x2*cos]
    out_left = x_left .* cos_pos .- x_right .* sin_pos
    out_right = x_left .* sin_pos .+ x_right .* cos_pos
    
    return vcat(out_left, out_right)
end

"""
    precompute_rope(seq_len, d_head) → (cos, sin)

Pre-calcula as tabelas de cosseno e seno para o RoPE.
"""
function precompute_rope(seq_len::Int, d_head::Int)
    half = d_head ÷ 2
    # Frequências: base 10000 como no Llama
    inv_freq = 1.0f0 ./ (10000.0f0 .^ (Float32.(0:2:half-1) ./ d_head))
    
    t = Float32.(0:seq_len-1)
    # outer product: (half, seq_len)
    freqs = inv_freq .* t'
    
    return cos.(freqs), sin.(freqs)
end

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

    # Pre-computar RoPE para o comprimento atual
    cos_pos, sin_pos = precompute_rope(seq_len, d_head)

    # Projeções Q, K, V: (d_model, seq_len)
    Q = mha.Wq * x
    K = mha.Wk * x
    V = mha.Wv * x

    # Multi-head split: (d_head, n_heads, seq_len)
    Q_mh = reshape(Q, d_head, n_heads, seq_len)
    K_mh = reshape(K, d_head, n_heads, seq_len)
    V_mh = reshape(V, d_head, n_heads, seq_len)

    # Aplicar RoPE em cada cabeça
    # Zygote-friendly: Usando stack/comprehension
    Q_rope = stack([apply_rope(Q_mh[:, h, :], cos_pos, sin_pos) for h in 1:n_heads]) # (d_head, seq_len, n_heads)
    K_rope = stack([apply_rope(K_mh[:, h, :], cos_pos, sin_pos) for h in 1:n_heads]) # (d_head, seq_len, n_heads)
    V_p = permutedims(V_mh, (1, 3, 2)) # (d_head, seq_len, n_heads)

    # Attention scores
    heads = [softmax(scale .* (K_rope[:, :, h]' * Q_rope[:, :, h]), dims=1) for h in 1:n_heads]
    
    out_heads = [V_p[:, :, h] * heads[h] for h in 1:n_heads]
    
    output_concat = reduce(vcat, out_heads)
    return mha.Wo * output_concat
end

# ============================================================
#  Spiking Synchrony Attention (SSA) - Estilo Isla-SNN
# ============================================================

"""
    SpikingSynchronyAttention

Atenção por Sincronia de Disparo. Em vez de Dot-Product, usa a proximidade 
temporal/espacial em um hipercubo unitário via Kernel RBF.
"""
mutable struct SpikingSynchronyAttention
    W_proj::Matrix{Float32} # Projeta para o Hipercubo [0, 1]^d
    Wv::Matrix{Float32}
    Wo::Matrix{Float32}
    tau::Float32           # Temperatura do Kernel (RBF precision)
    n_heads::Int
    d_head::Int
end

@functor SpikingSynchronyAttention (W_proj, Wv, Wo)

function SpikingSynchronyAttention(d_model::Int, n_heads::Int; tau=0.5f0)
    d_head = d_model ÷ n_heads
    scale = Float32(sqrt(d_model))
    return SpikingSynchronyAttention(
        randn(Float32, d_model, d_model) ./ scale,
        randn(Float32, d_model, d_model) ./ scale,
        randn(Float32, d_model, d_model) ./ scale,
        Float32(tau),
        n_heads,
        d_head
    )
end

function (ssa::SpikingSynchronyAttention)(x::Matrix{Float32})
    d_model, seq_len = size(x)
    n_heads = ssa.n_heads
    d_head = ssa.d_head

    # Pre-computar RoPE para sincronia
    cos_pos, sin_pos = precompute_rope(seq_len, d_head)

    # 1. Projeção para o Hipercubo Unitário (Baseado no Isla-SNN)
    P = Flux.sigmoid.(ssa.W_proj * x) # (d_model, seq_len)
    V = ssa.Wv * x

    # Multi-head split
    P_mh = reshape(P, d_head, n_heads, seq_len)
    V_mh = reshape(V, d_head, n_heads, seq_len)

    # Aplicar RoPE na projeção hipercúbica (faz a fase rotacionar no tempo)
    P_rope = stack([apply_rope(P_mh[:, h, :], cos_pos, sin_pos) for h in 1:n_heads]) # (d_head, seq_len, n_heads)
    V_p = permutedims(V_mh, (1, 3, 2))

    # 2. Kernel de Sincronia (RBF)
    out_heads = []
    for h in 1:n_heads
        Ph = P_rope[:, :, h] 
        Vh = V_p[:, :, h] 
        
        # Distância no espaço rotacionado (Atenção por Sincronia de Fase)
        dist_sq = sum(Ph.^2, dims=1)' .+ sum(Ph.^2, dims=1) .- 2 .* (Ph' * Ph)
        attn = exp.(-dist_sq ./ (2 * ssa.tau^2))
        attn_norm = attn ./ (sum(attn, dims=1) .+ 1f-6)
        push!(out_heads, Vh * attn_norm)
    end
    
    output_concat = reduce(vcat, out_heads)
    return ssa.Wo * output_concat
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

# ──────────────────────────────────────────────────────────────
#  Spiking Transformer Block
# ──────────────────────────────────────────────────────────────

"""
    SpikingTransformerBlock

Bloco que usa Spiking Synchrony Attention em vez de Dot-Product.
"""
mutable struct SpikingTransformerBlock
    attn::SpikingSynchronyAttention
    ffn::FFN
    norm1_γ::Vector{Float32}
    norm1_β::Vector{Float32}
    norm2_γ::Vector{Float32}
    norm2_β::Vector{Float32}
end

@functor SpikingTransformerBlock

function SpikingTransformerBlock(d_model::Int, n_heads::Int, d_ff::Int)
    return SpikingTransformerBlock(
        SpikingSynchronyAttention(d_model, n_heads),
        FFN(d_model, d_ff),
        ones(Float32, d_model),
        zeros(Float32, d_model),
        ones(Float32, d_model),
        zeros(Float32, d_model)
    )
end

function (block::SpikingTransformerBlock)(x::Matrix{Float32})
    x = x + block.attn(layer_norm(x, block.norm1_γ, block.norm1_β))
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
    blocks::Vector{Any}           # Mix de TransformerBlock e SpikingTransformerBlock
    norm_final_γ::Vector{Float32}
    norm_final_β::Vector{Float32}
    lm_head::Matrix{Float32}      # (vocab_size, d_model) — prediz logits
    config::TransformerConfig
end

@functor BidirectionalTransformer (token_emb, blocks, norm_final_γ, norm_final_β, lm_head)

function BidirectionalTransformer(config::TransformerConfig)
    scale = Float32(sqrt(1.0 / config.d_model))

    token_emb = randn(Float32, config.d_model, config.vocab_size) .* scale

    # Criamos uma arquitetura híbrida: 
    # Primeiras camadas: Atenção Padrão (Silício) para extração de features
    # Últimas camadas: Spiking Synchrony Attention (Neuromórfico) para síntese
    blocks = []
    for l in 1:config.n_layers
        if l <= config.n_layers ÷ 2
            push!(blocks, TransformerBlock(config.d_model, config.n_heads, config.d_ff))
        else
            push!(blocks, SpikingTransformerBlock(config.d_model, config.n_heads, config.d_ff))
        end
    end

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

    # 2. Positional Embedding (RoPE é aplicado dentro dos blocos de atenção agora)
    # Removemos o sinusoidal aditivo para usar a rotação dinâmica.

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
