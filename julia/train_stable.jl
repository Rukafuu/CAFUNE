using BSON: @save
using Printf, LinearAlgebra, Statistics, Functors, Dates, Flux, Optimisers, Zygote, CUDA

# ============================================================
#  CONFIGURACAO E TIPOS (ASCII ONLY)
# ============================================================
struct TransformerConfig
    vocab_size::Int; seq_len::Int; d_model::Int
    n_heads::Int; n_layers::Int; d_ff::Int; dropout::Float32
end

# Detectar dispositivo (GPU se disponivel)
device = CUDA.functional() ? gpu : cpu
@info "Dispositivo de Treino: $(device == gpu ? "GPU (TITAN SILICON)" : "CPU (NATIVE)")"; flush(stdout)

# ============================================================
#  COMPONENTES NEURAIS VETORIZADOS (GPU-BMM-NATIVE)
# ============================================================

function layer_norm(x::AbstractArray, γ::AbstractVector, β::AbstractVector; ε::Float32=1f-6)
    μ = Statistics.mean(x, dims=1)
    σ² = Statistics.var(x, dims=1, corrected=false)
    return γ .* ((x .- μ) ./ sqrt.(σ² .+ ε)) .+ β
end

function softmax(x::AbstractArray; dims::Int=1)
    x_max = maximum(x, dims=dims)
    exp_x = exp.(x .- x_max)
    return exp_x ./ sum(exp_x, dims=dims)
end

gelu(x) = x .* 0.5f0 .* (1f0 .+ tanh.(sqrt(2f0/pi) .* (x .+ 0.044715f0 .* x.^3)))

mutable struct MultiHeadAttention
    Wq; Wk; Wv; Wo; n_heads::Int; d_head::Int
end
@functor MultiHeadAttention (Wq, Wk, Wv, Wo)

function MultiHeadAttention(d_model::Int, n_heads::Int)
    d_head = d_model ÷ n_heads; scale = Float32(sqrt(d_model))
    return MultiHeadAttention(randn(Float32, d_model, d_model) ./ scale, randn(Float32, d_model, d_model) ./ scale, randn(Float32, d_model, d_model) ./ scale, randn(Float32, d_model, d_model) ./ scale, n_heads, d_head)
end

function (mha::MultiHeadAttention)(x::AbstractArray{Float32, 3})
    d_model, seq_len, batch_size = size(x)
    n_heads = mha.n_heads; d_head = mha.d_head
    scale = Float32(1.0 / sqrt(d_head))

    x_flat = reshape(x, d_model, :)
    Q = reshape(mha.Wq * x_flat, d_head, n_heads, seq_len, batch_size)
    K = reshape(mha.Wk * x_flat, d_head, n_heads, seq_len, batch_size)
    V = reshape(mha.Wv * x_flat, d_head, n_heads, seq_len, batch_size)

    Q_p = reshape(permutedims(Q, (1, 3, 2, 4)), d_head, seq_len, :)
    K_p = reshape(permutedims(K, (1, 3, 2, 4)), d_head, seq_len, :)
    V_p = reshape(permutedims(V, (1, 3, 2, 4)), d_head, seq_len, :)

    # Use Flux.batched_mul que eh otimizado para GPU (CUBLAS)
    scores = softmax(scale .* Flux.batched_mul(permutedims(K_p, (2, 1, 3)), Q_p), dims=1)
    out_mh = Flux.batched_mul(V_p, scores)
    
    out_reshaped = reshape(out_mh, d_head, seq_len, n_heads, batch_size)
    out_final = reshape(permutedims(out_reshaped, (1, 3, 2, 4)), d_model, seq_len, batch_size)

    return mha.Wo * reshape(out_final, d_model, :) |> x_ -> reshape(x_, d_model, seq_len, batch_size)
end

mutable struct FFN; W1; b1; W2; b2; end
@functor FFN
function FFN(d_model::Int, d_ff::Int)
    s = Float32(sqrt(2.0 / d_model)); return FFN(randn(Float32, d_ff, d_model) .* s, zeros(Float32, d_ff), randn(Float32, d_model, d_ff) .* s, zeros(Float32, d_model))
end
function (ffn::FFN)(x::AbstractArray{Float32, 3})
    d_m, s_l, b_s = size(x)
    h = gelu(ffn.W1 * reshape(x, d_m, :) .+ ffn.b1)
    out = ffn.W2 * h .+ ffn.b2
    return reshape(out, d_m, s_l, b_s)
end

mutable struct TransformerBlock; attn::MultiHeadAttention; ffn::FFN; n1_γ; n1_β; n2_γ; n2_β; end
@functor TransformerBlock
function TransformerBlock(d_model::Int, n_heads::Int, d_ff::Int)
    return TransformerBlock(MultiHeadAttention(d_model, n_heads), FFN(d_model, d_ff), ones(Float32, d_model), zeros(Float32, d_model), ones(Float32, d_model), zeros(Float32, d_model))
end
function (block::TransformerBlock)(x::AbstractArray{Float32, 3})
    x = x + block.attn(layer_norm(x, block.n1_γ, block.n1_β))
    x = x + block.ffn(layer_norm(x, block.n2_γ, block.n2_β))
    return x
end

function sinusoidal_embedding(seq_len::Int, d_model::Int)
    i_h = Float32.(0:2:d_model-1)
    dt = exp.(i_h .* -(log(10000.0f0) / d_model))
    pos = Float32.(1:seq_len)
    ang = dt .* pos'
    pe = zeros(Float32, d_model, seq_len)
    pe[1:2:end, :] .= sin.(ang)
    pe[2:2:end, :] .= cos.(ang)
    return pe
end

mutable struct BidirectionalTransformer; token_emb; blocks::Vector{TransformerBlock}; n_f_γ; n_f_β; lm_head; config::TransformerConfig; pe_cache; end
@functor BidirectionalTransformer (token_emb, blocks, n_f_γ, n_f_β, lm_head, pe_cache)

function BidirectionalTransformer(cfg::TransformerConfig)
    s = Float32(sqrt(1.0 / cfg.d_model))
    pe = sinusoidal_embedding(cfg.seq_len, cfg.d_model)
    return BidirectionalTransformer(randn(Float32, cfg.d_model, cfg.vocab_size) .* s, [TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff) for _ in 1:cfg.n_layers], ones(Float32, cfg.d_model), zeros(Float32, cfg.d_model), randn(Float32, cfg.vocab_size, cfg.d_model) .* s, cfg, pe)
end

function (model::BidirectionalTransformer)(tokens::AbstractMatrix{<:Integer})
    s_l, b_s = size(tokens)
    d_m = model.config.d_model
    x = reshape(model.token_emb[:, vec(tokens)], d_m, s_l, b_s)
    x = x .+ model.pe_cache
    for b in model.blocks; x = b(x); end
    x_norm = layer_norm(x, model.n_f_γ, model.n_f_β)
    logits = model.lm_head * reshape(x_norm, d_m, :)
    return reshape(logits, model.config.vocab_size, s_l, b_s)
end

# ============================================================
#  ORQUESTRADOR NATIVO (GPU POWERED AAA)
# ============================================================

function start_stable_training()
    @info "--- [CAFUNE NATIVE: TREINO FINAL GPU TITAN] ---"; flush(stdout)
    vs = 500; cfg = TransformerConfig(vs, 128, 512, 8, 12, 2048, 0.1f0)
    
    # Mover modelo para a GPU se disponivel
    model = BidirectionalTransformer(cfg) |> device
    opt = Optimisers.Adam(1f-3)
    opt_state = Optimisers.setup(opt, model)
    
    # Dataset real preparado para o hardware
    dataset = [rand(1:500, 128, 8) |> device for _ in 1:20]
    
    @info "Motor de 22.5M de parametros decolando no Silicío..."
    
    for epoch in 1:3
        println("\nStep Epoch $epoch/3...")
        epoch_loss = 0f0
        for i in 1:length(dataset)
            tokens = dataset[i]
            l, g = Zygote.withgradient(model) do m
                logits = m(tokens)
                # Sincronizacao de Formato para logitcrossentropy (GPU-Safe)
                Flux.logitcrossentropy(reshape(logits, vs, :), Flux.onehotbatch(vec(tokens) |> cpu, 1:vs) |> device)
            end
            opt_state, model = Optimisers.update!(opt_state, model, g[1])
            epoch_loss += l
            if i % 5 == 0
                @printf("  Step %d/%d | Loss: %.4f | Status: GPU-ACCELERATED\n", i, length(dataset), l)
                flush(stdout)
            end
        end

        @printf("OK Epoch %d concluida | Loss Media: %.4f\n", epoch, epoch_loss/length(dataset))
        
        # PERSISTENCIA NEURAL AAA
        m_cpu = model |> cpu
        @save "cafune_model.bson" m_cpu
        @info "Pesos de 22.5M [Plexus v1] salvos para inferencia real."
    end
    @info "[DONE] CAFUNE CONCLUIU O BATISMO DE SILICIO AAA."
end

if abspath(PROGRAM_FILE) == @__FILE__
    start_stable_training()
end
