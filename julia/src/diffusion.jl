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

"""
    forward_mask_functional(md, tokens, t) → (masked_tokens, mask)

Versão sem mutação de forward_mask — compatível com Zygote.withgradient.
Usa ifelse em vez de indexação in-place.
"""
function forward_mask_functional(md::MaskDiffusion, tokens::Vector{Int}, t::Float32)
    mask = rand(Float32, length(tokens)) .< t
    masked_tokens = ifelse.(mask, md.mask_token_id, tokens)
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
