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

# ============================================================
#  Geração Iterativa (Reverse Diffusion)
# ============================================================

"""
    generate(model, md, seq_len; num_steps, temperature, valid_ids) → Vector{Int}

Gera uma sequência via denoising iterativo (LLaDA-style).

Começa com tudo mascarado e vai revelando os tokens mais
confiantes a cada passo, do mais ruidoso ao mais limpo.

# Argumentos
- `model` — BidirectionalTransformer treinado
- `md::MaskDiffusion` — configuração do processo
- `seq_len::Int` — comprimento da sequência a gerar
- `num_steps::Int` — número de passos de denoising (padrão: 10)
- `temperature::Float32` — temperatura de amostragem (padrão: 1.0)
- `valid_ids::Union{Nothing,Vector{Int}}` — IDs permitidos na geração.
  Se fornecido, logits de tokens fora deste conjunto são zerados.
  Útil para filtrar tokens BPE inválidos/longos demais.
"""
function generate(model, md::MaskDiffusion, seq_len::Int;
                  num_steps::Int=10, temperature::Float32=1.0f0,
                  valid_ids::Union{Nothing,Vector{Int}}=nothing)

    # Começa completamente mascarado
    tokens = fill(md.mask_token_id, seq_len)

    for step in num_steps:-1:1
        # Fração de tokens a revelar neste passo
        t_now  = Float32(step / num_steps)
        t_next = Float32((step - 1) / num_steps)

        # Prediz logits para todos os tokens
        logits = model(tokens)   # (vocab_size+1, seq_len)

        # Zera o logit do próprio MASK para nunca amostrar MASK como resposta
        logits[md.mask_token_id, :] .= -Inf32

        # Filtra tokens inválidos (BPE corrompidos / muito longos)
        if valid_ids !== nothing
            all_ids = 1:size(logits, 1)
            invalid_ids = setdiff(all_ids, valid_ids)
            logits[invalid_ids, :] .= -Inf32
        end

        # Amostragem com temperatura
        probs = similar(logits)
        for i in 1:seq_len
            l = logits[:, i] ./ temperature
            l .-= maximum(l)
            e  = exp.(l)
            probs[:, i] = e ./ sum(e)
        end

        # Confiança = probabilidade máxima em cada posição
        confidences = [maximum(probs[:, i]) for i in 1:seq_len]

        # Quantos tokens revelar neste passo
        n_masked_now  = count(==(md.mask_token_id), tokens)
        n_masked_next = round(Int, t_next * seq_len)
        n_to_reveal   = max(0, n_masked_now - n_masked_next)

        if n_to_reveal > 0 && n_masked_now > 0
            # Índices ainda mascarados
            masked_idx = findall(==(md.mask_token_id), tokens)
            # Ordena por confiança decrescente e revela os mais confiantes
            conf_masked = confidences[masked_idx]
            order = sortperm(conf_masked, rev=true)
            to_reveal = masked_idx[order[1:min(n_to_reveal, length(order))]]

            for i in to_reveal
                # Amostra token da distribuição
                r = rand(Float32)
                cumsum_p = 0.0f0
                chosen = 1
                for k in 1:size(probs, 1)
                    cumsum_p += probs[k, i]
                    if r <= cumsum_p
                        chosen = k
                        break
                    end
                end
                tokens[i] = chosen
            end
        end
    end

    return tokens
end
