"""
    sampling.jl — Inferência Iterativa (o processo reverse do LLaDA)

    Algoritmo de geração:
    
    1. Começa com sequência 100% mascarada
    2. Para cada passo t = T, T-1, ..., 1:
       a. Modelo prevê logits para TODOS os tokens
       b. Calcula confiança de cada posição mascarada
       c. Desmascara as posições mais confiantes
       d. Tokens já desmascarados ficam fixos
    3. Resultado: texto gerado
    
    Estratégias de unmasking:
    - Confidence-based (padrão): desmascara as mais confiantes primeiro
    - Random: desmascara aleatoriamente (mais diversidade)
    - Scheduled: desmascara proporcionalmente ao schedule
"""

using Random, Printf

# ============================================================
#  Funções Auxiliares
# ============================================================

"""
    sample_from_logits(logits, temperature) → Int

Amostra um token dos logits com temperatura.
- temperature=0: argmax (greedy)
- temperature=1: sampling padrão
- temperature>1: mais aleatório
"""
function sample_from_logits(logits::Vector{Float32}; temperature::Float32=1f0)
    if temperature == 0f0
        return argmax(logits)
    end

    # Suavizar com temperatura
    scaled = logits ./ temperature
    # Softmax estável
    scaled .-= maximum(scaled)
    probs = exp.(scaled)
    probs ./= sum(probs)

    # Amostragem categórica
    r = rand(Float32)
    cumsum_p = 0f0
    for (i, p) in enumerate(probs)
        cumsum_p += p
        if r <= cumsum_p
            return i
        end
    end
    return length(probs)  # fallback
end

"""
    confidence_score(logits) → Float32

Score de confiança para uma posição: probabilidade do token mais provável.
Quanto maior, mais "certo" o modelo está sobre esse token.
"""
function confidence_score(logits::Vector{Float32})
    scaled = logits .- maximum(logits)
    probs = exp.(scaled)
    probs ./= sum(probs)
    return maximum(probs)
end

# ============================================================
#  Gerador Principal
# ============================================================

"""
    generate(model, md, seq_len; kwargs...) → Vector{Int}

Gera uma sequência de texto do zero usando denoising iterativo.

# Argumentos
- `model`: transformer bidirecional treinado
- `md`: configuração da difusão
- `seq_len`: comprimento da sequência a gerar

# Keywords
- `num_steps=50`: passos de denoising
- `temperature=1.0`: temperatura do sampling
- `strategy=:confidence`: estratégia de unmasking (:confidence ou :random)

# Retorna
Vector{Int} com os IDs dos tokens gerados
"""
function generate(model::BidirectionalTransformer, md::MaskDiffusion, seq_len::Int;
                  num_steps::Int=md.num_steps,
                  temperature::Float32=1f0,
                  strategy::Symbol=:confidence,
                  verbose::Bool=false)

    # ── Passo 0: Inicializar com sequência totalmente mascarada ──
    x = fill(md.mask_token_id, seq_len)   # [MASK, MASK, ..., MASK]

    # ── Loop de denoising ──
    for step in num_steps:-1:1
        t_current = Float32(step / num_steps)
        t_next    = Float32((step - 1) / num_steps)

        # Posições ainda mascaradas
        mask_positions = findall(x .== md.mask_token_id)
        if isempty(mask_positions)
            break  # Totalmente desmascarado!
        end

        # Forward pass do modelo
        logits = model(x)   # (vocab_size, seq_len)

        # Número de tokens para desmascara neste passo
        # Proporcional à redução de t: Δt = t_current - t_next
        n_masked = length(mask_positions)
        n_to_unmask = round(Int, n_masked * (t_current - t_next) / t_current)
        n_to_unmask = max(1, n_to_unmask)  # pelo menos 1
        n_to_unmask = min(n_to_unmask, n_masked)

        if strategy == :confidence
            # Desmascara as posições que o modelo está mais confiante
            confidences = [confidence_score(logits[:, pos]) for pos in mask_positions]
            sorted_idx  = sortperm(confidences, rev=true)
            positions_to_unmask = mask_positions[sorted_idx[1:n_to_unmask]]
        else  # :random
            positions_to_unmask = sample(mask_positions, n_to_unmask, replace=false)
        end

        # Desmascara e amostrar tokens
        for pos in positions_to_unmask
            x[pos] = sample_from_logits(logits[:, pos], temperature=temperature)
        end

        if verbose
            still_masked = count(x .== md.mask_token_id)
            @printf("  Passo %2d/%d | Desmascarados: %d | Mascarados: %d\n",
                    num_steps - step + 1, num_steps, seq_len - still_masked, still_masked)
        end
    end

    # Forçar qualquer restante mascarado para o token mais provável
    remaining = findall(x .== md.mask_token_id)
    if !isempty(remaining)
        logits = model(x)
        for pos in remaining
            x[pos] = argmax(logits[:, pos])
        end
    end

    return x
end

"""
    generate_with_prompt(model, md, prompt_tokens, total_len; kwargs...)

Geração com prompt: fixa os tokens do prompt e gera o restante.
Útil para completion e instruction-following.

# Exemplo
    prompt = tokenize("O gato")
    output = generate_with_prompt(model, md, prompt, 20)
    # → "O gato é preto e muito inteligente."
"""
function generate_with_prompt(model::BidirectionalTransformer, md::MaskDiffusion,
                               prompt_tokens::Vector{Int}, total_len::Int;
                               num_steps::Int=md.num_steps,
                               temperature::Float32=1f0,
                               verbose::Bool=false)

    prompt_len = length(prompt_tokens)
    @assert total_len > prompt_len "total_len deve ser maior que o comprimento do prompt"

    # Inicializar: prompt fixo + resto mascarado
    x = fill(md.mask_token_id, total_len)
    x[1:prompt_len] = prompt_tokens

    if verbose
        @printf("🎯 Gerando %d tokens após prompt de %d tokens...\n",
                total_len - prompt_len, prompt_len)
    end

    for step in num_steps:-1:1
        t_current = Float32(step / num_steps)
        t_next    = Float32((step - 1) / num_steps)

        # Apenas posições APÓS o prompt são mascaradas
        mask_positions = findall(i -> i > prompt_len && x[i] == md.mask_token_id, 1:total_len)
        if isempty(mask_positions)
            break
        end

        logits = model(x)

        n_masked = length(mask_positions)
        n_to_unmask = max(1, round(Int, n_masked * (t_current - t_next) / t_current))
        n_to_unmask = min(n_to_unmask, n_masked)

        confidences = [confidence_score(logits[:, pos]) for pos in mask_positions]
        sorted_idx  = sortperm(confidences, rev=true)

        for pos in mask_positions[sorted_idx[1:n_to_unmask]]
            x[pos] = sample_from_logits(logits[:, pos], temperature=temperature)
        end
    end

    # Completar qualquer restante
    remaining = findall(i -> i > prompt_len && x[i] == md.mask_token_id, 1:total_len)
    if !isempty(remaining)
        logits = model(x)
        for pos in remaining
            x[pos] = argmax(logits[:, pos])
        end
    end

    return x
end
