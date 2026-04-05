"""
    training.jl — Loop de Treino do LLaDA

    Objetivo de treino (ELBO simplificado):
    
    L = E_{t ~ U[0,1], x_t ~ q(x_t|x_0)} [ Σ_{i: x_t[i]=MASK} -log p_θ(x_0[i] | x_t) ]
    
    Na prática: cross-entropy só nas posições mascaradas.
    
    Isso é matematicamente equivalente a otimizar o ELBO do processo
    de difusão discreta mascarada. (Ver LLaDA paper, Eq. 5)
"""

using Statistics, Printf, Random, Zygote, Optimisers

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

    return -mean(log_probs)
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
    return mean(losses)
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
    return mean(losses)
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
        mean(losses)
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
#  Loop de Treino
# ============================================================

function train!(model::BidirectionalTransformer, md::MaskDiffusion,
                dataset::AbstractVector;
                epochs::Int=10, log_every::Int=100)

    # Inicializar otimizador moderno (AdamW do Optimisers.jl)
    opt = Optimisers.Adam(3f-4)
    opt_state = Optimisers.setup(opt, model)
    
    total_steps = 0
    best_loss = Inf

    @printf("\n[CAFUNE Training] Fase 2: Autodiff Ativado\n")
    @printf("   Treinando %d parametros...\n", count_params(model))
    @printf("   Dataset: %d sequencias\n\n", length(dataset))

    for epoch in 1:epochs
        epoch_loss = 0f0
        indices = randperm(length(dataset))

        for (step, idx) in enumerate(indices)
            tokens = dataset[idx]
            
            # Step de treino autodiff
            loss, opt_state, model = train_step!(model, md, tokens, opt_state)
            
            epoch_loss += loss
            total_steps += 1

            if total_steps % log_every == 0
                avg_loss = epoch_loss / step
                @printf("  Epoch %d | Step %d | Loss: %.4f\n", epoch, total_steps, avg_loss)
            end
        end

        avg_loss = epoch_loss / length(dataset)
        @printf("✅ Epoch %d concluída | Loss média: %.4f\n", epoch, avg_loss)

        if avg_loss < best_loss
            best_loss = avg_loss
            @printf("   ⭐ Melhor loss até agora!\n")
        end
    end

    @printf("\nTreino fase 2 concluido! Best loss: %.4f\n", best_loss)
    return model
end
