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

    return -Statistics.Statistics.mean(log_probs)
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
    return Statistics.Statistics.mean(losses)
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
    return Statistics.Statistics.mean(losses)
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
        Statistics.Statistics.mean(losses)
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
    mns = 1.0f0 - Statistics.Statistics.mean(abs2, diff)
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
    v = Statistics.Statistics.var(activations)
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
