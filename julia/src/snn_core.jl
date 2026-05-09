# ============================================================
# snn_core.jl — Módulo Spiking Neural Network (SNN) para o CAFUNE
# ============================================================
# Abordagem: Leaky Integrate-and-Fire (LIF)
# Paradigma: O processamento no tempo. Substitui a ativação
# contínua (ReLU/GELU) por disparos binários (Spikes).
# ============================================================

using Flux
using Flux: @functor
using Zygote

"""
    LIFCell

Célula "Leaky Integrate-and-Fire". Age como uma camada densa, 
mas com memória de voltagem e disparo em Timesteps.
"""
mutable struct LIFCell
    W::Matrix{Float32}        # Pesos Sinápticos
    b::Vector{Float32}        # Viés (Threshold basal)
    decay_array::Vector{Float32} # TSM PLIF: Decay adaptativo por timestep
    V_thresh::Float32         # Voltagem necessária para o Disparo (Spike = 1)
    tsm_gamma::Vector{Float32}# TSM: Parâmetro temporal treinável por timestep
end

@functor LIFCell (W, b, decay_array, tsm_gamma) # Zygote rastreia tudo para gradientes

function LIFCell(in_dim::Int, out_dim::Int; decay=0.9f0, V_thresh=1.0f0, timesteps=30)
    # Inicialização Xavier simplificada
    scale = Float32(sqrt(2.0 / (in_dim + out_dim)))
    W = randn(Float32, out_dim, in_dim) .* scale
    b = zeros(Float32, out_dim)
    decay_array = fill(Float32(decay), timesteps)
    tsm_gamma = ones(Float32, timesteps) # Inicializa escala TSM como 1.0 (neutro)
    return LIFCell(W, b, decay_array, Float32(V_thresh), tsm_gamma)
end

"""
    heaviside_surrogate(v, thresh)

Função Degrau com Gradiente Substituto (Surrogate Gradient).
No Forward Pass, age como um Spike biológico duro (0 ou 1).
No Backward Pass (Zygote), age como a derivada de uma curva Sigmoide suave.
Isso permite que o erro (Reward) escorra pelo tempo (BPTT).
"""
heaviside_surrogate(v, thresh) = Float32.(v .>= thresh)

Zygote.@adjoint function heaviside_surrogate(v, thresh)
    y = heaviside_surrogate(v, thresh)
    function pullback(Δy)
        # Derivada da função sigmoide: σ(x) = 1/(1+exp(-βx)) -> σ'(x) = β * σ(x) * (1 - σ(x))
        # Ajustamos o beta (steepness) para controlar o quão largo é o "funil" do gradiente.
        beta = 5.0f0
        x = v .- thresh
        sig = 1.0f0 ./ (1.0f0 .+ exp.(-beta .* x))
        # O gradiente que flui para trás é a derivada pontual vezes o gradiente recebido (Δy)
        grad_v = Δy .* beta .* sig .* (1.0f0 .- sig)
        
        return (grad_v, nothing) # nothing = gradiente do thresh não será treinado
    end
    return y, pullback
end

"""
    (cell::LIFCell)(v_prev, x, t)

Forward pass para um único TIMESTEP com arquitetura Pre-Spike Residual e TSM.
"""
function (cell::LIFCell)(v_prev::AbstractMatrix, x::AbstractMatrix, t::Int)
    # 1. Integração Sináptica com Pre-Spike Residual e TSM
    # A entrada analógica é escalonada pelo tsm_gamma atual para modular o ritmo
    excitation = (cell.W * x .+ cell.b) .* cell.tsm_gamma[t]
    v_mem = (v_prev .* cell.decay_array[t]) .+ excitation
    
    # 2. Mecanismo de Disparo (Fire) usando Surrogate Gradient
    spikes = heaviside_surrogate(v_mem, cell.V_thresh)
    
    # 3. Hard Reset: Zera a carga se disparou
    v_next = v_mem .* (1.0f0 .- spikes)
    
    return v_next, spikes
end

# ============================================================
# Simulador de Ondas Cerebrais (Encoder)
# ============================================================

"""
    poisson_encoder(x::AbstractMatrix, timesteps::Int)

Converte uma matriz contínua (ex: um Embedding) em um trem de pulsos 
usando um processo de Poisson. 
Se a entrada for forte (ex: 0.8), ele vai disparar spikes em 80% do tempo.
"""
function poisson_encoder(x::AbstractMatrix, timesteps::Int)
    # Cria uma simulação onde os valores de x definem a probabilidade de Spike
    # Retorna Tensor 3D (Timesteps x Dims x Batch)
    spikes = zeros(Float32, timesteps, size(x)...)
    for t in 1:timesteps
        # Um neurônio dispara se um rand() for menor que a força da entrada (clamped 0 a 1)
        prob = clamp.(x, 0.0f0, 1.0f0)
        spikes[t, :, :] .= Float32.(rand(Float32, size(x)...) .< prob)
    end
    return spikes
end

# ============================================================

"""
    build_hypercube_connectivity(D::Int)

Cria a matriz de adjacência para um hipercubo de D dimensões.
Dois neurônios se conectam se a distância de Hamming entre seus índices for 1.
Para D=11, gera uma matriz 2048x2048 onde cada neurônio tem exatamente 11 vizinhos.
"""
function build_hypercube_connectivity(D::Int)
    N = 2^D
    W = zeros(Float32, N, N)
    for i in 0:(N-1)
        for j in 0:(N-1)
            # Operação bit a bit XOR: conta quantos bits são diferentes
            if count_ones(i ⊻ j) == 1
                # Peso sináptico excitatório (+1) ou inibitório (-1) aleatório para dinâmica caótica
                W[i+1, j+1] = rand() > 0.5 ? 1.0f0 : -0.5f0
            end
        end
    end
    # Normalizar o ganho de energia pela dimensão D para não explodir
    return W ./ Float32(D)
end

"""
    SpikingDecoder

Evoluído para usar Reservatório 11D (2048 neurônios).
Os logits entram no reservatório, reverberam na estrutura 11D,
e depois são lidos pela camada de saída (65 tokens).
"""
mutable struct SpikingDecoder
    W_in::Matrix{Float32}     # Projeta de Vocab (65) -> Hypercube (2048)
    lif_res::LIFCell          # Reservatório 11D
    lif_out::LIFCell          # Readout Layer -> Vocab (65)
    value_head::Any           # UniGRPO Value Head
    timesteps::Int
    vocab_size::Int
end

@functor SpikingDecoder (W_in, lif_res, lif_out, value_head)

function SpikingDecoder(vocab_size::Int; timesteps=30, D=11)
    println("🌌 [SNN] Inicializando Topologia Hipercúbica $(D)D (Dimensões)")
    N_res = 2^D # 2048 neurônios no reservatório
    
    # 1. Matrizes de Projeção (Skip-Connection de Identidade para as 65 primeiras)
    W_in = randn(Float32, N_res, vocab_size) .* 0.05f0
    for i in 1:vocab_size
        W_in[i, i] = 1.0f0 # Conexão direta forte da palavra para o reservatório
    end
    
    # 2. LIF do Reservatório (Estrutura 11D Fixa)
    W_11D = build_hypercube_connectivity(D)
    lif_res = LIFCell(N_res, N_res; decay=0.90f0, V_thresh=1.0f0, timesteps=timesteps)
    lif_res.W .= W_11D # Ocupa os pesos com a geometria restrita 11D
    
    # 3. LIF de Saída (Readout linear SNN)
    lif_out = LIFCell(N_res, vocab_size; decay=0.80f0, V_thresh=2.0f0, timesteps=timesteps)
    
    # 4. Value Head (UniGRPO) para Early Stopping pulsado
    value_head = Chain(Dense(N_res, 128, gelu), Dense(128, 1, sigmoid))
    
    return SpikingDecoder(W_in, lif_res, lif_out, value_head, timesteps, vocab_size)
end

"""
    (decoder::SpikingDecoder)(logits)

Processa logits projetando para a 11D.
"""
function (decoder::SpikingDecoder)(logits::AbstractVector)
    # Suavização
    probs = exp.((logits .- maximum(logits)) ./ 1.0f0)
    probs ./= sum(probs)
    
    # Encoder Poisson no VOCAB (65)
    p_mat = reshape(probs, decoder.vocab_size, 1)
    pulse_train = poisson_encoder(p_mat, decoder.timesteps)
    
    # Estados de Memória Iniciais
    N_res = size(decoder.lif_res.W, 1)
    v_mem_res = zeros(Float32, N_res, 1)
    v_mem_out = zeros(Float32, decoder.vocab_size, 1)
    
    # O Tempo passa no Universo 11D...
    for t in 1:decoder.timesteps
        # 1. Pegar o pulso atual (Vocab)
        x_t_vocab = pulse_train[t, :, :]
        
        # 2. Projetar o pulso para o espaço 11D (excitar o reservatório)
        x_t_11D = decoder.W_in * x_t_vocab
        
        # 3. Dinâmica Caótica do Reservatório (A Magia 11D)
        v_mem_res, spikes_res = decoder.lif_res(v_mem_res, x_t_11D, t)
        
        # === VALUE HEAD (UniGRPO) ===
        # Lê a Tensão do cérebro para prever se a confiança já está alta o suficiente
        confidence = decoder.value_head(v_mem_res)[1]
        
        # 4. Readout Layer (tentar decodificar a resposta)
        v_mem_out, spikes_out = decoder.lif_out(v_mem_out, spikes_res, t)
        
        # Early Stopping Cognitivo: Parada ultra-rápida!
        if confidence >= 0.85f0 || sum(spikes_out) > 0
            active_ids = findall(vec(spikes_out) .> 0)
            if !isempty(active_ids)
                return rand(active_ids), confidence # Retorna Token e Confiança
            end
        end
    end
    
    # Se a energia dissipar antes da resposta clara, escolhemos com base na voltagem residual
    # Amostragem com temperatura SNN (T=1.0)
    v_vec = vec(v_mem_out)
    v_probs = exp.((v_vec .- maximum(v_vec)) ./ 1.0f0)
    v_probs ./= sum(v_probs)
    
    r = rand()
    acc = 0.0
    for (i, p) in enumerate(v_probs)
        acc += p
        if r <= acc
            return i, 0.0f0 # Fallback com confiança 0
        end
    end
    return argmax(v_vec), 0.0f0
end
