using Pkg; Pkg.activate(@__DIR__)
using BSON: @load, @save
using Flux, Statistics, Functors, LinearAlgebra, BSON, Optimisers, Zygote

# Carregar a arquitetura verdadeira do modelo
include("src/transformer.jl")

device = cpu

# Lórica de Vocabulário CAFUNE Revision 3 (Manual Parser)
function load_tokenizer()
    # Caminho absoluto para evitar erros de CWD
    vocab_path = normpath(joinpath(@__DIR__, "..", "vocab.json"))
    char2id = Dict{String, Int}()
    id2char = Dict{Int, String}()
    
    if isfile(vocab_path)
        content = read(vocab_path, String)
        # Regex global para pegar todas as chaves e valores do char2id
        # Ele vai pegar pares tipo " ": 5 ou "a": 30
        for m in eachmatch(r"\"(.*?)\":\s*(\d+)", content)
            key, id_str = m.captures
            if key == "vocab_size" continue end
            id = parse(Int, id_str)
            char2id[key] = id
            id2char[id] = key
        end
        # Debug Log NATIVO no Terminal
        println("  [LOG-MOTOR] Arquivo vocab.json lido em: $vocab_path")
        println("  [LOG-MOTOR] Tokens mapeados: $(length(char2id))")
        if haskey(char2id, " ")
            println("  [LOG-MOTOR] Canal de comunicacao VALIDADO (Espaco ID $(char2id[" "]))")
        else
            println("  [LOG-MOTOR] ALERTA: Canal de comunicacao corrompido (Espaco nao encontrado)")
        end
    else
        println("  [ERROR-MOTOR] FALHA CRITICA: Dicionario nao encontrado em $vocab_path")
    end
    return char2id, id2char
end

const (char2id, id2char) = load_tokenizer()
@info "[PULSE CHECK] ID do Espaço ' ' mapeado para: $(get(char2id, " ", "N/A"))"

include("src/snn_core.jl")

function generate_neural_response(prompt::AbstractString; max_len=100)
    # O usuário treinou rodando do diretório raiz do Lira, então o BSON parou lá
    model_path = joinpath(@__DIR__, "..", "..", "cafune_model.bson")
    if !isfile(model_path)
        return "Sinal de silício ausente (model.bson não encontrado)."
    end
    
    # Carregar modelo salvo
    @load model_path m_cpu
    model = m_cpu |> device
    
    # Tokenizacao Oficial (1-based Julia index)
    input_ids = Int[]
    for c in prompt
        # Busca ID ou [UNK] (ID 1)
        push!(input_ids, get(char2id, string(c), 1) + 1) 
    end

    if length(input_ids) > 128
        input_ids = input_ids[end-127:end]
    elseif length(input_ids) < 128
        # Padding [PAD] (ID 0 + 1 = 1)
        input_ids = vcat(ones(Int, 128 - length(input_ids)), input_ids)
    end
    
    # Iniciar Motor Híbrido: SNN Decoder
    # Usando o vocab de 65 (nossas letras treinadas) e 30 timesteps para pensar
    snn_decoder = SpikingDecoder(65; timesteps=30)
    
    # Loop de Geracao (Spiking)
    output_tokens = Int[]
    curr_input = vec(input_ids) 
    
    for _ in 1:20 
        logits = model(curr_input) 
        
        # Focar nas 65 letras treinadas
        last_col = logits[1:65, end]
        
        # --- [SNN SAMPLING: Cognição Neuromórfica] ---
        # O SNN converte a estatística em biologia e decide quando já tem confiança para disparar (Early Stopping)
        next_token, conf = snn_decoder(last_col)
        
        char_preview = get(id2char, next_token - 1, "")
        if conf >= 0.85f0
            println(" ⚡ Estalo Cognitivo (Early Stopping)! Token: '$(char_preview)' | Confiança: $(round(conf * 100, digits=2))%")
        else
            println(" ⏳ Reflexão 11D Completa. Token: '$(char_preview)' | Confiança final: $(round(conf * 100, digits=2))%")
        end
        
        push!(output_tokens, next_token)
        curr_input = vcat(curr_input[2:end], [next_token])
    end
    
    # Converter IDs em texto real usando o id2char
    res = ""
    for tid in output_tokens
        id = tid - 1 # Volta para 0-based do JSON
        char = get(id2char, id, "")
        if !isempty(char) && !startswith(char, "[")
            res *= char
        end
    end
    
    final_res = isempty(strip(res)) ? "..." : res
    
    @info "[MOTOR-OUT] Resposta: $final_res"
    return final_res, output_tokens, input_ids
end

# [RLAIF ONLINE LEARNING]
function train_rlaif_step(prompt_ids, response_ids, reward::Float32)
    @info "--- [RLAIF GRADIENT STEP] ---"
    @info "Reward Recebido: $reward"
    
    model_path = joinpath(@__DIR__, "..", "..", "cafune_model.bson")
    if !isfile(model_path) return end
    
    @load model_path m_cpu
    model = m_cpu |> device
    
    # Optimizer leve para o RLAIF online
    opt = Optimisers.Adam(1f-4)
    opt_state = Optimisers.setup(opt, model)
    
    # Reinforcement Learning: Reward Weighted Cross Entropy
    # Queremos maximizar a probabilidade dos tokens que ganharam reward alto
    l, g = Zygote.withgradient(model) do m
        # Reconstrói o contexto para cada token gerado
        loss = 0f0
        ctx = prompt_ids
        for tid in response_ids
            logits = m(ctx)
            # Logits for the next token
            target_logit = logits[tid, end]
            # Softmax log-prob
            log_prob = target_logit - log(sum(exp.(logits[:, end])))
            # Reward weighting (RL advantage-like)
            loss -= reward * log_prob
            # Shift context
            ctx = vcat(ctx[2:end], [tid])
        end
        return loss / length(response_ids)
    end
    
    opt_state, model = Optimisers.update!(opt_state, model, g[1])
    
    # Salvar pesos atualizados
    m_cpu = model |> cpu
    @save model_path m_cpu
    @info "[✓] Realinhamento Neural Concluido."
end

if !isempty(ARGS)
    println(generate_neural_response(ARGS[1]))
else
    println("Uso: julia inference.jl \"Prompt aqui\"")
end
