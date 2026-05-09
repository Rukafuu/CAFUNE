using Flux
using BSON: @load, @save
using Statistics
using Printf

println("Iniciando Pipeline RLAIF para Reservatório Hipercúbico 11D")

# Carregar o Core
include("src/transformer.jl")
include("src/snn_core.jl")
include("inference.jl") # Para pegar char2id / id2char

# Carregar Transformer Base (Professor interno)
model_path = joinpath(@__DIR__, "..", "..", "cafune_model.bson")
if !isfile(model_path)
    error("Erro: cafune_model.bson não encontrado! O Transformer deve estar treinado primeiro.")
end
@load model_path m_cpu
transformer = m_cpu |> cpu # Vamos treinar na CPU para simplificar a memória da SNN

# Inicializar o Aluno SNN (O Maxilar de 11D)
snn_path = joinpath(@__DIR__, "..", "snn_weights.bson")
if isfile(snn_path)
    println("[+] Carregando SNN Treinada anteriormente...")
    @load snn_path snn_decoder
else
    println("[+] Criando novo Reservatório 11D (SNN Decoder)...")
    snn_decoder = SpikingDecoder(65; timesteps=30, D=11)
end

# Apenas a camada Readout (lif_out) e a W_in serão treinadas. A 11D (lif_res) é FIXA.
params_to_train = Flux.params(snn_decoder.W_in, snn_decoder.lif_out.W, snn_decoder.lif_out.b)
optimizer = Flux.ADAM(0.005)

# Prompts do "Berçário" para estimular a SNN a falar
bercario_prompts = [
    "oi cafuné",
    "como voce esta hoje?",
    "me diga quem voce eh",
    "fale comigo por favor",
    "voce sente alguma coisa?"
]

# Avaliador Externo (Ling 2.6 via Python)
function get_reward(prompt::String, response::String)
    # Proteção contra chamadas vazias
    if isempty(strip(response)) || length(response) < 2
        return 0.1
    end
    
    # Chama o script Python do OpenRouter
    py_script = joinpath(@__DIR__, "..", "python", "rlaif_evaluator.py")
    
    try
        # Pega o stdout do script Python
        output = read(`python $py_script "$prompt" "$response"`, String)
        score = parse(Float64, strip(output))
        return score
    catch e
        println("  [!] Erro na API do OpenRouter: ", e)
        return 0.1
    end
end

println("\n Iniciando Sala de Aula RLAIF (Policy Gradient)...")

epochs = 3
samples_per_prompt = 3 # Gera 3 variações estocásticas por prompt

for epoch in 1:epochs
    println("\n=== EPOCH $epoch ===")
    
    for prompt in bercario_prompts
        println("\n📝 Prompt: \"$prompt\"")
        
        # 1. Transformar prompt em Contexto (Logits Base do Transformer)
        input_ids = [get(char2id, string(c), 1) + 1 for c in prompt]
        curr_input = vec(input_ids)
        
        # Faz o Transformer gerar o sinal base
        logits_base = transformer(curr_input)
        last_col = logits_base[1:65, end]
        
        # 2. Amostragem Estocástica (SNN gera respostas diferentes graças ao Poisson Encoder)
        variations = []
        for s in 1:samples_per_prompt
            # Zygote.ignore: A geração do texto é separada do cálculo do gradiente
            # porque a roleta russa não é diferenciável.
            # Vamos apenas coletar a frase gerada para mandar pro Juiz.
            
            # Precisamos gerar vários tokens (uma frase curta, 15 letras max pra não pesar a API agora)
            temp_input = copy(curr_input)
            output_tokens = Int[]
            
            for step in 1:15
                l_base = transformer(temp_input)[1:65, end]
                # Passa pelo SNN 
                nxt = snn_decoder(l_base)
                push!(output_tokens, nxt)
                temp_input = vcat(temp_input[2:end], [nxt])
            end
            
            # Decode
            res_str = ""
            for tid in output_tokens
                ch = get(id2char, tid - 1, "")
                if !startswith(ch, "[")
                    res_str *= ch
                end
            end
            push!(variations, (tokens=output_tokens, text=res_str))
        end
        
        # 3. Avaliação Ling 2.6 (O Julgamento)
        best_reward = 0.0
        best_var = variations[1]
        
        for (idx, var) in enumerate(variations)
            reward = get_reward(prompt, var.text)
            println("   Var $idx: \"$(var.text)\" => Nota: $reward")
            if reward > best_reward
                best_reward = reward
                best_var = var
            end
        end
        
        # 4. Atualização Sináptica (Policy Gradient na SNN)
        # Vamos recompensar o caminho de tensores que gerou a melhor variação
        if best_reward > 0.2
            println("   💡 Reforçando SNN para a resposta Vencedora com Loss multiplicada por -$best_reward")
            
            grads = Flux.gradient(params_to_train) do
                # Recriamos o forward pass APENAS para calcular a entropia cruzada da SNN
                # simulando a escolha da "melhor variação"
                loss = 0.0f0
                temp_input = copy(curr_input)
                
                for target_token in best_var.tokens
                    l_base = transformer(temp_input)[1:65, end]
                    
                    # Passo SNN "manual" para ter gradientes contínuos
                    probs = exp.((l_base .- maximum(l_base)) ./ 1.0f0)
                    probs ./= sum(probs)
                    p_mat = reshape(probs, snn_decoder.vocab_size, 1)
                    pulse_train = poisson_encoder(p_mat, snn_decoder.timesteps)
                    
                    # Passar pelo reservatório
                    N_res = size(snn_decoder.lif_res.W, 1)
                    v_res = zeros(Float32, N_res, 1)
                    v_out = zeros(Float32, snn_decoder.vocab_size, 1)
                    
                    for t in 1:snn_decoder.timesteps
                        x_11D = snn_decoder.W_in * pulse_train[t, :, :]
                        v_res, _ = snn_decoder.lif_res(v_res, x_11D)
                        v_out, _ = snn_decoder.lif_out(v_out, v_res) # Sem spikes duros aqui, usamos a voltagem (surrogate)
                    end
                    
                    # Policy Gradient Loss: CrossEntropy com a Voltagem Final da Membrana (v_out)
                    # Queremos maximizar a voltagem do 'target_token' ponderado pelo best_reward
                    v_vec = vec(v_out)
                    v_softmax = exp.(v_vec) ./ sum(exp.(v_vec))
                    
                    # Proteção log(0)
                    prob_target = clamp(v_softmax[target_token], 1e-7, 1.0)
                    
                    # Loss Negativa = Recompensa
                    loss += -Float32(best_reward) * log(prob_target)
                    
                    # Shift
                    temp_input = vcat(temp_input[2:end], [target_token])
                end
                return loss / length(best_var.tokens)
            end
            
            Flux.Optimise.update!(optimizer, params_to_train, grads)
        else
            println("Nenhuma resposta foi boa suficiente. Ignorando atualização (Noise Rejection).")
        end
    end
    
    # Salvar o maxilar SNN a cada época
    @save snn_path snn_decoder
    println(" Membranas 11D salvas com sucesso em snn_weights.bson")
end

println("Treinamento RLAIF Concluído! O Motor Híbrido evoluiu.")
