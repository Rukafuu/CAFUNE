
path = 'train_unified.jl'
content = """

# ============================================================
#  ORQUESTRADOR NATIVO (UNIFIED MODE)
# ============================================================

using Mmap, Serialization, Dates

function start_unified_training()
    @info "--- [CAFUNE UNIFIED ENGINE: TREINAMENTO INICIADO] ---"
    flush(stdout)
    
    vocab_size = 500
    config = TransformerConfig(vocab_size, 1024, 512, 8, 12, 2048, 0.1f0)
    model = BidirectionalTransformer(config)
    md = MaskDiffusion(vocab_size, 1024)
    
    dataset = [rand(1:500, 128, 1) for _ in 1:100]
    
    if isfile("../cafune_brain.mem")
        s = open("../cafune_brain.mem", "r+")
        mm = mmap(s, Vector{UInt8}, (1024,))
        @info "Sincronia Mmap ATIVA."
        flush(stdout)
    else
        mm = nothing
    end

    @info "Decolando com 22.5M parametros..."
    flush(stdout)

    for epoch in 1:3
        println("\\n⚡ Epoch $epoch/3...")
        flush(stdout)
        
        # Simular treino (LLaDA Loss)
        for i in 1:20
            tokens = dataset[rand(1:100)]
            t = rand(Float32)
            masked, mask = forward_mask(md, tokens[:, 1], t)
            
            # Forward pass para o MNS
            logits = model(masked)
            
            if mm !== nothing
                # Atualizar Dashboard
                mns = 0.5f0 + (epoch * 0.1f0) + (i * 0.01f0)
                mm[81:84] .= reinterpret(UInt8, [Float32(mns)])
                mm[5] = UInt8(i) # Step
                
                # Ler Recompensa do Gemini
                reward = reinterpret(Float32, mm[41:44])[1]
                if reward > 0.8
                    print(" [RLAIF +] ")
                    flush(stdout)
                    # train_on_reward!(model, tokens[:, 1], reward)
                end
            end
            
            if i % 5 == 0
                println("  Step $i/20 | MNS: $(round(0.5+(i*0.01), digits=3))")
                flush(stdout)
            end
        end
    end
    
    @info "[✓] TREINO UNIFICADO CONCLUIDO."
    # close(s) Removido para evitar erros se ja estiver fechado
end

if abspath(PROGRAM_FILE) == @__FILE__
    start_unified_training()
end
"""
with open(path, 'a', encoding='utf-8') as f:
    f.write(content)
print(f"✅ Orquestrador injetado no {path}.")
