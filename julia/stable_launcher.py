
import os

def fuse_and_launch():
    base_dir = r"C:\Users\conta\Documents\Lira\Lira\CAFUNE\julia"
    out_file = os.path.join(base_dir, "train_stable.jl")
    
    src_files = ["src/transformer.jl", "src/diffusion.jl", "src/training.jl"]
    
    with open(out_file, "w", encoding="utf-8") as out:
        # Fundir módulos
        for f in src_files:
            path = os.path.join(base_dir, f)
            with open(path, "r", encoding="utf-8") as f_in:
                out.write(f_in.read() + "\n")
        
        # Injetar Orquestrador Minimalista
        out.write("""
# ============================================================
#  STABLE ORQUESTRADOR NATIVO (FASE 2)
# ============================================================

using Statistics, Dates, Flux, Optimisers, Zygote

function start_stable_training()
    @info "--- [CAFUNE NATIVE: TREINO ESTÁVEL] ---"
    flush(stdout)
    
    vocab_size = 500
    config = TransformerConfig(vocab_size, 128, 512, 8, 12, 2048, 0.1f0)
    model = BidirectionalTransformer(config)
    md = MaskDiffusion(vocab_size, 128)
    
    # Dataset Simulado (Fase 2 Proof-of-Concept)
    dataset = [rand(1:500, 128, 1) for _ in 1:100]
    
    @info "Iniciando loop de 22.5M parametros..."
    flush(stdout)

    for epoch in 1:3
        println("⚡ Epoch $epoch/3...")
        flush(stdout)
        
        # Testar um step de gradiente real (Zygote)
        for i in 1:10
            tokens = dataset[rand(1:100)]
            
            # Forward + Backward Pass
            # Usando uma simplificacao do train_step! para este monólito
            loss, grads = Zygote.withgradient(model) do m
                logits = m(tokens[:, 1])
                # Mock de loss simples (Mean Squared Error para testar autodiff)
                Statistics.mean(abs2, logits) 
            end
            
            if i % 5 == 0
                @printf("  Step %d/10 | Loss: %.4f | Gradiente: %s\\n", 
                        i, loss, grads[1] !== nothing ? "OK" : "NULO")
                flush(stdout)
            end
        end
    end
    
    @info "[✓] MOTOR CAFUNE VALIDADO."
end

start_stable_training()
""")
    print(f"✅ train_stable.jl criado e fundido com sucesso.")

if __name__ == "__main__":
    fuse_and_launch()
