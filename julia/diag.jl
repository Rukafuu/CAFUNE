using Mmap, Serialization, Dates, Statistics, Flux

println("--- [INSPEÇÃO DE SAÚDE CAFUNE JULIA] ---")

try
    println("1. Testando Includes (src/)...")
    include("src/transformer.jl")
    include("src/diffusion.jl")
    include("src/training.jl")
    println("✅ Includes OK")

    println("2. Testando Mmap (../cafune_brain.mem)...")
    if isfile("../cafune_brain.mem")
        s = open("../cafune_brain.mem", "r+")
        mm = mmap(s, Vector{UInt8}, (1024,))
        println("✅ Mmap OK")
        close(s)
    else
        println("❌ Mmap não encontrado!")
    end

    println("3. Testando Inicialização do Modelo (22.5M params)...")
    config = TransformerConfig(500, 1024, 512, 8, 12, 2048, 0.1f0)
    model = BidirectionalTransformer(config)
    println("✅ Modelo OK")

    println("4. Testando Step de Gradiente (RLAIF)...")
    dataset = [rand(1:500, 128, 1)]
    loss = train_on_reward!(model, dataset[1], 1.0f0)
    println("✅ Gradiente OK (Loss=$loss)")

    println("--- [DIAGNÓSTICO CONCLUÍDO: O MOTOR ESTÁ SAUDÁVEL] ---")

catch e
    println("\n❌ ERRO FATAL DETECTADO:")
    println(e)
    # Exibir traceback completo
    for (f, l) in stacktrace(catch_backtrace())
        println("   - $f em $l")
    end
end
