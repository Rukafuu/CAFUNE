
using Pkg
Pkg.activate(".")
println("--- [CAFUNE MAINTENANCE] ---")
packages = ["BSON", "Flux", "CUDA", "Optimisers", "Zygote", "Functors"]
for p in packages
    try
        Pkg.add(p)
        println("✅ Pacote $p instalado com honra de silício.")
    catch e
        println("⚠️ Falha ao instalar $p: $e")
    end
end
Pkg.instantiate()
println("🚀 [CAFUNE] Arena Julia Estabilizada.")
