# install_deps.jl
using Pkg
Pkg.activate(".")
println("--- Instalando Dependências do CAFUNE ---")
try
    Pkg.add(["Zygote", "Functors", "Optimisers", "Statistics", "NNlib", "Flux"])
    println("✅ Instalação concluída!")
    println("--- Pré-compilando ---")
    Pkg.precompile()
    println("✅ Tudo pronto para o teste!")
catch e
    println("❌ Erro durante a instalacao:")
    showerror(stdout, e)
end
