# fix_project.jl
using Pkg

# Caminho do projeto CAFUNE/julia
project_path = joinpath(@__DIR__, ".")
println("--- Reconstruindo o Ambiente do CAFUNE em $project_path ---")

Pkg.activate(project_path)

# Removendo o antigo e adicionando com os UUIDs corretos do seu sistema
pkgs = ["Zygote", "Functors", "Optimisers", "Statistics", "NNlib", "Flux", "Random", "LinearAlgebra", "Printf", "BSON"]

println("Resolvendo pacotes: $pkgs")
Pkg.add(pkgs)

println("\n--- Status Final do DNA do CAFUNE ---")
Pkg.status()

println("\n✅ Projeto reconstruído com sucesso!")
