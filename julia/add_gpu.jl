
using Pkg
println("Instalando CUDA.jl para o CAFUNE...")
Pkg.add("CUDA")
using CUDA
println("GPU Detectada: ", CUDA.devices())
println("CAFUNE ESTA PRONTO PARA O SILICIO!")
