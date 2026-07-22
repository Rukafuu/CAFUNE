#!/usr/bin/env julia

include(joinpath(@__DIR__, "..", "src", "CAFUNE.jl"))
using .CAFUNE
using CUDA
using Functors

println("=== CAFUNE CUDA smoke test ===")
functional = CUDA.functional()
println("CUDA funcional: $functional")

config = TinyConfig(64)
model = BidirectionalTransformer(config)
tokens = rand(1:64, 16)

if functional
    println("GPU: $(CUDA.name(CUDA.device()))")
    gpu_model = Functors.fmap(CUDA.cu, model)
    logits = gpu_model(CUDA.cu(tokens))
    @assert size(logits) == (64, 16)
else
    logits = model(tokens)
    @assert size(logits) == (64, 16)
end

println("CAFUNE CUDA smoke test: OK")

