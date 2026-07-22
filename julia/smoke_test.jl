#!/usr/bin/env julia

include(joinpath(@__DIR__, "src", "CAFUNE.jl"))
using .CAFUNE

config = TinyConfig(64)
model = BidirectionalTransformer(config)

@assert count_params(model) > 0
@assert size(model(rand(1:64, 8))) == (64, 8)

# Variante BitNet: forward ternario e STE precisam permanecer treinaveis.
bit_model = BidirectionalTransformer(config; linear_mode=:bitnet)
@assert bit_model.blocks[1].attn.Wq isa BitLinear
@assert size(bit_model(rand(1:64, 8))) == (64, 8)
bit_layer = BitLinear(4, 4)
quantized = ternary_weight(bit_layer.weight)
@assert length(unique(round.(quantized ./ (CAFUNE.Statistics.mean(abs, bit_layer.weight) + eps(Float32))))) <= 3
gradient = CAFUNE.Zygote.gradient(layer -> sum(layer(ones(Float32, 4, 2))), bit_layer)[1]
@assert gradient !== nothing
@assert any(!iszero, gradient.weight)
@assert size(build_hypercube_connectivity(3)) == (8, 8)

cell = LIFCell(4, 8; timesteps=2)
v, w, spikes = cell(zeros(Float32, 8, 1), zeros(Float32, 8, 1), ones(Float32, 4, 1), 1)
@assert size(v) == size(w) == size(spikes) == (8, 1)

# Valida a sintaxe dos lançadores sem carregar checkpoints ou iniciar daemons.
Meta.parseall(read(joinpath(@__DIR__, "main_training.jl"), String))
Meta.parseall(read(joinpath(@__DIR__, "inference.jl"), String))
Meta.parseall(read(joinpath(@__DIR__, "engine_mmap.jl"), String))

println("CAFUNE smoke test: OK")
