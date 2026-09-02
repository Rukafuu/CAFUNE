#!/usr/bin/env julia

include(joinpath(@__DIR__, "src", "CAFUNE.jl"))
using .CAFUNE

config = TinyConfig(64)
model = BidirectionalTransformer(config)

mha_model = BidirectionalTransformer(config; attention_mode=:mha)
@assert all(block isa CAFUNE.TransformerBlock for block in mha_model.blocks)
@assert count_params(mha_model) > count_params(model)

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

evaluation = evaluate_masked_split(model, MaskDiffusion(64; mask_token_id=64),
                                   [reshape(collect(1:8), 8, 1)];
                                   t_levels=Float32[1.0], seeds=[42])
@assert evaluation["sample_count"] == 1
@assert evaluation["aggregate"]["masked_tokens"] > 0
@assert haskey(evaluation["by_mask_ratio"][1], "top_5_accuracy_mean")

cell = LIFCell(4, 8; timesteps=2)
v, w, spikes = cell(zeros(Float32, 8, 1), zeros(Float32, 8, 1), ones(Float32, 4, 1), 1)
@assert size(v) == size(w) == size(spikes) == (8, 1)

# Valida a sintaxe dos lançadores sem carregar checkpoints ou iniciar daemons.
Meta.parseall(read(joinpath(@__DIR__, "main_training.jl"), String))
Meta.parseall(read(joinpath(@__DIR__, "inference.jl"), String))
Meta.parseall(read(joinpath(@__DIR__, "engine_mmap.jl"), String))

println("CAFUNE smoke test: OK")
