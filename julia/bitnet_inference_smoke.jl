#!/usr/bin/env julia

checkpoint = normpath(joinpath(@__DIR__, "checkpoints", "sanity", "bitnet", "cafune_best.bson"))
isfile(checkpoint) || error("Execute primeiro: julia --project=julia julia/main_training.jl --sanity --bitnet")
ENV["CAFUNE_MODEL_PATH"] = checkpoint

include(joinpath(@__DIR__, "inference.jl"))
response, _, _ = generate_local_response(
    "O que e cafune?";
    max_new_tokens=8,
    num_steps=2,
    seed=42,
)

println("CAFUNE-BitNet inference smoke: OK")
println(response)
