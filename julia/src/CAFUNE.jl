module CAFUNE

using Random, LinearAlgebra, Statistics, Printf
using Zygote, Functors, Optimisers

include(joinpath(@__DIR__, "diffusion.jl"))
include(joinpath(@__DIR__, "transformer.jl"))
include(joinpath(@__DIR__, "training.jl"))
include(joinpath(@__DIR__, "sampling.jl"))

export
    MaskDiffusion, forward_mask, forward_mask_batch, sample_t,
    BidirectionalTransformer, TransformerConfig, count_params, TinyConfig, SmallConfig,
    train_step!, train!, compute_loss,
    generate, generate_with_prompt

end
