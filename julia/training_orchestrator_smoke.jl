#!/usr/bin/env julia

include(joinpath(@__DIR__, "main_training.jl"))

value_head = ValueHead(d_hidden=4)
tokens = fill(5, 8)
@assert isfinite(predict_value(value_head, tokens, 5, 64, 0f0))

optimizer_state = Optimisers.setup(Optimisers.Adam(1f-3), value_head)
loss, optimizer_state, value_head = train_value_head!(
    value_head,
    optimizer_state,
    tokens,
    5,
    64,
    0f0,
    0.5f0,
)
@assert isfinite(loss)

println("CAFUNE training orchestrator smoke: OK")
