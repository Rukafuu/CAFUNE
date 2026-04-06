
using Printf, Flux, Zygote, Optimisers, Statistics

# Versao minimalista para depuracao do monólito
struct Config; vs::Int; sl::Int; dm::Int; end
mutable struct Model; emb::Matrix{Float32}; head::Matrix{Float32}; cfg::Config; end
Flux.@functor Model (emb, head)

function (m::Model)(tks)
    x = m.emb[:, vec(tks)] # (dm, sl*batch)
    logits = m.head * x
    return reshape(logits, m.cfg.vs, m.cfg.sl, size(tks, 2))
end

function debug()
    println("--- DEBUG: IGNICAO CAFUNE ---")
    cfg = Config(500, 128, 512)
    model = Model(randn(Float32, 512, 500), randn(Float32, 500, 512), cfg)
    opt_state = Optimisers.setup(Optimisers.Adam(1f-3), model)
    tks = rand(1:500, 128, 8)
    
    println("Testando Step...")
    l, g = Zygote.withgradient(model) do m
        logits = m(tks)
        Flux.logitcrossentropy(reshape(logits, 500, :), Flux.onehotbatch(vec(tks), 1:500))
    end
    
    opt_state, model = Optimisers.update!(opt_state, model, g[1])
    @printf("Step de Teste Completo! Loss: %.4f\n", l)
    println("--- DEBUG: CAFUNE ESTA VIVO! ---")
end

debug()
