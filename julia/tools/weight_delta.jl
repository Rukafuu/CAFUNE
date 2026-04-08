"""
weight_delta.jl — Comparação de pesos entre dois checkpoints BSON do CAFUNE

Uso:
    julia julia/tools/weight_delta.jl <ckpt_antes.bson> <ckpt_depois.bson>

Calcula o delta L2 entre parâmetros treináveis de dois checkpoints BSON
gerados pelo main_training.jl e emite um veredito de estabilidade de treino.

Verditos (calibrados para ~45M params — modelo CAFUNE):
    Surgical    → Δ < 0.1   (atualização pontual, sem risco de regressão)
    Moderate    → Δ < 1.0   (passo normal de gradient descent)
    Aggressive  → Δ < 5.0   (LR alto ou reward forte — monitorar loss)
    Catastrophic→ Δ ≥ 5.0   (possível divergência — revisar hiperparâmetros)
"""

using BSON: @load
using Printf
using Statistics

# ── Extração plana de parâmetros treináveis ───────────────────────────────────

"""
Extrai recursivamente todos os arrays Float32 de uma struct (modelo BSON).
Retorna um vetor de vetores achatados.
"""
function extract_leaves(x)::Vector{Vector{Float32}}
    out = Vector{Vector{Float32}}()
    _walk!(out, x)
    return out
end

function _walk!(out::Vector{Vector{Float32}}, x)
    if x isa Array{Float32}
        push!(out, vec(x))
    elseif x isa Array{Float64}
        push!(out, Float32.(vec(x)))
    elseif x isa NamedTuple || x isa Tuple
        for f in x
            _walk!(out, f)
        end
    elseif x isa AbstractDict
        for v in values(x)
            _walk!(out, v)
        end
    elseif isstructtype(typeof(x))
        for fname in fieldnames(typeof(x))
            _walk!(out, getfield(x, fname))
        end
    end
end

# ── Carga de checkpoint ───────────────────────────────────────────────────────

"""
Carrega um checkpoint BSON e retorna (model, meta).
Lança erro com mensagem amigável se o arquivo for inválido.
"""
function load_ckpt(path::String)
    isfile(path) || error("Checkpoint não encontrado: $path")
    local model, meta
    try
        @load path model meta
    catch e
        error("Falha ao carregar $path: $e")
    end
    return model, meta
end

# ── Cálculo do delta ──────────────────────────────────────────────────────────

"""
Calcula o delta L2 normalizado entre os parâmetros de dois modelos.

    δ = √(Σ ||w_b - w_a||²) / √(Σ ||w_a||²)

A normalização pelo módulo do ponto de partida permite comparar
modelos de tamanhos diferentes e torna o veredito independente de escala.
"""
function compute_delta(model_before, model_after)
    leaves_a = extract_leaves(model_before)
    leaves_b = extract_leaves(model_after)

    length(leaves_a) == length(leaves_b) ||
        error("Arquiteturas incompatíveis: $(length(leaves_a)) vs $(length(leaves_b)) grupos de parâmetros")

    sq_diff  = 0.0
    sq_norm  = 0.0
    n_params = 0

    for (a, b) in zip(leaves_a, leaves_b)
        length(a) == length(b) ||
            error("Grupo de parâmetros com tamanhos incompatíveis: $(length(a)) vs $(length(b))")
        sq_diff  += sum((Float64.(b) .- Float64.(a)) .^ 2)
        sq_norm  += sum(Float64.(a) .^ 2)
        n_params += length(a)
    end

    δ = sq_norm > 0.0 ? sqrt(sq_diff) / sqrt(sq_norm) : sqrt(sq_diff)
    return δ, n_params
end

# ── Veredito ──────────────────────────────────────────────────────────────────

function verdict(δ::Float64)
    if δ < 0.1
        return "✅ SURGICAL   ", "Atualização pontual — sem risco de regressão."
    elseif δ < 1.0
        return "🟡 MODERATE   ", "Passo normal de gradient descent."
    elseif δ < 5.0
        return "🟠 AGGRESSIVE ", "LR alto ou reward forte — monitore a loss."
    else
        return "🔴 CATASTROPHIC", "Possível divergência — revise hiperparâmetros / LR / reward scale."
    end
end

# ── Relatório por grupo ───────────────────────────────────────────────────────

"""
Imprime os top-N grupos de parâmetros com maior delta absoluto.
Útil para pinpoint qual camada está mudando mais após RLAIF.
"""
function report_top_groups(model_before, model_after; top_n::Int=5)
    leaves_a = extract_leaves(model_before)
    leaves_b = extract_leaves(model_after)
    deltas   = Float64[]
    sizes    = Int[]

    for (a, b) in zip(leaves_a, leaves_b)
        d = sqrt(sum((Float64.(b) .- Float64.(a)) .^ 2))
        push!(deltas, d)
        push!(sizes,  length(a))
    end

    order    = sortperm(deltas, rev=true)
    shown    = min(top_n, length(order))

    println()
    println("  Top-$shown grupos por delta absoluto:")
    println("  ┌─────┬────────────┬────────────┐")
    println("  │ Rank│ |Δw| abs   │  n_params  │")
    println("  ├─────┼────────────┼────────────┤")
    for i in 1:shown
        idx = order[i]
        @printf("  │ %3d │ %10.4f │ %10d │\n", i, deltas[idx], sizes[idx])
    end
    println("  └─────┴────────────┴────────────┘")
end

# ── Main ──────────────────────────────────────────────────────────────────────

function main(args)
    length(args) == 2 || begin
        println("Uso: julia weight_delta.jl <ckpt_antes.bson> <ckpt_depois.bson>")
        exit(1)
    end

    path_a, path_b = args[1], args[2]

    println()
    println("══════════════════════════════════════════════════════")
    println("  CAFUNE — Análise de Delta de Pesos")
    println("══════════════════════════════════════════════════════")
    @printf("  Antes : %s\n", basename(path_a))
    @printf("  Depois: %s\n", basename(path_b))
    println()

    println("Carregando checkpoints...")
    model_a, meta_a = load_ckpt(path_a)
    model_b, meta_b = load_ckpt(path_b)

    println("  Epoch antes : $(get(meta_a, \"epoch\", \"?\"))")
    println("  Loss  antes : $(round(get(meta_a, \"loss\", NaN), digits=4))")
    println("  Epoch depois: $(get(meta_b, \"epoch\", \"?\"))")
    println("  Loss  depois: $(round(get(meta_b, \"loss\", NaN), digits=4))")
    println()

    println("Calculando delta L2 normalizado...")
    δ, n_params = compute_delta(model_a, model_b)
    label, msg  = verdict(δ)

    println()
    @printf("  Δ (normalizado)  : %.6f\n", δ)
    @printf("  Parâmetros totais: %d (%.2fM)\n", n_params, n_params / 1e6)
    println()
    println("  Veredito: $label")
    println("  → $msg")

    report_top_groups(model_a, model_b, top_n=5)

    # Δ loss
    loss_a = get(meta_a, "loss", NaN)
    loss_b = get(meta_b, "loss", NaN)
    if !isnan(loss_a) && !isnan(loss_b)
        Δloss = loss_b - loss_a
        println()
        @printf("  ΔLoss: %+.4f (%s)\n", Δloss,
                Δloss < 0 ? "melhora" : Δloss > 0 ? "regressão" : "estável")
    end

    println()
    println("══════════════════════════════════════════════════════")
end

main(ARGS)
