"""
    evaluation.jl — Avaliação reproduzível para difusão mascarada.

As métricas são calculadas somente nas posições mascaradas. Cada combinação de
nível de ruído e seed recebe a mesma sequência de máscaras, tornando resultados
de checkpoints e ablações comparáveis.
"""

using JSON, Random, Statistics

const DEFAULT_EVAL_T_LEVELS = Float32[0.1, 0.25, 0.5, 0.75, 0.9]
const DEFAULT_EVAL_SEEDS = Int[20_260_721, 20_260_722, 20_260_723]

function _masked_metrics(model::BidirectionalTransformer, md::MaskDiffusion,
                         tokens::AbstractVector{Int}, t::Float32; top_k::Int=5)
    masked_tokens, mask = forward_mask(md, collect(tokens), t)
    masked_count = count(mask)
    masked_count == 0 && return (loss=0f0, accuracy=0f0, top_k_accuracy=0f0, masked_count=0)

    logits = model(masked_tokens)
    loss = cross_entropy_masked(logits, tokens, mask)
    correct = 0
    top_k_correct = 0
    k = min(top_k, size(logits, 1))
    for position in findall(mask)
        target = tokens[position]
        ranked = partialsortperm(view(logits, :, position), 1:k, rev=true)
        correct += ranked[1] == target
        top_k_correct += target in ranked
    end
    return (
        loss=Float32(loss),
        accuracy=Float32(correct / masked_count),
        top_k_accuracy=Float32(top_k_correct / masked_count),
        masked_count=masked_count,
    )
end

"""
    evaluate_masked_split(model, md, dataset; kwargs...) -> Dict

Avalia um split congelado em níveis de máscara e seeds fixos. `loss_mean` de
`aggregate` é a métrica usada para selecionar o melhor checkpoint: ela pondera
cada token mascarado igualmente em todos os níveis e seeds.
"""
function evaluate_masked_split(model::BidirectionalTransformer, md::MaskDiffusion,
                               dataset::AbstractVector;
                               t_levels::AbstractVector{<:Real}=DEFAULT_EVAL_T_LEVELS,
                               seeds::AbstractVector{<:Integer}=DEFAULT_EVAL_SEEDS,
                               max_samples::Int=length(dataset), top_k::Int=5)
    isempty(dataset) && error("Não é possível avaliar um split vazio.")
    sample_count = min(max_samples, length(dataset))
    sample_count > 0 || error("max_samples deve ser positivo.")
    rows = Any[]
    aggregate_loss = 0.0
    aggregate_correct = 0
    aggregate_top_k = 0
    aggregate_masked = 0

    for t_raw in t_levels
        t = Float32(t_raw)
        0f0 < t <= 1f0 || error("Nível de máscara deve estar em (0, 1].")
        seed_losses = Float64[]
        seed_accuracies = Float64[]
        seed_top_k_accuracies = Float64[]
        row_masked = 0
        for seed_raw in seeds
            Random.seed!(seed_raw)
            loss_sum = 0.0
            correct_sum = 0
            top_k_sum = 0
            masked_sum = 0
            for example in @view dataset[1:sample_count]
                tokens = vec(example)
                metrics = _masked_metrics(model, md, tokens, t; top_k=top_k)
                loss_sum += metrics.loss * metrics.masked_count
                correct_sum += round(Int, metrics.accuracy * metrics.masked_count)
                top_k_sum += round(Int, metrics.top_k_accuracy * metrics.masked_count)
                masked_sum += metrics.masked_count
            end
            masked_sum == 0 && error("Nenhum token foi mascarado durante a avaliação.")
            push!(seed_losses, loss_sum / masked_sum)
            push!(seed_accuracies, correct_sum / masked_sum)
            push!(seed_top_k_accuracies, top_k_sum / masked_sum)
            aggregate_loss += loss_sum
            aggregate_correct += correct_sum
            aggregate_top_k += top_k_sum
            aggregate_masked += masked_sum
            row_masked += masked_sum
        end
        push!(rows, Dict(
            "mask_ratio" => Float64(t),
            "loss_mean" => mean(seed_losses),
            "loss_std" => length(seed_losses) > 1 ? std(seed_losses) : 0.0,
            "accuracy_mean" => mean(seed_accuracies),
            "accuracy_std" => length(seed_accuracies) > 1 ? std(seed_accuracies) : 0.0,
            "top_$(top_k)_accuracy_mean" => mean(seed_top_k_accuracies),
            "top_$(top_k)_accuracy_std" => length(seed_top_k_accuracies) > 1 ? std(seed_top_k_accuracies) : 0.0,
            "masked_tokens_per_seed" => row_masked ÷ length(seeds),
        ))
    end

    return Dict(
        "schema_version" => 1,
        "sample_count" => sample_count,
        "seeds" => collect(seeds),
        "top_k" => top_k,
        "by_mask_ratio" => rows,
        "aggregate" => Dict(
            "loss_mean" => aggregate_loss / aggregate_masked,
            "masked_token_accuracy" => aggregate_correct / aggregate_masked,
            "top_$(top_k)_accuracy" => aggregate_top_k / aggregate_masked,
            "masked_tokens" => aggregate_masked,
        ),
    )
end

"""Escreve um artefato de avaliação JSON, pronto para comparação entre runs."""
function save_evaluation(path::AbstractString, evaluation::Dict; metadata::Dict=Dict())
    mkpath(dirname(path))
    payload = copy(evaluation)
    payload["metadata"] = metadata
    open(path, "w") do io
        JSON.print(io, payload, 2)
        println(io)
    end
    return path
end
