#!/usr/bin/env julia

"""Avalia um checkpoint em um split congelado sem iniciar treino."""

using BSON, JSON, TOML, SHA

const SCRIPT_DIR = @__DIR__
const ROOT = normpath(joinpath(SCRIPT_DIR, ".."))

include(joinpath(SCRIPT_DIR, "src", "CAFUNE.jl"))
using .CAFUNE

function parse_args(args)
    checkpoint = joinpath(SCRIPT_DIR, "checkpoints", "cafune_best.bson")
    split = "validation"
    max_samples = typemax(Int)
    i = 1
    while i <= length(args)
        i < length(args) || error("Uso: evaluate.jl [--checkpoint PATH] [--split validation|test] [--max-samples N]")
        flag, value = args[i], args[i + 1]
        if flag == "--checkpoint"
            checkpoint = value
        elseif flag == "--split"
            split = value
        elseif flag == "--max-samples"
            max_samples = parse(Int, value)
        else
            error("Argumento desconhecido: $flag")
        end
        i += 2
    end
    split in ("validation", "test") || error("--split deve ser validation ou test")
    max_samples > 0 || error("--max-samples deve ser positivo")
    return checkpoint, split, max_samples
end

function main()
    checkpoint, split, max_samples = parse_args(ARGS)
    checkpoint = abspath(checkpoint)
    isfile(checkpoint) || error("Checkpoint não encontrado: $checkpoint")

    config = TOML.parsefile(joinpath(ROOT, "config", "research.toml"))
    tokenizer = config["tokenizer"]
    dataset_path = joinpath(ROOT, tokenizer["dataset"])
    raw_tokens = JSON.parsefile(dataset_path)
    manifest = JSON.parsefile(joinpath(ROOT, config["data"]["splits"]))
    actual_sha = bytes2hex(sha256(read(dataset_path)))
    actual_sha == manifest["dataset_sha256"] || error("Dataset mudou desde a geração dos splits; execute prepare_splits.py.")

    # SentencePiece é 0-based; embeddings Julia são 1-based.
    dataset = [Int.(sequence) .+ 1 for sequence in raw_tokens]
    indices = Int.(manifest["splits"][split]) .+ 1
    split_dataset = dataset[indices]
    spm = JSON.parsefile(joinpath(ROOT, tokenizer["config"]))
    md = MaskDiffusion(Int(spm["vocab_size"]) - 1; mask_token_id=Int(spm["mask_id"]) + 1, num_steps=20)

    local model, meta
    BSON.@load checkpoint model meta
    evaluation = evaluate_masked_split(model, md, split_dataset; max_samples=max_samples)
    output = joinpath(SCRIPT_DIR, "evaluations", "$(split)_$(basename(checkpoint, ".bson")).json")
    save_evaluation(output, evaluation; metadata=Dict(
        "split" => split,
        "checkpoint" => checkpoint,
        "checkpoint_meta" => meta,
        "dataset_sha256" => actual_sha,
    ))
    aggregate = evaluation["aggregate"]
    println("Evaluation: $output")
    println("$(split) loss: $(round(aggregate["loss_mean"], digits=4)) | masked accuracy: $(round(aggregate["masked_token_accuracy"] * 100, digits=2))%")
end

main()
