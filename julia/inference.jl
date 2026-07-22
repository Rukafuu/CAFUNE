using Pkg
Pkg.activate(@__DIR__)

using BSON, JSON, Random

include(joinpath(@__DIR__, "src", "CAFUNE.jl"))
using .CAFUNE

# Compatibilidade com checkpoints históricos salvos por main_training.jl,
# quando os tipos ainda eram serializados no namespace Main.
const TransformerConfig = CAFUNE.TransformerConfig
const BitLinear = CAFUNE.BitLinear
const MultiHeadAttention = CAFUNE.MultiHeadAttention
const SpikingSynchronyAttention = CAFUNE.SpikingSynchronyAttention
const FFN = CAFUNE.FFN
const TransformerBlock = CAFUNE.TransformerBlock
const SpikingTransformerBlock = CAFUNE.SpikingTransformerBlock
const BidirectionalTransformer = CAFUNE.BidirectionalTransformer

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_MODEL_PATH = joinpath(@__DIR__, "checkpoints", "cafune_best.bson")
const MODEL_PATH = get(ENV, "CAFUNE_MODEL_PATH", DEFAULT_MODEL_PATH)
const TOKENIZER_CLI = joinpath(PROJECT_ROOT, "python", "tokenizer_cli.py")
const SPM_CONFIG_PATH = joinpath(PROJECT_ROOT, "python", "spm_config.json")


function load_cafune_model(path::AbstractString=MODEL_PATH)
    isfile(path) || error("Checkpoint não encontrado: $path. Treine o modelo ou defina CAFUNE_MODEL_PATH.")
    payload = BSON.load(path)
    model = get(payload, :model, get(payload, :m_cpu, nothing))
    model === nothing && error("Checkpoint sem chave 'model' ou 'm_cpu': $path")
    expected_vocab = Int(JSON.parsefile(SPM_CONFIG_PATH)["vocab_size"])
    model.config.vocab_size == expected_vocab || error(
        "Checkpoint incompatível: vocab=$(model.config.vocab_size), esperado=$expected_vocab"
    )
    return model
end


function tokenizer_call(operation::String, input::String)
    command = `python $TOKENIZER_CLI $operation`
    return read(pipeline(command, stdin=IOBuffer(input)), String)
end


encode_prompt(prompt::AbstractString) = Int.(JSON.parse(tokenizer_call("encode", String(prompt)))) .+ 1

function decode_tokens(token_ids::Vector{Int})
    zero_based = token_ids .- 1
    return strip(tokenizer_call("decode", JSON.json(zero_based)))
end


function generate_local_response(prompt::AbstractString;
                                 model_path::AbstractString=MODEL_PATH,
                                 max_new_tokens::Int=48,
                                 num_steps::Int=20,
                                 temperature::Float32=0.7f0,
                                 seed::Int=42)
    isempty(strip(prompt)) && error("Prompt vazio")
    model = load_cafune_model(model_path)
    spm = JSON.parsefile(SPM_CONFIG_PATH)
    mask_id = Int(spm["mask_id"]) + 1
    bos_id = Int(spm["bos_id"]) + 1
    vocab_size = Int(spm["vocab_size"])
    seq_len = model.config.seq_len

    prompt_ids = vcat(bos_id, encode_prompt(strip(prompt) * " "))
    max_prompt = max(1, seq_len - max_new_tokens)
    prompt_ids = prompt_ids[1:min(length(prompt_ids), max_prompt)]
    total_len = min(seq_len, length(prompt_ids) + max_new_tokens)
    md = MaskDiffusion(vocab_size - 1; mask_token_id=mask_id, num_steps=num_steps)

    Random.seed!(seed)
    generated = generate_with_prompt(
        model,
        md,
        prompt_ids,
        total_len;
        num_steps=num_steps,
        temperature=temperature,
    )
    response_ids = generated[length(prompt_ids)+1:end]
    return decode_tokens(response_ids), response_ids, prompt_ids
end


if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("Uso: julia --project=julia julia/inference.jl \"seu prompt\"")
    response, _, _ = generate_local_response(join(ARGS, " "))
    println(response)
end
