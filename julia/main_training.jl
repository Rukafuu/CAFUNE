# ============================================================
#  main_training.jl — Orquestrador de Treino CAFUNE
#
#  Bloco 1 (Base):
#    - Paths absolutos baseados em @__DIR__
#    - Carrega social_data.json real (300 pares, deduplicado)
#    - Tokenização character-level via vocab.json
#    - Checkpointing por epoch com metadados
#    - Resume automático do melhor checkpoint
# ============================================================

using Mmap, Dates, Statistics, Printf, JSON
using BSON: @save, @load

include("src/transformer.jl")
include("src/diffusion.jl")
include("src/training.jl")

# ── Paths absolutos ───────────────────────────────────────────────
const SCRIPT_DIR  = @__DIR__
const MEM_FILE    = normpath(joinpath(SCRIPT_DIR, "..", "cafune_brain.mem"))
const CORPUS_FILE = normpath(joinpath(SCRIPT_DIR, "..", "python", "social_data.json"))
const VOCAB_FILE  = normpath(joinpath(SCRIPT_DIR, "..", "vocab.json"))
const CKPT_DIR    = joinpath(SCRIPT_DIR, "checkpoints")
const BEST_CKPT   = joinpath(CKPT_DIR, "cafune_best.bson")

# ── Tokenizador character-level ───────────────────────────────────

"""Carrega vocab.json e retorna (char2id, vocab_size)."""
function load_vocab(path::String)
    data     = JSON.parsefile(path)
    char2id  = Dict{String,Int}(string(k) => Int(v) for (k, v) in data["char2id"])
    return char2id, Int(data["vocab_size"])
end

"""Converte texto → Vector{Int} com [BOS] e [EOS]."""
function tokenize_text(text::String, char2id::Dict{String,Int};
                       bos_id::Int=2, eos_id::Int=3, unk_id::Int=1)
    ids = [bos_id]
    for c in text
        push!(ids, get(char2id, string(c), unk_id))
    end
    push!(ids, eos_id)
    return ids
end

"""Pad ou trunca para seq_len."""
function pad_or_truncate(ids::Vector{Int}, seq_len::Int; pad_id::Int=0)
    length(ids) >= seq_len && return ids[1:seq_len]
    return vcat(ids, fill(pad_id, seq_len - length(ids)))
end

# ── Carregamento do dataset ───────────────────────────────────────

"""
Carrega social_data.json, remove duplicatas e tokeniza.
Retorna vetor de matrizes (seq_len × 1) prontas para train!().
"""
function load_dataset(corpus_file::String, char2id::Dict{String,Int}, seq_len::Int)
    raw     = JSON.parsefile(corpus_file)
    seen    = Set{String}()
    dataset = Vector{Matrix{Int}}()

    for entry in raw
        user     = strip(get(entry, "user",     ""))
        response = strip(get(entry, "response", ""))
        text     = isempty(response) ? user : user * " " * response
        isempty(text) && continue
        text in seen  && continue
        push!(seen, text)

        ids    = tokenize_text(text, char2id)
        padded = pad_or_truncate(ids, seq_len)
        push!(dataset, reshape(padded, seq_len, 1))
    end

    duplicates = length(raw) - length(dataset)
    @info "Dataset: $(length(dataset)) sequências únicas | $duplicates duplicatas removidas"
    return dataset
end

# ── Checkpoint ────────────────────────────────────────────────────

"""Salva checkpoint por epoch (julia/checkpoints/cafune_epochN_lossL.bson)."""
function save_checkpoint(model, epoch::Int, loss::Float32,
                         config::TransformerConfig, vocab_size::Int)
    mkpath(CKPT_DIR)
    loss_str = @sprintf("%.4f", loss)
    ts       = Dates.format(now(), "yyyymmdd_HHMMSS")
    path     = joinpath(CKPT_DIR, "cafune_epoch$(epoch)_loss$(loss_str)_$(ts).bson")

    meta = Dict(
        "epoch"      => epoch,
        "loss"       => Float64(loss),
        "vocab_size" => vocab_size,
        "d_model"    => config.d_model,
        "n_layers"   => config.n_layers,
        "n_heads"    => config.n_heads,
        "seq_len"    => config.seq_len,
        "timestamp"  => string(now()),
    )
    @save path model=model meta=meta
    @info "  Checkpoint salvo: $(basename(path))"
    return path
end

"""Atualiza o melhor checkpoint (cafune_best.bson)."""
function save_best(model, epoch::Int, loss::Float32,
                   config::TransformerConfig, vocab_size::Int)
    mkpath(CKPT_DIR)
    meta = Dict(
        "epoch"      => epoch,
        "loss"       => Float64(loss),
        "vocab_size" => vocab_size,
        "d_model"    => config.d_model,
        "n_layers"   => config.n_layers,
        "n_heads"    => config.n_heads,
        "seq_len"    => config.seq_len,
        "timestamp"  => string(now()),
    )
    @save BEST_CKPT model=model meta=meta
    @info "  ⭐ Melhor checkpoint atualizado (epoch=$epoch, loss=$loss)"
end

"""
Tenta carregar o melhor checkpoint existente.
Retorna (model, meta, start_epoch) ou (nothing, nothing, 0).
"""
function try_resume()
    !isfile(BEST_CKPT) && return nothing, nothing, 0

    @info "Checkpoint encontrado: $BEST_CKPT"
    local model, meta
    try
        @load BEST_CKPT model meta
        @info "  Resumindo do epoch $(meta["epoch"]) | loss $(round(meta["loss"], digits=4))"
        return model, meta, Int(meta["epoch"])
    catch e
        @warn "Falha ao carregar checkpoint: $e — iniciando do zero."
        return nothing, nothing, 0
    end
end

# ── Sessão principal ──────────────────────────────────────────────

function start_training_session()
    @info "══════════════════════════════════════════════"
    @info "  CAFUNE ENGINE — Treino Fase 2"
    @info "══════════════════════════════════════════════"

    # Hiperparâmetros
    SEQ_LEN      = 128
    D_MODEL      = 256
    N_HEADS      = 8
    N_LAYERS     = 6
    D_FF         = 1024
    EPOCHS       = 20
    MAX_LR       = 3e-4
    WARMUP_RATIO = 0.15

    # ── 1. Vocabulário ──────────────────────────────────────────
    @info "1. Carregando vocabulário..."
    if !isfile(VOCAB_FILE)
        @error "vocab.json não encontrado: $VOCAB_FILE"
        return
    end
    char2id, vocab_size = load_vocab(VOCAB_FILE)
    @info "   $vocab_size tokens carregados de $(basename(VOCAB_FILE))"

    # ── 2. Dataset ──────────────────────────────────────────────
    @info "2. Carregando dataset..."
    if !isfile(CORPUS_FILE)
        @error "social_data.json não encontrado: $CORPUS_FILE"
        return
    end
    dataset = load_dataset(CORPUS_FILE, char2id, SEQ_LEN)

    # ── 3. Modelo — resume ou inicializa ────────────────────────
    @info "3. Inicializando modelo..."
    existing_model, _, start_epoch = try_resume()

    # vocab_size + 1 para incluir o token [MASK] como ID extra
    config = TransformerConfig(vocab_size + 1, SEQ_LEN, D_MODEL, N_HEADS, N_LAYERS, D_FF, 0.0f0)
    md     = MaskDiffusion(vocab_size; num_steps=20)

    if existing_model !== nothing
        model = existing_model
        @info "   Modelo restaurado | $(round(count_params(model)/1e6, digits=2))M params"
    else
        model = BidirectionalTransformer(config)
        start_epoch = 0
        @info "   Modelo novo | $(round(count_params(model)/1e6, digits=2))M params"
    end

    # ── 4. Barramento mmap ──────────────────────────────────────
    @info "4. Conectando barramento mmap..."
    mm = nothing; s = nothing
    if isfile(MEM_FILE)
        s  = open(MEM_FILE, "r+")
        mm = mmap(s, Vector{UInt8}, (1024,))
        @info "   Barramento RLAIF ativo."
    else
        @warn "   cafune_brain.mem não encontrado — RLAIF desativado."
    end

    # ── 5. Loop de treino com checkpointing ────────────────────
    epochs_remaining = EPOCHS
    best_loss = Inf32
    @info "5. Iniciando treino | epochs $(start_epoch+1)→$(start_epoch+EPOCHS)"

    for epoch in 1:EPOCHS
        actual_epoch = start_epoch + epoch
        println()
        @info "── Epoch $actual_epoch/$( start_epoch + EPOCHS) ──────────────────────"
        flush(stdout)

        # Um epoch de treino (train! retorna o modelo atualizado)
        model = train!(model, md, dataset;
                       epochs        = 1,
                       max_lr        = MAX_LR,
                       warmup_ratio  = WARMUP_RATIO)

        # Estimar loss em amostra (máx 20 sequências para ser rápido)
        sample_n     = min(20, length(dataset))
        sample_idx   = randperm(length(dataset))[1:sample_n]
        epoch_losses = Float32[compute_loss(model, md, dataset[i]) for i in sample_idx]
        avg_loss     = mean(epoch_losses)

        @printf("   Loss estimada (n=%d): %.4f\n", sample_n, avg_loss)

        # Checkpoint por epoch
        save_checkpoint(model, actual_epoch, Float32(avg_loss), config, vocab_size)

        # Melhor checkpoint
        if avg_loss < best_loss
            best_loss = avg_loss
            save_best(model, actual_epoch, Float32(avg_loss), config, vocab_size)
        end

        # Leitura do sinal RLAIF (Bloco 2 vai usar isso de verdade)
        if mm !== nothing
            reward = reinterpret(Float32, mm[41:44])[1]
            if 0.0f0 < reward <= 1.0f0
                @info "   [RLAIF] Sinal do mentor: $(round(reward, digits=3)) (Bloco 2: aplicação ao gradiente)"
            end
        end
    end

    # ── 6. Finalização ──────────────────────────────────────────
    @info "══════════════════════════════════════════════"
    @info "  Treino concluído | Best loss: $(round(best_loss, digits=4))"
    @info "  Checkpoints em: $CKPT_DIR"
    @info "══════════════════════════════════════════════"

    mm !== nothing && close(s)
end

if abspath(PROGRAM_FILE) == @__FILE__
    start_training_session()
end
