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
const TRAIN_LOG   = joinpath(SCRIPT_DIR, "training_log.jsonl")

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

"""Pad ou trunca para seq_len.
pad_id=1 (UNK) porque Julia usa índices 1-based — índice 0 é inválido no embedding.
"""
function pad_or_truncate(ids::Vector{Int}, seq_len::Int; pad_id::Int=1)
    length(ids) >= seq_len && return ids[1:seq_len]
    return vcat(ids, fill(pad_id, seq_len - length(ids)))
end

# ── Carregamento do dataset ───────────────────────────────────────

"""
Carrega social_data.json + bercario_data.jsonl, remove duplicatas e tokeniza.
Aceita schemas: {user, response} e {prompt, target}.
Retorna vetor de matrizes (seq_len × 1) prontas para train!().
"""
function load_dataset(corpus_file::String, char2id::Dict{String,Int}, seq_len::Int)
    seen    = Set{String}()
    dataset = Vector{Matrix{Int}}()
    total_raw = 0

    function add_entry!(text::String)
        isempty(text) && return
        text in seen  && return
        push!(seen, text)
        ids    = tokenize_text(text, char2id)
        padded = pad_or_truncate(ids, seq_len)
        push!(dataset, reshape(padded, seq_len, 1))
    end

    # ── social_data.json ({user, response}) ──────────────────────
    if isfile(corpus_file)
        raw = JSON.parsefile(corpus_file)
        total_raw += length(raw)
        for entry in raw
            user     = strip(get(entry, "user",     ""))
            response = strip(get(entry, "response", ""))
            add_entry!(isempty(response) ? user : user * " " * response)
        end
    end

    # ── bercario_data.jsonl ({prompt, target} ou {prompt, response}) ──
    bercario = normpath(joinpath(dirname(corpus_file), "bercario_data.jsonl"))
    if isfile(bercario)
        open(bercario, "r") do f
            for line in eachline(f)
                line = strip(line)
                isempty(line) && continue
                try
                    entry    = JSON.parse(line)
                    prompt   = strip(get(entry, "prompt",   ""))
                    target   = strip(get(entry, "target",   get(entry, "response", "")))
                    total_raw += 1
                    add_entry!(isempty(target) ? prompt : prompt * " " * target)
                catch
                end
            end
        end
    end

    duplicates = total_raw - length(dataset)
    @info "Dataset: $(length(dataset)) sequencias unicas de $total_raw entradas ($duplicates duplicatas)"
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
    EPOCHS       = 100
    MAX_LR       = 8e-6   # pico do cosine — sobe gradualmente via warmup
    WARMUP_RATIO = 0.05  # 5% warmup linear + cosine decay até MAX_LR/10

    # ── 1. Vocabulário ──────────────────────────────────────────
    @info "1. Carregando vocabulário..."
    if !isfile(VOCAB_FILE)
        @error "vocab.json não encontrado: $VOCAB_FILE"
        return
    end
    char2id, vocab_size = load_vocab(VOCAB_FILE)
    @info "   $vocab_size tokens carregados de $(basename(VOCAB_FILE))"

    # Filtra tokens válidos para geração: apenas tokens curtos (≤4 chars)
    # O vocab BPE tem tokens corrompidos com frases inteiras (IDs 90-460)
    id2char     = Dict(v => k for (k, v) in char2id)
    valid_ids   = [id for (id, tok) in id2char if length(tok) <= 4]
    sort!(valid_ids)
    @info "   $(length(valid_ids)) tokens válidos para geração (comprimento ≤ 4 chars)"

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
    # Layout de offsets (0-based Python = 1-based Julia + 1):
    #   0      CmdID (uint8)
    #   20-27  Timestamp de geração (float64, Unix time)
    #   40-43  Gemini MNS score    (float32)
    #   44-47  MNS local score     (float32)
    #   48-51  Raegis penalty      (float32)
    #   52-55  Guardian penalty    (float32) ← anomalia comportamental
    #   60     Ethics flag         (uint8)
    @info "4. Conectando barramento mmap..."
    mm = nothing; s = nothing
    if isfile(MEM_FILE)
        s  = open(MEM_FILE, "r+")
        mm = mmap(s, Vector{UInt8}, (2048,))
        @info "   Barramento RLAIF ativo."
    else
        @warn "   cafune_brain.mem não encontrado — RLAIF desativado."
    end

    # Optimizer separado para RLAIF (LR menor para não sobrescrever treino supervisionado)
    RLAIF_LR     = 1e-5
    opt_rl       = Optimisers.Adam(Float32(RLAIF_LR))
    opt_state_rl = Optimisers.setup(opt_rl, model)

    # ── 5. Loop de treino com checkpointing e RLAIF ────────────
    best_loss = Inf32
    @info "5. Iniciando treino | epochs $(start_epoch+1)→$(start_epoch+EPOCHS)"

    for epoch in 1:EPOCHS
        actual_epoch = start_epoch + epoch
        println()
        @info "── Epoch $actual_epoch/$(start_epoch + EPOCHS) ──────────────────────"
        flush(stdout)

        # ── Treino supervisionado ──
        model = train!(model, md, dataset;
                       epochs        = 1,
                       max_lr        = MAX_LR,
                       warmup_ratio  = WARMUP_RATIO)

        # ── Estimar loss ──
        sample_n     = min(20, length(dataset))
        sample_idx   = randperm(length(dataset))[1:sample_n]
        epoch_losses = Float32[compute_loss(model, md, dataset[i]) for i in sample_idx]
        avg_loss     = mean(epoch_losses)
        @printf("   Loss estimada (n=%d): %.4f\n", sample_n, avg_loss)

        # ── Escrita do timestamp e loss no mmap ──
        if mm !== nothing
            ts_bytes   = reinterpret(UInt8, [Float64(Dates.datetime2unix(now()))])
            mm[21:28] .= ts_bytes   # offset 20 (0-based) = índice 21 (1-based)
            loss_bytes = reinterpret(UInt8, [Float64(avg_loss)])
            mm[33:40] .= loss_bytes  # offset 32 (0-based) = índice 33 (1-based)
        end

        # ── Geração iterativa e escrita no mmap (buffer 200-600) ──
        # (deve ocorrer ANTES do handshake para o teacher ter texto para avaliar)
        if mm !== nothing
            try
                id2char = Dict(v => k for (k, v) in char2id)
                gen_ids = generate(model, md, SEQ_LEN; num_steps=20, temperature=0.5f0, valid_ids=valid_ids)
                decoded = join([get(id2char, id, "") for id in gen_ids if id != md.mask_token_id && id > 4])
                decoded_bytes = Vector{UInt8}(codeunits(decoded)[1:min(end, 399)])
                mm[201:200+length(decoded_bytes)] .= decoded_bytes
                mm[201+length(decoded_bytes):600] .= 0x00

                # Escreve contexto de prompt no slot 600-1000 (avaliado pelo teacher)
                prompt_ctx = "CAFUNE gerou no epoch $actual_epoch:"
                prompt_bytes = Vector{UInt8}(codeunits(prompt_ctx)[1:min(end, 399)])
                mm[601:600+length(prompt_bytes)] .= prompt_bytes
                mm[601+length(prompt_bytes):1000] .= 0x00

                # ── Handshake com BitNet Teacher ──────────────────────────────
                # 0x03 = "novo output pronto, aguarda avaliação"
                mm[1] = 0x03
                @info "   [RLAIF] Handshake 0x03 → aguardando BitNet Teacher..."

                wait_start = time()
                while mm[1] != 0x04 && (time() - wait_start) < 30.0
                    sleep(0.5)
                end

                if mm[1] == 0x04
                    @info "   [RLAIF] Handshake 0x04 ✓ — teacher respondeu"
                    mm[1] = 0x00  # reseta para o próximo epoch
                else
                    @warn "   [RLAIF] Teacher não respondeu em 30s — continuando sem score este epoch"
                end
            catch e
                @warn "   [RLAIF] Erro na geração/handshake: $e"
            end
        end

        # ── Leitura e combinação dos sinais RLAIF ──
        if mm !== nothing
            mns_score        = reinterpret(Float32, mm[41:44])[1]   # offset 40 — score do teacher
            mns_local        = reinterpret(Float32, mm[45:48])[1]   # offset 44 — score secundário
            raegis_penalty   = reinterpret(Float32, mm[49:52])[1]   # offset 48
            guardian_penalty = reinterpret(Float32, mm[53:56])[1]   # offset 52
            ethics_flag      = mm[61]                                # offset 60

            # Validar ranges (protege contra lixo na memória)
            mns_score        = isnan(mns_score)        ? 0.0f0 : clamp(mns_score,        0.0f0, 1.0f0)
            mns_local        = isnan(mns_local)        ? 0.0f0 : clamp(mns_local,        0.0f0, 1.0f0)
            raegis_penalty   = isnan(raegis_penalty)   ? 0.0f0 : clamp(raegis_penalty,   0.0f0, 1.0f0)
            guardian_penalty = isnan(guardian_penalty) ? 0.0f0 : clamp(guardian_penalty, 0.0f0, 0.5f0)

            # Combina teacher score (70%) com MNS local (30%)
            α = mns_score > 0.0f0 ? 0.7f0 : 0.0f0
            combined = α * mns_score + (1.0f0 - α) * mns_local

            # Ethics flag dobra a penalidade Raegis (sicofância detectada)
            effective_raegis  = ethics_flag == 0x01 ? raegis_penalty * 2.0f0 : raegis_penalty
            total_penalty     = effective_raegis + guardian_penalty
            combined_reward   = max(0.0f0, combined - total_penalty)

            if combined_reward > 0.0f0
                @info "   [RLAIF] Teacher=$(round(mns_score,digits=3)) MNS=$(round(mns_local,digits=3)) Raegis=$(round(effective_raegis,digits=3)) Guardian=$(round(guardian_penalty,digits=3)) → Reward=$(round(combined_reward,digits=3))"

                rl_idx  = rand(1:length(dataset))
                rl_loss, opt_state_rl, model = train_on_reward!(
                    model, md, opt_state_rl, dataset[rl_idx], combined_reward
                )
                @printf("   [RLAIF] Loss RLAIF: %.4f\n", rl_loss)
            else
                @info "   [RLAIF] Reward zerado ou nulo — sem atualização RLAIF neste epoch."
            end
        end

        # ── Log de progresso (lido pelo dashboard) ──
        open(TRAIN_LOG, "a") do f
            entry = JSON.json(Dict(
                "epoch"      => actual_epoch,
                "total"      => start_epoch + EPOCHS,
                "loss"       => round(Float64(avg_loss), digits=4),
                "best_loss"  => round(Float64(min(avg_loss, best_loss)), digits=4),
                "timestamp"  => Dates.format(now(), "HH:MM:SS"),
                "dataset_n"  => length(dataset),
            ))
            println(f, entry)
        end

        # ── Checkpoints ──
        save_checkpoint(model, actual_epoch, Float32(avg_loss), config, vocab_size)
        if avg_loss < best_loss
            best_loss = avg_loss
            save_best(model, actual_epoch, Float32(avg_loss), config, vocab_size)
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
