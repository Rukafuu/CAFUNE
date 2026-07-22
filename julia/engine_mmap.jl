using Pkg
Pkg.activate(@__DIR__)

using Mmap

include(joinpath(@__DIR__, "inference.jl"))

const MEM_FILE = joinpath(PROJECT_ROOT, "cafune_brain.mem")
const MEM_SIZE = 2048
const PROMPT_RANGE = 601:1000
const RESPONSE_RANGE = 201:600


function read_buffer(mm, range)
    bytes = Vector{UInt8}(mm[range])
    terminator = findfirst(==(0x00), bytes)
    terminator !== nothing && resize!(bytes, terminator - 1)
    return String(bytes)
end


function write_buffer!(mm, range, text::AbstractString)
    mm[range] .= 0x00
    bytes = Vector{UInt8}(codeunits(text))
    count = min(length(bytes), length(range) - 1)
    count > 0 && (mm[first(range):first(range)+count-1] .= bytes[1:count])
end


function run_native_cafune()
    isfile(MEM_FILE) || error("mmap não encontrado: $MEM_FILE. Execute python python/cafune.py init-runtime")
    filesize(MEM_FILE) == MEM_SIZE || error("mmap deve ter $MEM_SIZE bytes")
    stream = open(MEM_FILE, "r+")
    mm = mmap(stream, Vector{UInt8}, (MEM_SIZE,))
    mm[1] = 0x00
    @info "CAFUNE local aguardando prompts" checkpoint=MODEL_PATH

    try
        while true
            if mm[1] == 0x01
                mm[1] = 0x02
                prompt = read_buffer(mm, PROMPT_RANGE)
                try
                    response, _, _ = generate_local_response(prompt)
                    write_buffer!(mm, RESPONSE_RANGE, response)
                catch error
                    @error "Falha na geração local" exception=(error, catch_backtrace())
                    write_buffer!(mm, RESPONSE_RANGE, "[CAFUNE Error] $(sprint(showerror, error))")
                finally
                    mm[1] = 0x00
                end
            end
            sleep(0.1)
        end
    finally
        close(stream)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    run_native_cafune()
end
