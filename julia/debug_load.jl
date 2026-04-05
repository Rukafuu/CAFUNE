# debug_load.jl
try
    include("src/CAFUNE.jl")
    println("SUCESSO")
catch e
    println("ERRO_DETALHADO:")
    showerror(stdout, e, catch_backtrace())
    println()
end
