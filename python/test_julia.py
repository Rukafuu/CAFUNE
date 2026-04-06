import os
import juliacall

try:
    jl = juliacall.new_embedded_julia()
    julia_dir = r"c:\Users\conta\Documents\Lira\Lira\CAFUNE\julia"
    jl.eval(f'using Pkg; Pkg.activate(r"{julia_dir}")')
    jl.eval('using Flux, CUDA, Optimisers, BSON')
    inference_script = os.path.join(julia_dir, "inference.jl").replace("\\", "/")
    jl.eval(f'include("{inference_script}")')
    print("Success")
except Exception as e:
    print(f"Exception: {e}")
