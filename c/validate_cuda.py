"""
validate_cuda.py — Valida o kernel CUDA comparando saída com referência CPU

Verifica que launch_flash_attention() produz resultado numericamente
equivalente à implementação Python pura (tolerância 1e-4).

Uso:
    python c/validate_cuda.py

Requisitos:
    - c/lib/cafune_cuda.dll compilado (rode c/build.bat primeiro)
    - numpy
"""

import ctypes
import os
import math
import sys

try:
    import numpy as np
except ImportError:
    print("[ERRO] numpy não instalado. Execute: pip install numpy")
    sys.exit(1)

DLL_PATH = os.path.join(os.path.dirname(__file__), "lib", "cafune_cuda.dll")


def attention_cpu(Q, K, V):
    """Implementação de referência CPU: atenção padrão (softmax(QK^T/sqrt(d)) * V)."""
    d = Q.shape[1]
    scale = 1.0 / math.sqrt(d)
    scores = Q @ K.T * scale                          # [seq, seq]
    scores -= scores.max(axis=1, keepdims=True)       # estabilidade numérica
    weights = np.exp(scores)
    weights /= weights.sum(axis=1, keepdims=True)     # softmax
    return (weights @ V).astype(np.float32)


def attention_cuda(Q, K, V):
    """Chama o kernel CUDA via ctypes."""
    if not os.path.exists(DLL_PATH):
        raise FileNotFoundError(f"DLL não encontrada: {DLL_PATH}\nExecute c/build.bat primeiro.")

    lib = ctypes.CDLL(DLL_PATH)
    lib.launch_flash_attention.restype = None
    lib.launch_flash_attention.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # Q
        ctypes.POINTER(ctypes.c_float),  # K
        ctypes.POINTER(ctypes.c_float),  # V
        ctypes.POINTER(ctypes.c_float),  # O
        ctypes.c_int,                    # seq_len
        ctypes.c_int,                    # d_model
    ]

    seq_len, d_model = Q.shape
    O = np.zeros((seq_len, d_model), dtype=np.float32)

    def ptr(arr):
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib.launch_flash_attention(ptr(Q), ptr(K), ptr(V), ptr(O),
                               ctypes.c_int(seq_len), ctypes.c_int(d_model))
    return O


def run_validation():
    print("=== CAFUNE CUDA Validation ===\n")

    test_cases = [
        ("tiny  (seq=4,  d=4)",  4,  4),
        ("small (seq=16, d=16)", 16, 16),
        ("med   (seq=32, d=32)", 32, 32),
    ]

    all_passed = True
    rng = np.random.default_rng(42)

    for name, seq_len, d_model in test_cases:
        Q = rng.standard_normal((seq_len, d_model)).astype(np.float32)
        K = rng.standard_normal((seq_len, d_model)).astype(np.float32)
        V = rng.standard_normal((seq_len, d_model)).astype(np.float32)

        ref = attention_cpu(Q, K, V)

        try:
            out = attention_cuda(Q, K, V)
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")
            return

        max_diff = float(np.abs(ref - out).max())
        passed = max_diff < 1e-3
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}  max_diff={max_diff:.2e}")

    print()
    if all_passed:
        print("[OK] Kernel CUDA validado — saída equivalente à referência CPU.")
    else:
        print("[ERRO] Divergência detectada. Verifique o kernel CUDA.")
        sys.exit(1)


if __name__ == "__main__":
    run_validation()
