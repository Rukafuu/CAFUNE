@echo off
echo [CAFUNE Forge] Iniciando compilacao do Kernel CUDA...
echo 🔨 Alvo: c/src/attention.cu -> c/lib/cafune_cuda.dll

if not exist "c\lib" mkdir "c\lib"

nvcc -shared -o c/lib/cafune_cuda.dll c/src/attention.cu --compiler-options /LD

if %errorlevel% neq 0 (
    echo ❌ Erro na compilacao CUDA! Verifique o toolkit.
    exit /b %errorlevel%
)

echo ✅ cafune_cuda.dll forjada com sucesso no silicio!
