@echo off
REM build.bat — Compila o kernel CUDA do CAFUNE para cafune_cuda.dll
REM
REM Requisitos:
REM   - NVIDIA CUDA Toolkit (nvcc no PATH)
REM   - GPU com Compute Capability >= 6.0
REM
REM Uso:
REM   cd c
REM   build.bat
REM
REM Saída:
REM   c\lib\cafune_cuda.dll  (carregado por julia/src/transformer.jl via ccall)

setlocal

set SRC=src\attention.cu
set OUT=lib\cafune_cuda.dll
set ARCH=sm_61

echo [CAFUNE CUDA Build] Compilando %SRC% ...

where nvcc >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRO] nvcc nao encontrado. Instale o CUDA Toolkit e adicione ao PATH.
    exit /b 1
)

if not exist lib mkdir lib

nvcc -O2 -arch=%ARCH% --compiler-options "/MD" -shared -o %OUT% %SRC%

if %errorlevel% neq 0 (
    echo [ERRO] Compilacao falhou. Verifique os logs acima.
    exit /b 1
)

echo [OK] DLL gerada: %OUT%
echo.
echo Para validar numericamente contra CPU, execute:
echo     python c\validate_cuda.py
endlocal
