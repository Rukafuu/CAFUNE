# Dockerfile — CAFUNE Neural Engine
#
# Requisitos do host:
#   - Docker >= 24
#   - NVIDIA Container Toolkit (para GPU): https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
#
# Build:
#   docker build -t cafune:latest .
#
# Run (CPU):
#   docker run --rm -it cafune:latest
#
# Run (GPU):
#   docker run --rm -it --gpus all cafune:latest

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV JULIA_VERSION=1.10.2
ENV JULIA_PATH=/opt/julia

# ── Sistema base ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    wget \
    ca-certificates \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# ── Julia ─────────────────────────────────────────────────────
RUN wget -q https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-${JULIA_VERSION}-linux-x86_64.tar.gz \
    && tar -xzf julia-${JULIA_VERSION}-linux-x86_64.tar.gz -C /opt \
    && mv /opt/julia-${JULIA_VERSION} ${JULIA_PATH} \
    && rm julia-${JULIA_VERSION}-linux-x86_64.tar.gz

ENV PATH="${JULIA_PATH}/bin:${PATH}"

# ── Python deps ───────────────────────────────────────────────
WORKDIR /app
COPY python/requirements.txt ./python/requirements.txt
RUN pip install --no-cache-dir -r python/requirements.txt

# ── Julia deps ────────────────────────────────────────────────
COPY julia/Project.toml ./julia/Project.toml
RUN julia --project=./julia -e 'using Pkg; Pkg.instantiate()'

# ── CUDA kernel (opcional — só compila se nvcc disponível) ────
COPY c/ ./c/
RUN cd c && \
    mkdir -p lib && \
    nvcc -O2 -arch=sm_61 --compiler-options "-fPIC" -shared \
         -o lib/cafune_cuda.so src/attention.cu 2>/dev/null || \
    echo "[INFO] nvcc não compilou o kernel CUDA — usando fallback CPU."

# ── Código da aplicação ───────────────────────────────────────
COPY python/ ./python/
COPY julia/   ./julia/
COPY vocab.json ./

# Criar arquivo mmap inicial
RUN python3 -c "open('python/cafune_brain.mem', 'wb').write(b'\\x00' * 1024)"

ENV PYTHONUNBUFFERED=1
ENV JULIA_NUM_THREADS=auto

EXPOSE 5000

CMD ["python3", "python/dashboard.py"]
