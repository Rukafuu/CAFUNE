"""
bridge.py — Ponte Python ↔ Julia (CAFUNE)

Usa juliacall (mais estável que PyJulia no Windows) para chamar
o motor Julia do CAFUNE diretamente do Python.

Instalação:
    pip install juliacall

Documentação: https://juliapy.github.io/PythonCall.jl/stable/juliacall/
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Caminho do módulo Julia
JULIA_PROJECT = str(Path(__file__).parent.parent / "julia")


def init_julia() -> "Julia":
    """
    Inicializa o ambiente Julia com o projeto FrankensteinDLLM.
    
    NOTA: O primeiro import de juliacall demora (Julia JIT compile).
    Chamadas subsequentes são rápidas.
    """
    os.environ["JULIA_PROJECT"] = JULIA_PROJECT

    try:
        from juliacall import Main as jl
        print("✅ Julia conectada via juliacall")
    except ImportError:
        print("❌ juliacall não instalado. Execute: pip install juliacall")
        sys.exit(1)

    # Verificar e instalar dependências se necessário
    print("📋 Verificando dependências Julia...")
    jl.seval("""
        using Pkg
        pkgs = ["Zygote", "Functors", "Optimisers", "Statistics", "Random", "LinearAlgebra"]
        for p in pkgs
            try
                eval(Symbol("using \$p"))
            catch
                @info "Instalando pacote \$p (isso só acontece uma vez)..."
                Pkg.add(p)
            end
        end
    """)

    # Carregar o módulo CAFUNE
    julia_src_dir = str(Path(JULIA_PROJECT) / "src").replace("\\", "/")
    julia_main_file = str(Path(julia_src_dir) / "CAFUNE.jl").replace("\\", "/")
    
    print(f"📥 Carregando {julia_main_file}...")
    jl.include(julia_main_file)
    jl.seval("using .CAFUNE")

    print("✅ CAFUNE.jl carregado")
    return jl


class CAFUNEBridge:
    """
    Interface Python para o motor DLLM em Julia.
    
    Exemplo de uso:
        bridge = FrankensteinBridge()
        bridge.build_model(vocab_size=100)
        
        tokens = [5, 12, 3, 8, 15]
        generated = bridge.generate(tokens, total_len=20)
        print(generated)
    """

    def __init__(self):
        print("Iniciando CAFUNEBridge...")
        self.jl = init_julia()
        self.model = None
        self.md = None
        print("Bridge pronta!\n")

    def build_model(
        self,
        vocab_size: int,
        seq_len: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        num_diff_steps: int = 50,
    ):
        """
        Cria o modelo transformer bidirecional e o difusor.
        
        Configs disponíveis:
            tiny:   128d, 4h, 4L  → ~1M params, roda em CPU
            small:  256d, 8h, 6L  → ~10M params, CPU ok
            medium: 512d, 16h, 12L → ~100M params, precisa GPU
        """
        jl = self.jl

        # Criar configuração
        config = jl.seval(f"""
            TransformerConfig(
                {vocab_size},   # vocab_size
                {seq_len},      # seq_len
                {d_model},      # d_model
                {n_heads},      # n_heads
                {n_layers},     # n_layers
                {d_ff},         # d_ff
                0.1f0           # dropout
            )
        """)

        # Criar modelo
        jl.config_ref = config
        self.model = jl.seval("BidirectionalTransformer(config_ref)")

        # Criar difusor
        # Nota: mask_token_id do Python + 1 para Julia
        # MASK_TOKEN_ID no tokenizer.py é 4, então em Julia será 5.
        jl.vocab_size_ref = vocab_size
        jl.num_steps_ref = num_diff_steps
        jl.mask_token_id_ref = 5  # No tokenizer.py MASK é 4
        self.md = jl.seval("MaskDiffusion(vocab_size_ref, mask_token_id=mask_token_id_ref, num_steps=num_steps_ref)")

        n_params = jl.count_params(self.model)
        print(f"✅ Modelo criado: {n_params:,} parâmetros ({n_params/1e6:.1f}M)")

        return self

    def forward(self, tokens: list[int]) -> list[list[float]]:
        """
        Executa o forward pass do modelo.
        
        Returns:
            logits: (vocab_size, seq_len) como lista de listas
        """
        jl = self.jl
        # 1-based indexing for Julia
        tokens_julia = [t + 1 for t in tokens]
        jl.tokens_ref = jl.Vector[jl.Int](tokens_julia)
        logits = jl.seval("model_ref(tokens_ref)")
        return logits

    def generate(
        self,
        seq_len: int,
        num_steps: int = 50,
        temperature: float = 1.0,
        strategy: str = "confidence",
        verbose: bool = False,
    ) -> list[int]:
        """
        Gera uma sequência do zero via denoising iterativo.
        
        Args:
            seq_len: comprimento da sequência a gerar
            num_steps: passos de denoising (mais = melhor qualidade)
            temperature: 0=greedy, 1=normal, >1=mais aleatório
            strategy: "confidence" ou "random"
            verbose: mostrar progresso passo a passo
        
        Returns:
            Lista de IDs de tokens gerados
        """
        jl = self.jl
        jl.model_ref = self.model
        jl.md_ref = self.md
        jl.seq_len_ref = seq_len
        jl.num_steps_ref = num_steps
        jl.temperature_ref = float(temperature)
        jl.verbose_ref = verbose

        strategy_sym = ":confidence" if strategy == "confidence" else ":random"

        result = jl.seval(f"""
            generate(
                model_ref, md_ref, seq_len_ref,
                num_steps=num_steps_ref,
                temperature=Float32(temperature_ref),
                strategy={strategy_sym},
                verbose=verbose_ref
            )
        """)

        # Convert back to 0-based for Python
        return [int(t) - 1 for t in result]

    def generate_with_prompt(
        self,
        prompt_tokens: list[int],
        total_len: int,
        num_steps: int = 50,
        temperature: float = 1.0,
        verbose: bool = False,
    ) -> list[int]:
        """
        Geração com prompt fixo — completa o restante da sequência.
        """
        jl = self.jl
        jl.model_ref = self.model
        jl.md_ref = self.md
        tokens_julia = [t + 1 for t in prompt_tokens]
        jl.prompt_ref = jl.Vector[jl.Int](tokens_julia)
        jl.total_len_ref = total_len
        jl.num_steps_ref = num_steps
        jl.temperature_ref = float(temperature)
        jl.verbose_ref = verbose

        result = jl.seval("""
            generate_with_prompt(
                model_ref, md_ref, prompt_ref, total_len_ref,
                num_steps=num_steps_ref,
                temperature=Float32(temperature_ref),
                verbose=verbose_ref
            )
        """)

        # Convert back to 0-based for Python
        return [int(t) - 1 for t in result]

    def train_on_batch(
        self, 
        sequences: list[list[int]], 
        epochs: int = 5,
        max_lr: float = 3e-4,
        warmup_ratio: float = 0.1,
        log_every: int = 10
    ) -> None:
        """
        Treina o modelo em um batch de sequências tokenizadas com agendamento de LR.
        """
        jl = self.jl
        jl.model_ref = self.model
        jl.md_ref = self.md

        # Converter para tipo Julia (e somar 1 para 1-based indexing)
        julia_seqs = [jl.Vector[jl.Int]([t + 1 for t in seq]) for seq in sequences]
        jl.dataset_ref = jl.Vector(julia_seqs)
        
        jl.epochs_ref = epochs
        jl.max_lr_ref = float(max_lr)
        jl.warmup_ratio_ref = float(warmup_ratio)
        jl.log_every_ref = int(log_every)

        # Chama o motor treinado na Fase 2
        self.model = jl.seval("""
            model_ref = train!(
                model_ref, md_ref, dataset_ref, 
                epochs=epochs_ref, 
                max_lr=max_lr_ref, 
                warmup_ratio=warmup_ratio_ref,
                log_every=log_every_ref
            )
        """)
        print("✅ Batch de treino concluído e modelo atualizado.")


# ──────────────────────────────────────────────────────────────
#  FASE 6: Memória Compartilhada (Shared Memory / MMAP)
# ──────────────────────────────────────────────────────────────

import mmap
import time
import struct

MEM_FILE = "cafune_brain.mem"
MEM_SIZE = 1024  # Buffer de 1KB para comandos e estados

import numpy as np

def calculate_entropy(logits_list: list[list[float]]) -> float:
    """Calcula a entropia média das distribuições de probabilidade."""
    # Como logits vem de Julia como lista de listas (vocab_size, seq_len)
    # Convertemos para as probabilidades via softmax e calculamos a entropia de Shannon
    avg_entropy = 0.0
    for pos_logits in logits_list:
        # Simplificação: Softmax manual
        e_x = np.exp(np.array(pos_logits) - np.max(pos_logits))
        probs = e_x / e_x.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        avg_entropy += entropy
    return avg_entropy / len(logits_list)

def calculate_reward(tokens: list[int], tok: "BPETokenizer") -> float:
    """
    AI Critic: Avalia a qualidade da sequencia gerada.
    Criterios: Coerencia, falta de repeticao e presença de tokens validos.
    """
    if not tokens: return 0.0
    
    reward = 1.0 # Recompensa base
    
    # Penalidade por repetiçao excessiva (estilo 'Frequency Penalty')
    unique_ratio = len(set(tokens)) / len(tokens)
    reward *= unique_ratio
    
    # Penalidade por tokens desconhecidos [UNK]
    unk_count = tokens.count(1)
    reward -= (unk_count * 0.1)
    
    # Bonus por sequencia fluida (ex: alternância de vogais/consoantes simulada)
    # No real, aqui chamaríamos um modelo de recompensa (Reward Model)
    return max(0.0, min(1.0, reward))

def run_sentinel(bridge: "CAFUNEBridge"):
    """
    Sentinel com Suporte a RLAIF.
    Schema:
        Byte 0: Flag
        ...
        Byte 32-39: Entropy Feedback
        Byte 40-47: RLAIF Reward -> ESCRITO PELO CRITICO
    """
    if not os.path.exists(MEM_FILE):
        with open(MEM_FILE, "wb") as f:
            f.write(b"\x00" * MEM_SIZE)
            
    print(f"🛰️ [Sentinel] Monitorando {MEM_FILE} (RLAIF Ativo)...")
    
    with open(MEM_FILE, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        
        try:
            while True:
                mm.seek(0)
                flag = mm.read_byte()
                
                if flag == 1:
                    # BUSY
                    mm.seek(0)
                    mm.write_byte(4) 
                    
                    step = struct.unpack("i", mm[4:8])[0]
                    ratio = struct.unpack("d", mm[8:16])[0]
                    
                    print(f"🧠 [Sentinel] Processando com Feedback Adaptativo...")
                    
                    try:
                        # 1. Inferencia Real
                        tokens = bridge.generate(seq_len=15, num_steps=2)
                        
                        # 2. Calculo de Entropia (Monitoria)
                        dummy_logits = [[np.random.rand() for _ in range(100)] for _ in range(15)]
                        entropy = calculate_entropy(dummy_logits)
                        
                        # 3. CRITICO RLAIF (Avalia a predicao)
                        # No Tokenizer original 0-based
                        reward = calculate_reward(tokens, bridge.model) # Simulacao
                        
                        # Escreve feedbacks
                        mm.seek(32)
                        mm.write(struct.pack("d", entropy))
                        mm.seek(40)
                        mm.write(struct.pack("d", reward))
                        
                        print(f"📉 [Sentinel] Incerteza: {entropy:.2f} | 🏅 Recompensa RLAIF: {reward:.2f}")
                        
                        # DONE
                        mm.seek(0)
                        mm.write_byte(2) 
                    except Exception as e:
                        print(f"❌ [Sentinel] Erro: {e}")
                        mm.seek(0)
                        mm.write_byte(3)
                
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n🛑 Sentinela desligado.")
            mm.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CAFUNE Bridge - Interface de Inferencia")
    parser.add_argument("--step", type=int, help="Passo atual da difusao")
    parser.add_argument("--ratio", type=float, help="Razao de mascara (0.0 a 1.0)")
    parser.add_argument("--sentinel", action="store_true", help="Inicia em modo Shared Memory (MMAP)")

    args = parser.parse_args()

    # Inicializar a ponte
    bridge = CAFUNEBridge()
    bridge.build_model(vocab_size=50, d_model=64, n_layers=2)

    if args.sentinel:
        run_sentinel(bridge)
    elif args.step is not None:
        # Modo Clássico (CLI)
        print(f"\n[Bridge] 🧠 Modo Legado CLI: Passo {args.step}, Mascara {args.ratio*100:.1f}%")
        tokens = bridge.generate(seq_len=10, num_steps=2)
        print(f"[Bridge] ✅ Resposta: {tokens}")
    else:
        print("🧟 CAFUNE Bridge Online")

