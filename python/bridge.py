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

    def train_on_batch(self, sequences: list[list[int]], epochs: int = 5) -> None:
        """
        Treina o modelo em um batch de sequências tokenizadas.
        """
        jl = self.jl
        jl.model_ref = self.model
        jl.md_ref = self.md

        # Converter para tipo Julia (e somar 1 para 1-based indexing)
        julia_seqs = [jl.Vector[jl.Int]([t + 1 for t in seq]) for seq in sequences]
        jl.dataset_ref = jl.Vector(julia_seqs)
        jl.epochs_ref = epochs

        # Fase 2: O retorno de train! e o modelo atualizado
        self.model = jl.seval("model_ref = train!(model_ref, md_ref, dataset_ref, epochs=epochs_ref)")


# ──────────────────────────────────────────────────────────────
#  Demo rápida (sem Julia instalada: mostra o fluxo esperado)
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CAFUNE Bridge - Interface de Inferencia")
    parser.add_argument("--step", type=int, help="Passo atual da difusao")
    parser.add_argument("--ratio", type=float, help="Razao de mascara (0.0 a 1.0)")
    parser.add_argument("--prompt", type=str, default="A vida", help="Texto inicial")
    parser.add_argument("--batch", type=int, default=1, help="Tamanho do batch")

    args = parser.parse_args()

    if args.step is not None:
        print(f"\n[Bridge] 🧠 Recebido do Haskell: Passo {args.step}, Mascara {args.ratio*100:.1f}%")
        
        # Inicializar a ponte real (Isso chama a Julia e carrega o CAFUNE)
        bridge = CAFUNEBridge()
        
        # Configurar um modelo ultra-leve para o teste de orquestração
        # vocab=50, d_model=64, layers=2
        bridge.build_model(
            vocab_size=50, 
            seq_len=128, 
            d_model=64, 
            n_heads=4, 
            n_layers=2, 
            d_ff=128,
            num_diff_steps=args.step if args.step > 0 else 1
        )
        
        # Executar uma geração rápida para validar o sistema completo
        print(f"[Bridge] 🧪 Solicitando Denoising ao Motor Julia (Batch: {args.batch})...")
        tokens = bridge.generate(seq_len=10, num_steps=2) # 2 passos rápidos
        print(f"[Bridge] ✅ Resposta do Motor: {tokens}")
        print(f"[Bridge] ✨ Passo {args.step} concluído com sucesso.")
    else:
        print("🧟 CAFUNE — Interface da Bridge Python-Julia")
        print("=" * 50)
        print("\nPara usar, execute:")
        print("\n  from bridge import CAFUNEBridge")
        print("  bridge = CAFUNEBridge()")
        print("  bridge.build_model(vocab_size=50)")

