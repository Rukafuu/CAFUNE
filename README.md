<p align="center">
  <img src="assets/logo.png" width="400" alt="CAFUNE">
</p>

# CAFUNE

**Cognição Artificial Fundamentada em Neuromorfologia** — um motor de
linguagem experimental que combina difusão mascarada bidirecional e dinâmica
SNN.

O núcleo atual é escrito em Julia e reúne Transformer bidirecional, RoPE,
Spiking Synchrony Attention, células PLIF/SFA e um reservatório hipercúbico
11D. Python cuida dos serviços, dados, avaliação RLAIF e observabilidade.
Haskell e CUDA permanecem integrações experimentais e opcionais.

## Estado do projeto

O repositório está em consolidação. A implementação canônica vive em
`julia/src/`; scripts autocontidos antigos não são uma segunda fonte do modelo.
Consulte [a arquitetura](docs/ARCHITECTURE.md) e o
[mapa de consolidação](docs/REPOSITORY_MAP.md) antes de alterar componentes.

## Estrutura

```text
assets/             identidade visual e diagramas
c/                  aceleração CUDA experimental
docs/               arquitetura e decisões do projeto
haskell/             orquestração experimental
julia/src/           motor canônico
julia/*.jl           lançadores e ferramentas de treino/inferência
python/              serviços, datasets e testes
```

## Preparação

Requisitos principais: Python 3.11+, Julia 1.10+ e, opcionalmente, uma GPU
NVIDIA compatível com CUDA.

```powershell
Copy-Item .env.example .env
python -m pip install -r python/requirements.txt
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
```

Verifique a instalação e os artefatos sem iniciar treino:

```powershell
python python/cafune.py doctor
python python/cafune.py init-runtime
python -m pytest python/tests -q
julia --project=julia julia/smoke_test.jl
```

Audite as metas experimentais contra a configuração e os artefatos locais:

```powershell
python python/verify_research.py
```

A configuração de pesquisa atual possui 12 camadas, BPE SentencePiece de 1.999
tokens e 45.363.968 parâmetros reais. A contagem considera seis blocos MHA e
seis blocos SSA. Loss de validação
continua uma meta pendente até existir avaliação reproduzível. Latência é
apenas diagnóstico: qualidade e execução local são os critérios principais.

Prepare os splits determinísticos antes do treino:

```powershell
python python/prepare_splits.py
```

Valide forward, backward, optimizer e checkpoint sem iniciar o treino longo:

```powershell
julia --project=julia julia/main_training.jl --sanity
```

O resultado fica isolado em `julia/checkpoints/sanity/` e nunca é usado pela
inferência principal.

Dashboard local:

```powershell
python python/dashboard.py
```

Depois de existir um checkpoint compatível em `julia/checkpoints/cafune_best.bson`,
execute uma resposta inteiramente local com:

```powershell
julia --project=julia julia/inference.jl "Olá, quem é você?"
```

O daemon usado pelo bridge Python é iniciado com:

```powershell
julia --project=julia julia/engine_mmap.jl
```

## Contratos importantes

- Há um único mmap canônico: `cafune_brain.mem`, na raiz, com 2048 bytes.
- Caminhos de runtime são definidos por `python/cafune_config.py`.
- Checkpoints, logs, caches e execuções W&B não pertencem ao código-fonte.
- `OPENROUTER_API_KEY` habilita o avaliador RLAIF externo.
- CUDA, Haskell, BitNet e Flair são opcionais; o núcleo não deve depender
  deles para ser importado e testado.

## Segurança operacional

Treino e RLAIF podem alterar checkpoints e consumir serviços externos. Rode o
`doctor` e os testes antes de iniciar esses processos. Nenhum comando de
verificação acima realiza treino ou chama APIs de modelos.

## Licença

O código-fonte e a documentação autoral do CAFUNE são disponibilizados sob a
[Apache License 2.0](LICENSE). Datasets, modelos, pesos, vocabulários derivados,
artefatos de treino e imagens não são automaticamente cobertos por essa licença;
consulte [DATA_AND_ASSETS.md](DATA_AND_ASSETS.md) antes de reutilizá-los.
### Variante ternária BitNet

```powershell
julia --project=julia julia/main_training.jl --sanity --bitnet
julia --project=julia julia/main_training.jl --bitnet
```

A baseline Float32 segue como padrão. A variante usa pesos mestres Float32 para
treino, pesos ternários no forward e checkpoints próprios em
`julia/checkpoints/bitnet/`.

RLAIF é opcional e só deve ser ativado quando um teacher compatível estiver
respondendo pelo mmap:

```powershell
julia --project=julia julia/main_training.jl --bitnet --rlaif
```
