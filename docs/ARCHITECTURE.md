# Arquitetura atual do CAFUNE

Este documento define a arquitetura canônica do projeto a partir da evolução
neuromórfica de 11 de maio de 2026.

## Núcleo suportado

O CAFUNE atual é um modelo de linguagem de difusão mascarada, bidirecional,
implementado em Julia. O núcleo combina:

- Transformer bidirecional com RoPE;
- blocos híbridos com Spiking Synchrony Attention (SSA);
- reservatório SNN 11D com células PLIF, SFA e gradientes substitutos;
- treino supervisionado por difusão e alinhamento RLAIF;
- observabilidade e serviços auxiliares em Python.

## Responsabilidades

| Área | Responsabilidade | Fonte canônica |
|---|---|---|
| Modelo | Transformer, SSA e RoPE | `julia/src/transformer.jl` |
| Dinâmica SNN | PLIF, SFA, hipercubo 11D e decoder | `julia/src/snn_core.jl` |
| Difusão | Máscara e agenda de denoising | `julia/src/diffusion.jl` |
| Treino | Loss, otimização e métricas | `julia/src/training.jl` |
| Inferência | Sampling e geração | `julia/src/sampling.jl` |
| Pacote | API pública Julia | `julia/src/CAFUNE.jl` |
| IPC | Layout e caminhos do mmap | `python/cafune_config.py` |
| Serviços | Dashboard e avaliação | `python/` |

`julia/train_unified.jl` é um experimento histórico autocontido. Ele não deve
ser usado como segunda definição do modelo: novas mudanças pertencem aos
arquivos de `julia/src/`.

## Contrato de memória compartilhada

Existe um único arquivo canônico na raiz do repositório:
`cafune_brain.mem`, com 2048 bytes. Os offsets públicos ficam centralizados em
`python/cafune_config.py`. Serviços não devem construir caminhos próprios.

## Artefatos locais

Checkpoints, logs, dados gerados, execuções do W&B e arquivos mmap são estado
de execução, não código-fonte. Eles permanecem ignorados pelo Git e não devem
ser duplicados em subdiretórios.

## Metas experimentais

`config/research.toml` é a fonte canônica dos hiperparâmetros e metas públicas.

## Variante BitNet

`julia/main_training.jl --bitnet` mantém pesos mestres em Float32, mas usa
pesos ternários b1.58 e ativações fake-quant int8 no forward. O gradiente passa
por um straight-through estimator (STE). A baseline Float32 continua sendo o
padrão e seus checkpoints permanecem separados dos checkpoints BitNet.

O repositório externo `bitnet.cpp` é opcional e somente de inferência. Aponte
para uma instalação local com `BITNET_ROOT`; seus kernels atuais têm dimensões
fixas do modelo oficial 2B e não são ligados diretamente às matrizes do CAFUNE.
`python/verify_research.py` confere automaticamente contagem de parâmetros,
arquitetura e contrato do tokenizador. Resultados de loss e latência só mudam
de `PENDING` para `PASS` quando houver artefatos reproduzíveis.

## Estado das outras linguagens

- Python é a camada de serviços, dados, avaliação e observabilidade.
- Haskell é um orquestrador experimental; não define o modelo.
- C/CUDA é uma aceleração experimental e opcional; o caminho Julia puro é o
  fallback suportado.
