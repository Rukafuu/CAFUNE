# Mapa de consolidação

## Atual

- `julia/src/`: implementação canônica do motor.
- `julia/inference.jl`: inferência híbrida em evolução.
- `julia/inference.jl`: geração local BPE por difusão condicionada.
- `python/bridge.py`: cliente síncrono do motor.
- `python/dashboard.py`: monitor local.
- `python/raegis_sentinel.py`, `guardian_reward.py` e
  `rlaif_evaluator.py`: avaliação e recompensa.
- `python/tests/`: testes automatizados.

## Compatibilidade ou experimento

- `julia/train_unified.jl`: snapshot autocontido da evolução de maio; deve ser
  decomposto antes de voltar a ser um lançador oficial.
- `haskell/`: protótipo de orquestração tipada.
- `c/`: protótipo CUDA, não obrigatório.
- scripts `debug_*`, `diag*`, `install_*` e `test_*`: ferramentas de
  desenvolvimento, candidatas a `tools/`.

## Legado

- `phase1_report.md`: registro histórico da arquitetura de 0,8M parâmetros.
- `STRATEGY_AAA.md`: visão e roadmap, não contrato técnico.
- `python/train.py` e lançadores antigos: pipeline anterior ao motor híbrido.
- O antigo treino SNN de 65 caracteres foi removido; um refinador SNN novo só
  será treinado depois de o Transformer BPE produzir respostas coerentes.
- O antigo worktree `CAFUNE/optimistic-mirzakhani/` foi auditado e removido.
  Seus únicos utilitários relevantes foram consolidados em
  `python/download_canarim.py` e `julia/tools/test_cuda.jl`.

## Artefatos gerados

- `*.bson`, `*.mem`, `*.jsonl`, `wandb/`, logs e caches Python.

Nenhum item legado deve voltar a ser importado pelo fluxo principal. A remoção
física pode ocorrer depois que o fluxo atual tiver testes ponta a ponta.
