# Registro de experimentos

Este arquivo preserva a trilha científica do CAFUNE. Registre uma entrada antes
de iniciar qualquer treino que produza resultado comparável.

Cada entrada deve declarar hipótese, configuração, dados/splits, sementes,
métrica de decisão, estado e links para artefatos. Não promova uma hipótese a
conclusão sem um relatório de avaliação reproduzível.

## EXP-001 — CAFUNE-mini como bancada de ablações

- **Hipótese:** uma configuração de aproximadamente 7M parâmetros permite
  comparar escolhas arquiteturais antes de escalar para o modelo canônico de
  45M.
- **Branch:** `feature/cafune-mini`.
- **Configuração:** `config/experiments/cafune-mini.toml` — 8 camadas,
  `d_model=256`, 8 heads, `d_ff=1024`, 7.071.744 parâmetros.
- **Controles:** SentencePiece BPE 1.999, `seq_len=128`,
  `python/dataset_tokens.json` e `python/dataset_splits.json` idênticos à
  baseline canônica.
- **Métrica de decisão:** `aggregate.loss_mean` no split de validação, com
  máscaras em 0,10 / 0,25 / 0,50 / 0,75 / 0,90 e três seeds.
- **Estado:** pronto para treino; sanity training e CI passaram. Ainda não há
  métrica de qualidade ou conclusão empírica.
- **Próxima comparação:** MHA-only versus MHA+SSA, com esta configuração e os
  mesmos controles.

## EXP-002 — Controle MHA-only para a SSA

- **Hipótese:** a SSA pode ser comparada de forma isolada contra MHA quando
  capacidade, tokenizer, dados, splits, avaliação e seeds permanecem fixos.
- **Branch:** `feature/mha-baseline`.
- **Configuração:** `config/experiments/mha-baseline.toml` — 8 camadas MHA,
  `d_model=256`, 8 heads e `d_ff=960`, total de 7.071.232 parâmetros.
- **Pareamento:** o híbrido EXP-001 tem 7.071.744 parâmetros; a diferença é de
  512 (0,007%) porque a FFN varia em incrementos discretos. Não há SSA nesta
  variante.
- **Controles:** o mesmo BPE 1.999, sequência 128, dataset, split, seeds e
  métrica de decisão de EXP-001.
- **Estado:** implementação e sanity check em CI; nenhum resultado de treino
  comparável foi produzido ainda.

## Convenções de artefatos

- Checkpoints: `julia/checkpoints/<experimento>/`.
- Avaliações: `julia/evaluations/`; preserve no JSON a configuração, checkpoint,
  split, hash do dataset e seeds.
- Resultados agregados e interpretação entram neste arquivo após confirmação de
  reprodutibilidade. Logs brutos não pertencem ao Git.
