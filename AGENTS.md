# CAFUNE — guia de colaboração

## Fluxo Git e releases

- `main` é a linha estável: mantenha-a verde no CI e não faça commits diretos.
- Todo trabalho começa em uma branch publicada e integrada por pull request.
- Use `feature/<hipotese>` para capacidades ou experimentos, por exemplo
  `feature/cafune-mini`, `feature/mha-baseline` e `feature/tokenizer-ablation`.
- Use `fix/<problema>` para correções. Para uma correção urgente de uma release
  publicada, use `hotfix/<problema>`.
- Releases são tags semânticas (`v0.1.0`, `v0.1.1`). Crie `release/vX.Y` apenas
  quando for necessário manter correções nessa linha sem incluir trabalho novo.
- Não misture hipóteses experimentais diferentes na mesma branch ou PR.

## Contratos técnicos

- A implementação canônica é `julia/src/`; não reviva `julia/train_unified.jl`
  nem lançadores legados como uma segunda fonte do modelo.
- `config/research.toml` define a configuração de pesquisa canônica.
- `config/experiments/cafune-mini.toml` é a configuração de ~7,07M parâmetros
  para ablações. Preserve tokenizer, dados e splits ao comparar mini vs baseline.
- `python/dataset_splits.json` é o contrato de splits determinísticos. Se o
  dataset tokenizado mudar, regenere os splits antes de treinar ou avaliar.
- O melhor checkpoint é escolhido por `aggregate.loss_mean` da avaliação de
  validação; nunca por loss de treino.
- Antes de iniciar um treino comparável, registre hipótese e controles em
  `docs/EXPERIMENTS.md`. Só registre conclusões após existir avaliação
  reproduzível.
- Checkpoints, avaliações geradas, logs, caches e mmap são artefatos de runtime,
  não código-fonte.

## Verificação

Antes de abrir PR, rode o que estiver disponível:

```powershell
python python/verify_research.py
python -m pytest python/tests -q
julia --project=julia julia/smoke_test.jl
```

Se Julia não estiver disponível localmente, não declare a alteração validada
somente por inspeção: publique a branch e aguarde o GitHub Actions concluir.
Não inicie treino, RLAIF ou chamadas externas como parte da verificação.
