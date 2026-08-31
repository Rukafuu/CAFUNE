# Data, models, and assets

The Apache License 2.0 in this repository applies to CAFUNE source code and
documentation authored for the project. It does not automatically license or
relicense datasets, trained weights, generated corpora, tokenizer models,
training artifacts, or image assets.

## Current provenance status

The following tracked material requires separate provenance and licensing
review before redistribution or downstream commercial use:

- `python/bercario_data.jsonl`, including entries marked `gemini-web`,
  `gemini-seed`, or without a source;
- derived dataset splits, token files, vocabularies, and SentencePiece models;
- `python/social_data.json` and copies or derivatives of that data;
- images under `assets/` and `docs/assets/`;
- checkpoints, logs, and other training outputs if they are added later.

No license is granted for these materials by the repository's Apache-2.0
license unless a file-specific notice explicitly says otherwise.

## Canarim importer

`python/download_canarim.py` references the external
`dominguesm/Canarim-Instruct-PTBR-Dataset`, which is distributed under
CC BY-NC 4.0 at the time of this notice. Content imported from that dataset
remains subject to its upstream terms and is not covered by Apache-2.0.

Before adding or publishing new data, models, weights, or assets, record their
source, applicable license, transformation history, and redistribution terms.
