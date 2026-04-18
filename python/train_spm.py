"""
train_spm.py — Treina tokenizador SentencePiece no dataset PT-BR do CAFUNE

Uso:
    python python/train_spm.py

Gera:
    python/cafune_spm.model  — modelo SentencePiece
    python/cafune_spm.vocab  — vocabulário legível
    python/vocab_spm.json    — mapa {token: id} compatível com o Julia
"""

import io
import sys
import json
import tempfile
import os
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import sentencepiece as spm

SCRIPT_DIR   = Path(__file__).parent
DATA_FILE    = SCRIPT_DIR / "bercario_data.jsonl"
MODEL_PREFIX = str(SCRIPT_DIR / "cafune_spm")
VOCAB_JSON   = SCRIPT_DIR / "vocab_spm.json"

VOCAB_SIZE   = 2000   # subwords PT-BR — suficiente pra 5M params
CHARACTER_COVERAGE = 0.9999  # cobre todos os chars PT-BR incluindo acentos

# Tokens especiais — mesmos IDs do tokenizador antigo para compatibilidade
PAD_ID  = 0
UNK_ID  = 1
BOS_ID  = 2
EOS_ID  = 3
MASK_ID = 4  # [MASK] para difusão mascarada — não é nativo do SPM, adicionamos manualmente


def extract_texts() -> list[str]:
    texts = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                for field in ("prompt", "target", "user", "response", "text"):
                    if field in obj and obj[field]:
                        texts.append(str(obj[field]).strip())
            except json.JSONDecodeError:
                pass
    return texts


def main():
    print("=== CAFUNE SentencePiece Trainer ===")
    print(f"Dataset: {DATA_FILE}")

    texts = extract_texts()
    print(f"Textos extraidos: {len(texts)}")
    print(f"Total chars: {sum(len(t) for t in texts):,}")

    # Escreve corpus em arquivo temporário (SPM precisa de arquivo)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt",
                                     delete=False, dir=SCRIPT_DIR) as tmp:
        tmp_path = tmp.name
        for t in texts:
            tmp.write(t + "\n")

    print(f"\nTreinando SentencePiece (vocab_size={VOCAB_SIZE})...")

    # IDs reservados: PAD=0, UNK=1, BOS=2, EOS=3 — MASK=4 adicionado manualmente depois
    spm.SentencePieceTrainer.train(
        input=tmp_path,
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE - 1,   # -1 porque vamos inserir [MASK] manualmente
        character_coverage=CHARACTER_COVERAGE,
        model_type="bpe",            # BPE — melhor para PT-BR morfológico
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_piece="[PAD]",
        unk_piece="[UNK]",
        bos_piece="[BOS]",
        eos_piece="[EOS]",
        # Não dividir por dígitos/pontuação agressivamente
        split_digits=False,
        # Preservar espaços como parte do token (estilo GPT)
        add_dummy_prefix=False,
        # Normalização mínima — preservar acentos PT-BR
        normalization_rule_name="identity",
        user_defined_symbols=["[MASK]"],  # ID 4 = MASK
        # Treino mais robusto
        input_sentence_size=500000,
        shuffle_input_sentence=True,
    )

    os.unlink(tmp_path)
    print(f"Modelo salvo: {MODEL_PREFIX}.model")

    # Carrega e exporta vocab JSON compatível com Julia
    sp = spm.SentencePieceProcessor()
    sp.load(f"{MODEL_PREFIX}.model")

    real_vocab_size = sp.get_piece_size()
    print(f"Vocab real: {real_vocab_size} tokens")

    # Monta char2id para o Julia
    char2id = {}
    for i in range(real_vocab_size):
        piece = sp.id_to_piece(i)
        char2id[piece] = i

    # Estatísticas
    print(f"\nEstatísticas:")
    print(f"  PAD=[PAD]={sp.piece_to_id('[PAD]')}")
    print(f"  UNK=[UNK]={sp.piece_to_id('[UNK]')}")
    print(f"  BOS=[BOS]={sp.piece_to_id('[BOS]')}")
    print(f"  EOS=[EOS]={sp.piece_to_id('[EOS]')}")
    print(f"  MASK=[MASK]={sp.piece_to_id('[MASK]')}")

    # Testa tokenização
    test = "Ultimamente tenho me sentido meio perdido, sabe?"
    tokens = sp.encode(test, out_type=str)
    ids = sp.encode(test)
    print(f"\nTeste: '{test}'")
    print(f"  Tokens: {tokens}")
    print(f"  IDs:    {ids}")
    print(f"  Compressão: {len(test)} chars → {len(ids)} tokens")

    # Salva vocab JSON
    out = {
        "char2id": char2id,
        "vocab_size": real_vocab_size,
        "model_file": "cafune_spm.model",
        "tokenizer": "sentencepiece",
        "mask_token_id": sp.piece_to_id("[MASK]"),
        "pad_token_id":  PAD_ID,
        "unk_token_id":  UNK_ID,
        "bos_token_id":  BOS_ID,
        "eos_token_id":  EOS_ID,
    }
    VOCAB_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nVocab JSON salvo: {VOCAB_JSON}")
    print(f"\nPróximo passo: python python/apply_spm.py  (substitui vocab.json)")


if __name__ == "__main__":
    main()
