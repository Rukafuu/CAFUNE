"""
download_canarim.py — Baixa o Canarim-Instruct-PTBR-Dataset e integra com
bercario_data.jsonl existente.

Uso:
    python python/download_canarim.py            # preview — não sobrescreve
    python python/download_canarim.py --apply    # integra e salva

Resultado:
    bercario_data.jsonl atualizado com Canarim + dados originais (sem duplicatas)

Estratégia:
    - bercario_data.jsonl original: especialização emocional PT-BR (mantido inteiro)
    - Canarim: fluência geral PT-BR (amostrado pra não dominar demais)
    - Deduplicação por hash do prompt
    - Filtra entradas com output muito curto (<10 chars) ou muito longo (>800 chars)
"""

import io
import sys
import json
import hashlib
import random
import argparse
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from datasets import load_dataset

SCRIPT_DIR   = Path(__file__).parent
BERCARIO     = SCRIPT_DIR / "bercario_data.jsonl"
BERCARIO_BAK = SCRIPT_DIR / "bercario_data.jsonl.bak"

# Quantas entradas do Canarim incluir
# 317K total — usamos 50K pra não dominar o bercario emocional
CANARIM_SAMPLE = 50_000

# Filtros de qualidade
MIN_OUTPUT_LEN = 10    # respostas muito curtas não ensinam nada
MAX_OUTPUT_LEN = 800   # respostas enormes não cabem bem em seq_len=128
MIN_PROMPT_LEN = 5


def sha8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def load_existing() -> tuple[list[dict], set[str]]:
    """Carrega bercario_data.jsonl existente."""
    entries = []
    seen = set()
    if not BERCARIO.exists():
        return entries, seen
    with open(BERCARIO, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                prompt = obj.get("prompt", "").strip()
                if prompt:
                    seen.add(sha8(prompt))
                    entries.append(obj)
            except json.JSONDecodeError:
                pass
    return entries, seen


def build_prompt(instruction: str, input_text: str) -> str:
    """Monta o prompt no mesmo estilo do bercario (simples, sem template verboso)."""
    instruction = instruction.strip()
    input_text  = (input_text or "").strip()
    if input_text:
        return f"{instruction}\n{input_text}"
    return instruction


def filter_entry(prompt: str, target: str) -> bool:
    """Retorna True se a entrada passa no filtro de qualidade."""
    if len(prompt)  < MIN_PROMPT_LEN:  return False
    if len(target)  < MIN_OUTPUT_LEN:  return False
    if len(target)  > MAX_OUTPUT_LEN:  return False
    # Remove entradas que são só listas de bullets sem frase
    if target.count("\n") > 15:        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Aplica a integração (sobrescreve bercario_data.jsonl)")
    parser.add_argument("--sample", type=int, default=CANARIM_SAMPLE,
                        help=f"Quantas entradas do Canarim usar (padrão: {CANARIM_SAMPLE})")
    args = parser.parse_args()

    print("=== Integração Canarim + bercario_data ===\n")

    # 1. Carrega dados existentes
    existing, seen_hashes = load_existing()
    print(f"bercario_data.jsonl existente: {len(existing)} entradas")

    # 2. Baixa Canarim
    print(f"\nBaixando Canarim-Instruct-PTBR-Dataset...")
    print("(Pode demorar alguns minutos na primeira vez — ~64MB)\n")
    ds = load_dataset("dominguesm/Canarim-Instruct-PTBR-Dataset", split="train")
    print(f"Canarim carregado: {len(ds)} exemplos")

    # 3. Converte e filtra
    canarim_entries = []
    for row in ds:
        prompt = build_prompt(row["instruction"], row.get("input", ""))
        target = (row.get("output") or "").strip()

        if not filter_entry(prompt, target):
            continue

        h = sha8(prompt)
        if h in seen_hashes:
            continue

        seen_hashes.add(h)
        canarim_entries.append({
            "prompt":  prompt,
            "target":  target,
            "source":  "canarim",
        })

    print(f"Após filtro de qualidade: {len(canarim_entries)} entradas válidas")

    # 4. Amostra pra não dominar os dados emocionais
    if len(canarim_entries) > args.sample:
        random.seed(42)
        canarim_entries = random.sample(canarim_entries, args.sample)
        print(f"Amostrado para: {len(canarim_entries)} entradas")

    # 5. Combina (bercario primeiro — maior prioridade temática)
    combined = existing + canarim_entries
    print(f"\nTotal combinado: {len(combined)} entradas")
    print(f"  bercario original: {len(existing)}")
    print(f"  Canarim adicionado: {len(canarim_entries)}")

    # 6. Estatísticas de comprimento
    prompt_lens = [len(e["prompt"]) for e in combined]
    target_lens = [len(e["target"]) for e in combined]
    print(f"\nEstatísticas:")
    print(f"  Prompt: avg={sum(prompt_lens)//len(prompt_lens)} chars, "
          f"max={max(prompt_lens)}")
    print(f"  Target: avg={sum(target_lens)//len(target_lens)} chars, "
          f"max={max(target_lens)}")

    # 7. Exemplo
    sample = canarim_entries[0] if canarim_entries else {}
    print(f"\nExemplo Canarim:")
    print(f"  Prompt: {sample.get('prompt','')[:80]}...")
    print(f"  Target: {sample.get('target','')[:80]}...")

    if not args.apply:
        print("\n[DRY RUN] Use --apply para salvar.")
        return

    # 8. Backup + salva
    if BERCARIO.exists():
        BERCARIO.rename(BERCARIO_BAK)
        print(f"\nBackup: {BERCARIO_BAK}")

    with open(BERCARIO, "w", encoding="utf-8") as f:
        for entry in combined:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Salvo: {BERCARIO} ({len(combined)} entradas)")
    print("\nPróximo passo: python python/train_spm.py  (retreinar tokenizador)")
    print("Depois:        python python/tokenize_dataset.py")


if __name__ == "__main__":
    main()
