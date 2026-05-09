import json
import os

print("Preparando Dataset SFT via BPE Guloso...")
vocab_path = "vocab_bpe.json"
data_path = "social_data.json"
out_path = "../julia/tokenized_dataset.jl"

with open(vocab_path, "r", encoding="utf-8") as f:
    vocab_data = json.load(f)
vocab = vocab_data["vocab"]

# Sort vocab keys by length descending for greedy match
sorted_vocab_keys = sorted(vocab.keys(), key=len, reverse=True)

with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = []
for item in data:
    text = item["user"] + " [SEP] " + item["response"]
    tokens = []
    
    # Greedy tokenizer
    temp_text = text
    while temp_text:
        matched = False
        for k in sorted_vocab_keys:
            if temp_text.startswith(k):
                tokens.append(vocab[k])
                temp_text = temp_text[len(k):]
                matched = True
                break
        if not matched:
            tokens.append(1) # UNK
            temp_text = temp_text[1:]
            
    # Fix Python 0-index to Julia 1-index for array indexing
    tokens = [t + 1 for t in tokens]
    
    # Pad to 128 with the adjusted [PAD] token (0 + 1 = 1)
    if len(tokens) < 128:
        pad_size = 128 - len(tokens)
        tokens = [1] * pad_size + tokens
    else:
        tokens = tokens[-128:]
        
    dataset.append(tokens)

with open(out_path, "w", encoding="utf-8") as f:
    f.write("# Gerado automaticamente por prepare_dataset.py\n")
    f.write("real_dataset = [\n")
    for toks in dataset:
        f.write(f"    reshape({toks}, 128, 1),\n")
    f.write("]\n")

print(f"Dataset salvo com {len(dataset)} sequencias em {out_path}.")
