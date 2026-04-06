import json
import collections
from pathlib import Path

def train_bpe(corpus_path, vocab_size=500):
    print(f"--- [TREINO BPE: CAFUNE ENGINE FASE 2] ---")
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 1. Extrair todos os textos para o corpus bruto
    texts = []
    for item in data:
        texts.append(item["user"])
        texts.append(item["response"])
    
    # 2. Inicializar vocabulário com caracteres básicos
    vocab = collections.Counter("".join(texts))
    ids = {char: i + 5 for i, char in enumerate(sorted(vocab.keys()))}
    # Tokens especiais: [PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3, [MASK]=4
    
    # 3. Representar o corpus como sequências de tokens
    corpus = [list(text) for text in texts]
    
    merges = {}
    num_merges = vocab_size - len(ids) - 5
    
    print(f"Iniciando {num_merges} mergens para atingir VocabSize={vocab_size}...")
    
    for i in range(num_merges):
        pairs = collections.Counter()
        for word in corpus:
            for pair in zip(word, word[1:]):
                pairs[pair] += 1
        
        if not pairs:
            break
        
        best_pair = pairs.most_common(1)[0][0]
        new_token = f"{best_pair[0]}{best_pair[1]}"
        
        new_id = len(ids) + 5
        ids[new_token] = new_id
        merges[best_pair] = new_token
        
        # Atualizar corpus com o novo token
        new_corpus = []
        for word in corpus:
            new_word = []
            skip = False
            for j in range(len(word)-1):
                if skip:
                    skip = False
                    continue
                if (word[j], word[j+1]) == best_pair:
                    new_word.append(new_token)
                    skip = True
                else:
                    new_word.append(word[j])
            if not skip:
                new_word.append(word[-1])
            new_corpus.append(new_word)
        corpus = new_corpus
        
        if (i+1) % 50 == 0:
            print(f"   [{i+1}/{num_merges}] Merge: '{best_pair}' -> '{new_token}'")
            
    # 4. Salvar Vocabulário e Merges
    vocab_file = Path("CAFUNE/python/vocab_bpe.json")
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump({"vocab": ids, "merges": [list(m) for m in merges.keys()]}, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ [BPE TOKENIZER] Treinamento Concluido: {len(ids)+5} tokens totais.")
    print(f"   Vocabulario salvo em {vocab_file}")

if __name__ == "__main__":
    train_bpe("CAFUNE/python/social_data.json")
