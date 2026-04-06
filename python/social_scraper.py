import requests
import json
import os
import time

def scrape_social_context():
    # 1. Alvos: Dialogos que exigem Teoria da Mente (ToM) e Empatia
    # Simulando coleta de datasets publicos de Redes Sociais/Foruns
    print("\n=== [CAFUNE: SOCIAL INGESTION ACTIVE] ===")
    print("Coletando dialogos ricos em contexto social (mimese de perspectiva)...")
    
    # Exemplo: Coletando de um repositório de diálogos (Ex: Emotional Context Dataset)
    dataset_url = "https://raw.githubusercontent.com/datasets/dialogues-example/main/context.json"
    
    try:
        # response = requests.get(dataset_url)
        # data = response.json()
        
        # Fallback Local / Dados Sinteticos de Treino (Conforme Passo 1 d Strategy)
        social_data = [
            {"context": "Empatia", "text": "Entendo como voce se sente, mas precisamos ver o outro lado."},
            {"context": "Ironia", "text": "Uau, que ideia brilhante, voce realmente pensou muito nisso (sarcasmo)."},
            {"context": "Negociacao", "text": "Eu fecho o trato, mas preciso de uma garantia de retorno."},
        ]
        
        output_file = "social_train_data.jsonl"
        with open(output_file, "a") as f:
            for item in social_data:
                f.write(json.dumps(item) + "\n")
        
        print(f" [✓] {len(social_data)} Amostras de Contexto Social Ingeridas em {output_file}")
    except Exception as e:
        print(f" Erro na ingestao: {e}")

if __name__ == "__main__":
    scrape_social_context()
