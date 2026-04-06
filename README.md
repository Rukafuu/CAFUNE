<p align="center">
  <img src="assets/logo.png" width="400" alt="CAFUNE Logo">
</p>

# 🧟 CAFUNE: Neural Engine de Difusão Adaptativa (45.1M)

<p align="center">
  <img src="https://img.shields.io/badge/Haskell-5e5086?style=for-the-badge&logo=haskell&logoColor=white" alt="Haskell">
  <img src="https://img.shields.io/badge/Julia-9558b2?style=for-the-badge&logo=julia&logoColor=white" alt="Julia">
  <img src="https://img.shields.io/badge/Python-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-76b900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA">
</p>

> _"Não estava eu formando um ser horrível e transgressor?"_ — Mary Shelley

**CAFUNE** (**C**omposite **A**rchitecture for **F**ast **U**niversal **N**oise-reduction **E**ngine) é um motor de difusão híbrido de elite, projetado para simular o processamento cognitivo humano através de uma arquitetura heterogênea de alto desempenho. O sistema utiliza um loop de feedback adaptativo entre **Haskell** (Orquestração), **Julia** (Inferência) e **CUDA** (Aceleração).

---

## 🧬 ARQUITETURA DO SISTEMA (Frankenstein Flow)

O diagrama abaixo ilustra a comunicação de latência zero entre as camadas do ecossistema via **Shared Memory MMAP**.

```mermaid
graph TD
    subgraph "Nervos (Haskell)"
        A[Orquestrador Adaptativo] -->|Sched/Entropy| B(RLAIF Loop)
    end
    
    subgraph "Pele (Python)"
        B -->|IPC Shared Memory| C[Bridge MMAP]
        C -->|Sinal de Recompensa| D{AI Critic / Raegis}
    end
    
    subgraph "Músculos (Julia)"
        C -->|Zero-Copy| E[Transformer Bidirecional]
        E -->|Autodiff Zygote| F[Online Fine-Tuning]
    end
    
    subgraph "Esqueleto (C/CUDA)"
        F -->|Fused Kernel| G[Flash Attention v2]
        G -->|VRAM| H((GPU Speed))
    end
```

---

## 🧠 FUNDAMENTAÇÃO EM NEUROCIÊNCIA COGNITIVA

Diferente de modelos autoregressivos tradicionais, o CAFUNE é inspirado no **Sistema de Neurônios Espelho (MNS)** e na **Codificação Preditiva**.

*   **Ressonância Funcional**: O motor transforma informações textuais em representações internas de intenção, permitindo que a IA compreenda a ação "por dentro".
*   **Teoria da Mente (ToM)**: Arquitetura otimizada para ativar circuitos funcionais em camadas superiores para detecção de estados mentais e intenções complexas.
*   **Mirror Neuron Index (CMNI)**: Métrica implementada para quantificar a capacidade de espelhamento do modelo.

O **Mirror Neuron Score (MNS)** de cada neurônio é calculado como:

$$MNS_n = \frac{\Delta \mu_n(D_f) + \Delta \mu_n(D_t)}{2}$$

---

## 🚀 ESPECIFICAÇÕES CAFUNE v2.5 (2026)

A versão **45.1M** (codinome: *Lira-Mega-Boost*) representa o auge da arquitetura de difusão social.

1.  **Net2Net Expansion**: Expansão vertical de 6 para 12 camadas, preservando o conhecimento prévio da v1.0 e permitindo novas sinapses via ruído epsilon controlado ($10^{-5}$).
2.  **Aprendizado Autônomo (RLAIF)**: Loop de feedback com o **Gemini 2.5 Pro Pro**, avaliando coerência, intenção e empatia a cada pulso de ressonância.
3.  **Dicionário Social 🧿**: Vocabulário BPE de **1949 tokens** especializado em diálogos humanos reais.
4.  **Zero-Copy Architecture**: Fluxo de dados entre linguagens via **MMAP (1024 bytes)** sem overhead de serialização.

---

## 📊 MÉTRICAS TÉCNICAS (Elite)

| Métrica | Valor | Status |
| :--- | :--- | :--- |
| **Parâmetros** | 45,166,412 (45.1M) | ✅ GIGANTE |
| **Camadas** | 12 (Transformer Bidirecional) | ✅ EXPANDIDO |
| **Vocabulário** | 1949 Tokens BPE Sociais | 🧿 ALPHA SOCIAL |
| **Latência** | ~14ms (CUDA direct) | 🚀 OTIMIZADO |

---

## 🛡️ RAEGIS: MONITORAMENTO ÉTICO MECANÍSTICO

Integração com o sistema **Raegis v2.5.1** para mitigação de vícios algorítmicos e auditoria de intenção:
*   **Anti-Sicofancia**: Filtra a tendência do modelo de adulação ao usuário.
*   **Maturity Index**: Monitora a entropia da resposta para prevenir colapso e repetitividade.
*   **Gold Samples**: Experiências com score > 0.8 são salvas no `experience_buffer` para evolução contínua.

---

## 🛠️ COMO EXECUTAR O ORGANISMO

1.  **Gere a Visão**: `python python/train_bpe.py`
2.  **Dashboard**: `python python/dashboard.py` (Visualização Web)
3.  **Treinamento/Inca**: `julia --project=. main_training.jl` dentro de `julia/`.
4.  **Haskell**: `./gradlew run` dentro de `haskell/` (Orquestrador).

---

**VEJA O CAFUNE EM AÇÃO NO ECOSSISTEMA LIRA**  
[Acesse o Landing Page AAA](file:///C:/Users/conta/Documents/Lira/Lira/landing-page-cafune/index.html)  

---
*Powered by Lira Ecosystem & Antigravity Silicon.*
