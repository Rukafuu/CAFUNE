
<p align="center">
  <img src="assets/logo.png" width="400" alt="CAFUNE Logo">
</p>

# CAFUNE: Neural Engine de Difusão Adaptativa

<p align="center">
  <img src="https://img.shields.io/badge/Haskell-5e5086?style=for-the-badge&logo=haskell&logoColor=white" alt="Haskell">
  <img src="https://img.shields.io/badge/Julia-9558b2?style=for-the-badge&logo=julia&logoColor=white" alt="Julia">
  <img src="https://img.shields.io/badge/Python-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-76b900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA">
</p>

**CAFUNE** (Composite Architecture for Fast Universal Noise-reduction Engine) é um motor de difusão híbrido de elite, projetado para simular o processamento cognitivo humano através de uma arquitetura heterogênea de alto desempenho. O sistema utiliza um loop de feedback adaptativo entre Haskell (Orquestração), Julia (Inferência) e CUDA (Aceleração).

---

## ARQUITETURA DO SISTEMA (Frankenstein Flow)

O diagrama abaixo ilustra a comunicação de latência zero entre as camadas do ecossistema.

```mermaid
graph TD
    subgraph "Cérebro (Haskell)"
        A[Orquestrador Adaptativo] -->|Sched/Entropy| B(RLAIF Loop)
    end
    
    subgraph "Sentinela (Python)"
        B -->|IPC Shared Memory| C[Bridge MMAP]
        C -->|Sinal de Recompensa| D{AI Critic / Raegis}
    end
    
    subgraph "Motor (Julia)"
        C -->|Zero-Copy| E[Transformer Bidirecional]
        E -->|Autodiff Zygote| F[Online Fine-Tuning]
    end
    
    subgraph "Ossos (C/CUDA)"
        F -->|Fused Kernel| G[Flash Attention v2]
        G -->|VRAM| H((GPU Speed))
    end
```

---

## FUNDAMENTAÇÃO EM NEUROCIÊNCIA COGNITIVA

Diferente de modelos autoregressivos tradicionais, o CAFUNE é inspirado no Sistema de Neurônios Espelho (MNS).

*   **Ressonância Funcional**: O motor transforma informações textuais em representações internas de intenção, permitindo que a IA compreenda a ação por dentro.
*   **Codificação Preditiva**: O sistema busca constantemente minimizar o erro de previsão, agindo como um análogo funcional à hierarquia cortical humana.
*   **Teoria da Mente (ToM)**: Arquitetura otimizada para ativar circuitos funcionais em camadas superiores para detecção de estados mentais e intenções complexas.
*   **Mirror Neuron Index (CMNI)**: Métrica implementada para quantificar a capacidade de espelhamento do modelo.

O Mirror Neuron Score (MNS) de cada neurônio é calculado como:

<p align="center">
  <img src="assets/mns_equation.png" width="500" alt="MNS Equation">
</p>

---

## PERFORMANCE DE BAIXO NÍVEL (ZPM)

O CAFUNE segue o Zombie Performance Manifesto (ZPM): cada ciclo de clock é otimizado ao limite.

*   **Shared Memory MMAP**: Eliminação do gargalo de IO. Haskell e Julia operam via memória mapeada, garantindo latência de microssegundos.
*   **Flash Attention v2 Customizado**: Implementação de Tiling para reduzir acessos à memória global da GPU, otimizando o uso da SRAM.
*   **Zero-Copy Architecture**: Fluxo de dados entre linguagens sem overhead de serialização ou cópias desnecessárias.

---

## 📊 MÉTRICAS

| Métrica | Valor | Status |
|--------|------|--------|
| Parâmetros | 72,960 | ✅ |
| Loss | 2.73 ↓ | 📉 |
| Latência | ~12ms | 🚀 |

---

---

## RAEGIS: MONITORAMENTO ÉTICO MECANÍSTICO

Integração com o sistema Raegis para mitigação de vícios algorítmicos:
*   **Anti-Sicofancia**: Filtra a tendência do modelo de validar crenças subjetivas incorretas.
*   **Mimese de Perspectiva**: Previne a criação de câmaras de eco gerativas.

---

## COMO EXECUTAR O ORGANISMO

1. **Dashboard**: `python python/dashboard.py` (Visualização Web)
2. **Ponte de Dados**: `python python/bridge.py --sentinel`
3. **Cérebro Orquestrador**: `./gradlew run` dentro de `haskell/`.

---

**VEJA O CAFUNE EM AÇÃO NO ECOSSISTEMA LIRA**  
[Acesse o Landing Page AAA](file:///C:/Users/conta/Documents/Lira/Lira/landing-page-cafune/index.html)  

---
*Powered by Lira Ecosystem & Antigravity Silicon.*
