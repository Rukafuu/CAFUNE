module Main where

import Scheduler
import Orchestrator
import System.IO

-- | Loop principal de difusao adaptativa com RLAIF
diffusionLoop :: DiffusionState -> IO ()
diffusionLoop state
    | step state >= totalSteps state = putStrLn "🏁 Difusão Concluída! O CAFUNE atingiu o Alinhamento Neural."
    | otherwise = do
        -- 1. Dispara Inferência e recebe feedback duplo (Incerteza + Recompensa RLAIF)
        (entropy, reward) <- dispatchInference (step state) (maskRatio state) (strategy state)
        
        let conf = max 0.0 (1.0 - (entropy / 4.0))
            
        -- 2. Log de Performance
        putStrLn $ printf "[Feedback] Confiança: %.2f | Reward RLAIF: %.2f" conf reward
        
        -- 3. Lógica de Reforço: Se reward for muito baixo, REPETIMOS o passo (Auto-Ajuste)
        let nextState = if reward < 0.3 && step state > 0
                        then state { confidence = conf * 0.5 } -- Penaliza e repete
                        else (updateState state) { confidence = conf }
            
        diffusionLoop nextState

main :: IO ()
main = do
    -- Forçar output sem buffer para ver o log visceral em tempo real
    hSetBuffering stdout NoBuffering
    putStrLn "=== CAFUNE: Orquestração Central (Phase 3: Nervous System) ==="
    
    -- Configuração: 20 passos com estratégia COSINE (Melhor para convergência textual)
    let totalPassos = 20
        estrategia  = Cosine
        initialBrain = initialState totalPassos estrategia
        
    putStrLn $ "[Brain] Iniciando difusão estratégica [" ++ show estrategia ++ "]..."
    brainStep initialBrain
