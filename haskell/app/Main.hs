module Main where

import Scheduler
import Orchestrator
import System.IO

-- | Loop de decisao do cerebro
brainStep :: DiffusionState -> IO ()
brainStep state
    | step state >= totalSteps state = 
        putStrLn "\n[Brain] Geracao completa. Pensamento formado."
    | otherwise = do
        let nextState = updateState state
        -- Cérebro envia o sinal nervoso para o corpo
        dispatchInference (step nextState) (maskRatio nextState)
        brainStep nextState

main :: IO ()
main = do
    -- Forçar output sem buffer para ver o log em tempo real
    hSetBuffering stdout NoBuffering
    putStrLn "=== CAFUNE: Sistema Nervoso Central (Haskell) ==="
    putStrLn "[Brain] Iniciando processo de geracao com 10 passos..."
    
    let initialBrain = initialState 10 -- 10 passos de difusao
    brainStep initialBrain
