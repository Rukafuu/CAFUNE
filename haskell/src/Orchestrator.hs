module Orchestrator (dispatchInference) where

import System.Process
import System.Exit
import Text.Printf

-- | Dispara a inferencia na "Mao" (Python/Julia)
dispatchInference :: Int -> Double -> IO ()
dispatchInference t ratio = do
    putStrLn $ printf "[Orchestrator] Comando para o passo %d (Mascara: %.1f%%)" t (ratio * 100)
    
    -- Ajuste o caminho se necessario (relativo a raiz do projeto Lira)
    (_, _, _, ph) <- createProcess (proc "python" [
        "CAFUNE/python/bridge.py", 
        "--step", show t, 
        "--ratio", show ratio
        ])
    
    exitCode <- waitForProcess ph
    case exitCode of
        ExitSuccess   -> return ()
        ExitFailure n -> putStrLn $ "!!! Erro na Bridge: " ++ show n
