module Orchestrator (dispatchInference) where

import System.Process
import System.Exit
import Text.Printf
import Scheduler (ScheduleStrategy(..))
import System.IO
import Foreign.Marshal.Alloc
import Foreign.Storable
import Foreign.Ptr
import Control.Concurrent (threadDelay)
import qualified Data.ByteString as B
import qualified Data.ByteString.Internal as BI

-- | Dispara a inferencia e retorna (Entropia, Recompensa)
-- ATENCAO: O lado Python usa filelock (cafune_brain.mem.lock) para exclusao mutua.
-- Este lado Haskell nao adquire o mesmo lock — assuma acesso exclusivo por design
-- (apenas um processo Haskell deve rodar por vez).
dispatchInference :: Int -> Double -> ScheduleStrategy -> IO (Double, Double)
dispatchInference t ratio strat = do
    putStrLn $ printf "🧠 [Orchestrator] Enviando Pulso [%s] | Passo %d | Ratio %.2f" 
                     (show strat) t ratio
    
    let memFile = "cafune_brain.mem"
    
    withBinaryFile memFile ReadWriteMode $ \h -> do
        hSeek h AbsoluteSeek 0
        hPutChar h '\x01'
        hSeek h AbsoluteSeek 4
        B.hPut h (B.pack [fromIntegral t, 0, 0, 0])
        hSeek h AbsoluteSeek 8
        B.hPut h (B.pack [round (ratio * 100), 0, 0, 0, 0, 0, 0, 0])
        hFlush h

    -- Aguarda o Sentinela processar (espera flag 2: DONE)
    waitForDone memFile 0
    
    -- LÊ FEEDBACKS (Entropia no 32, Recompensa no 40)
    withBinaryFile memFile ReadMode $ \h -> do
        hSeek h AbsoluteSeek 32
        eChar <- hGetChar h
        let entropy = fromIntegral (fromEnum eChar) / 100.0
        
        hSeek h AbsoluteSeek 40
        rChar <- hGetChar h
        let reward = fromIntegral (fromEnum rChar) / 100.0
        
        return (entropy, reward)

-- | Loop de espera ativa (Polling) para resposta da Bridge
waitForDone :: FilePath -> Int -> IO ()
waitForDone path count
    | count > 500 = putStrLn "⚠️  Timeout: O Sentinela Python nao respondeu."
    | otherwise = do
        status <- withBinaryFile path ReadMode $ \h -> do
            hSeek h AbsoluteSeek 0
            hGetChar h
        
        case status of
            '\x02' -> return () -- DONE!
            '\x03' -> putStrLn "❌ Erro no processamento da Bridge."
            _      -> do
                threadDelay 10000 -- 10ms
                waitForDone path (count + 1)
