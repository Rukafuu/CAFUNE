module Scheduler (
    DiffusionState(..),
    initialState,
    updateState
) where

-- | Estado da difusão: passo atual, total de passos e razão de máscara
data DiffusionState = DiffusionState {
    step       :: Int,
    totalSteps :: Int,
    maskRatio  :: Double
} deriving (Show)

-- | Inicializa com 100% de máscara (ruído total)
initialState :: Int -> DiffusionState
initialState total = DiffusionState 0 total 1.0

-- | Calcula o próximo estado (Decaimento Linear como no LLaDA)
updateState :: DiffusionState -> DiffusionState
updateState (DiffusionState s total _) =
    let nextS = s + 1
        -- A razão de máscara cai de 1.0 para 0.0
        newRatio = max 0.0 (1.0 - (fromIntegral (s + 1) / fromIntegral total))
    in DiffusionState nextS total newRatio
