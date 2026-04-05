module Scheduler (
    DiffusionState(..),
    ScheduleStrategy(..),
    initialState,
    updateState
) where

-- | Estratégias disponíveis para o decaimento de máscara (Denoising)
data ScheduleStrategy 
    = Linear        -- ^ Queda constante e simples
    | Cosine        -- ^ Queda suave nas pontas, rápida no meio (Recomendado)
    | Sigmoid       -- ^ Acelera a revelação no final da difusão
    deriving (Show, Eq)

-- | Estado da difusão: passo atual, total de passos, razão de máscara e confiança (entropia inversa)
data DiffusionState = DiffusionState {
    step        :: Int,
    totalSteps  :: Int,
    maskRatio   :: Double,
    strategy    :: ScheduleStrategy,
    confidence  :: Double  -- ^ Feedback de incerteza (menor entropia = maior confiança)
} deriving (Show)

-- | Inicializa com 100% de máscara e confiança total (estado inicial)
initialState :: Int -> ScheduleStrategy -> DiffusionState
initialState total strat = DiffusionState 0 total 1.0 strat 1.0

-- | Calcula o próximo estado baseado na estratégia e no feedback de confiança
updateState :: DiffusionState -> DiffusionState
updateState state@(DiffusionState s total ratio strat conf)
    | s >= total = state 
    | otherwise  =
        let nextS = s + 1
            progress = fromIntegral nextS / fromIntegral total
            
            -- Se a confiança for baixa (< 0.5), desaceleramos a abertura da máscara
            adaptiveModifier = if conf < 0.5 then 0.1 else 0.0
            
            baseRatio = calculateRatio progress strat
            newRatio = max 0.0 (baseRatio + adaptiveModifier)
            
        in DiffusionState nextS total newRatio strat conf

-- | Funções matemáticas de decaimento
calculateRatio :: Double -> ScheduleStrategy -> Double
calculateRatio p Linear = max 0.0 (1.0 - p)
calculateRatio p Cosine = 
    -- Cosine decay: 0.5 * (1 + cos(pi * p))
    let piVal = 3.141592653589793
    in 0.5 * (1.0 + cos(piVal * p))
calculateRatio p Sigmoid =
    -- Sigmoid decay: 1 / (1 + exp(10*(p-0.5)))
    -- Invertido: 1 - sigmoid
    let sigmoid x = 1.0 / (1.0 + exp(10.0 * (x - 0.5)))
    in sigmoid (1.0 - p)
