#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {

/**
 * Kernel de Atencao (Capa Simples):
 * Calcula o produto escalar escalonado entre Query (Q) e Key (K).
 * Q: [seq_len, d_k]
 * K: [seq_len, d_k]
 * Scores: [seq_len, seq_len]
 */
__global__ void attention_score_kernel(float* Q, float* K, float* scores, int seq_len, int d_k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < seq_len && col < seq_len) {
        float sum = 0.0f;
        for (int i = 0; i < d_k; i++) {
            // Acesso as matrizes Q e K (Assumindo Row-Major)
            sum += Q[row * d_k + i] * K[col * d_k + i];
        }
        
        // Fator de escala (opcional: 1.0f / sqrtf(d_k))
        scores[row * seq_len + col] = sum;
    }
}

/**
 * Disparador (Launcher) do kernel para ser chamado pelo Haskell ou Julia.
 * Este wrapper lida com a configuração de blocos e threads da GPU.
 */
__declspec(dllexport) void launch_attention(float* d_Q, float* d_K, float* d_scores, int seq_len, int d_k) {
    // Definimos um grid de 16x16 threads
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((seq_len + 15) / 16, (seq_len + 15) / 16);
    
    // Dispara o kernel na GPU
    attention_score_kernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_K, d_scores, seq_len, d_k);
    
    // Sincroniza o processamento
    cudaDeviceSynchronize();
}

}
