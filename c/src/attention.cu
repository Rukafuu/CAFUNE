#include <cuda_runtime.h>
#include <math.h>

extern "C" {

#define TILE_SIZE 16

/**
 * Kernel de Flash-Style Tiled Attention:
 * Calcula Atenção completa em um único kernel fundido (Fused).
 * 
 * Q, K, V: [seq_len, d_model]
 * Output:  [seq_len, d_model]
 */
__global__ void flash_attention_kernel(
    float* Q, float* K, float* V, float* O, 
    int seq_len, int d_model, float scale
) {
    int q_idx = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (q_idx >= seq_len) return;

    // Buffers locais para Online Softmax
    float row_max = -1e20f;
    float row_sum = 0.0f;
    float res[TILE_SIZE] = {0.0f}; // Acumulador local para o output

    // Loop sobre os blocos (tiles) das chaves (K) e valores (V)
    for (int tile = 0; tile < (seq_len + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        __shared__ float sK[TILE_SIZE][TILE_SIZE];
        __shared__ float sV[TILE_SIZE][TILE_SIZE];

        // Carregamento cooperativo para a Shared Memory
        int k_idx = tile * TILE_SIZE + threadIdx.x;
        if (k_idx < seq_len) {
            for (int i = 0; i < TILE_SIZE; i++) {
                if (q_idx < seq_len) {
                    sK[threadIdx.x][i] = K[k_idx * d_model + i];
                    sV[threadIdx.x][i] = V[k_idx * d_model + i];
                }
            }
        }
        __syncthreads();

        // Cálculo de Atenção no Bloco (Online Softmax logic)
        for (int j = 0; j < TILE_SIZE; j++) {
            float score = 0.0f;
            for (int k = 0; k < d_model; k++) {
                score += Q[q_idx * d_model + k] * K[(tile * TILE_SIZE + j) * d_model + k];
            }
            score *= scale;

            float old_max = row_max;
            row_max = fmaxf(row_max, score);
            float exp_val = expf(score - row_max);
            
            // Re-escala o acumulador anterior para o novo máximo
            float scale_factor = expf(old_max - row_max);
            row_sum = row_sum * scale_factor + exp_val;

            for (int i = 0; i < d_model; i++) {
                // Simplificado: Aqui multiplicaríamos pelo V
                O[q_idx * d_model + i] = (O[q_idx * d_model + i] * scale_factor) + (exp_val * V[(tile * TILE_SIZE + j) * d_model + i]);
            }
        }
        __syncthreads();
    }

    // Normalização final
    for (int i = 0; i < d_model; i++) {
        O[q_idx * d_model + i] /= row_sum;
    }
}

__declspec(dllexport) void launch_flash_attention(
    float* d_Q, float* d_K, float* d_V, float* d_O, 
    int seq_len, int d_model
) {
    float scale = 1.0f / sqrtf((float)d_model);
    int threads = TILE_SIZE;
    int blocks = (seq_len + TILE_SIZE - 1) / TILE_SIZE;
    
    flash_attention_kernel<<<blocks, threads>>>(d_Q, d_K, d_V, d_O, seq_len, d_model, scale);
    cudaDeviceSynchronize();
}

}
