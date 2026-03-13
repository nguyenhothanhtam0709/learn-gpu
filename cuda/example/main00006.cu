/**
 * Tiling matrix multiplication on CUDA
 */

/**
 * NOTE:
 * - Cooperative Loading
 * - Compute
 * - Sliding Window
 *
 * Apply GPU's Memory Coalescing
 */

#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matrix_multiplication(const float *__restrict__ A,
                                      const float *__restrict__ B,
                                      float *__restrict__ C,
                                      int M, int N, int K)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t)
    {

        if (row < M && (t * TILE_SIZE + threadIdx.x) < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < K && (t * TILE_SIZE + threadIdx.y) < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * K + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < K)
        C[row * K + col] = sum;
}

void solve(const float *A, // M x N matrix
           const float *B, // N x K matrix
           float *C,       // M x K matrix
           int M, int N, int K)
{
    // NOTE: dim3(xDim, yDim, zDim)
    dim3 threadsPerBlock(16, 16); // 256 threads
    dim3 blocksPerGrid(/* Width: */ (K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       /* Height: */ (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);

    cudaDeviceSynchronize();
}

int main()
{
    int M = 1000, N = 100, K = 2000;
    int N_A = M * N,
        N_B = N * K,
        N_C = M * K;
    float *A, *B, *C;

    // Allocate Unified Memory -accessible from CPU or GPU
    cudaMallocManaged(&A, N_A * sizeof(float));
    cudaMallocManaged(&B, N_B * sizeof(float));
    cudaMallocManaged(&C, N_C * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N_A; i++)
        A[i] = 1.0f;
    for (int i = 0; i < N_B; i++)
        B[i] = 2.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    solve(A, B, C, M, N, K);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Time: " << milliseconds << " ms" << std::endl;

    // Free memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}