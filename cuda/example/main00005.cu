/**
 * Matrix multiplication on CUDA
 */

#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

__global__ void matrix_multiplication(const float *A, const float *B, float *C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int row_stride = blockDim.y * gridDim.y;
    int col_stride = blockDim.x * gridDim.x;

    for (int r = row; r < M; r += row_stride)
        for (int c = col; c < K; c += col_stride)
        {
            float result = 0.0f;
            for (int k = 0; k < N; ++k)
                // Access A[row][k] and B[k][col]
                result += A[r * N + k] * B[k * K + c];

            C[r * K + c] = result;
        }
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