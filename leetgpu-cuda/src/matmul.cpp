#include <cuda_runtime.h>

/// @note Disadvantage:
/// - No Global memory access in inner loop
///     Every thread loads N elements from A and B
/// - No shared memory
/// - No tiling
/// - No coalescing optimization
/// not OK for real workloads.
__global__ void matrix_multiplication_kernel_1(const float *A, const float *B, float *C, int M, int N, int K)
{
    // Calculate global row (row) and column (col) for the output element C[row][col]
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K)
    {
        float result = 0.0f;
        for (int k = 0; k < N; ++k)
            // Access A[row][k] and B[k][col]
            result += A[row * N + k] * B[k * K + col];

        C[row * K + col] = result;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int M, int N, int K)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,  // col
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y); // row

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}