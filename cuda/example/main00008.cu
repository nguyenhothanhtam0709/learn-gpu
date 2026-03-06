/**
 * Count array elements
 */

/**
 * NOTE: Apply block level shared memory
 */

#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

__global__ void count_equal_kernel(const int *__restrict__ input,
                                   int *__restrict__ output,
                                   int N,
                                   const int K)
{
    __shared__ int blockCount; // Shared within block

    if (0 == threadIdx.x)
        blockCount = 0;
    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid

    int localCount = 0;

    // grid-stride loop
    for (int i = index; i < N; i += stride)
        if (K == input[i])
            localCount++;

    atomicAdd(&blockCount, localCount); // Atomic in shared memory
    __syncthreads();

    if (0 == threadIdx.x && blockCount > 0)
        atomicAdd(output, blockCount);
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int *input, int *output, int N, int K)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
}

int main()
{
    int N = 1'000'000, K = 12;
    int *A;
    int *count;

    // Allocate Unified Memory -accessible from CPU or GPU
    CUDA_CHECK(cudaMallocManaged(&A, N * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&count, sizeof(int)));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++)
        A[i] = i % 121;
    *count = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    solve(A, count, N, K);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Time: " << milliseconds << " ms" << std::endl;

    // Free memory
    cudaFree(A);
    cudaFree(count);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}