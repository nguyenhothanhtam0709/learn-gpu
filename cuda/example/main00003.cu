/**
 * Elemental-wise add on CUDA
 */

#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

// Kernel function to add the elements of two arrays
__global__ void vector_add(const float *A, const float *B, float *C, int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid

    // grid-stride loop
    for (int i = index; i < N; i += stride)
        C[i] = A[i] + B[i];
}

void solve(const float *A, const float *B, float *C, int N)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

int main()
{
    int N = 1 << 20;
    float *A, *B, *C;

    // Allocate Unified Memory -accessible from CPU or GPU
    cudaMallocManaged(&A, N * sizeof(float));
    cudaMallocManaged(&B, N * sizeof(float));
    cudaMallocManaged(&C, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++)
    {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    solve(A, B, C, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Time: " << milliseconds << " ms" << std::endl;

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        maxError = fmax(maxError, fabs(C[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}