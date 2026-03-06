/**
 * Count array elements
 */

/**
 * NOTE: Apply block level shared memory
 */

#include <iostream>
#include <algorithm>
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

#define THREADS_PER_BLOCK 256
#define THREADS_PER_WARP 32
#define WARPS_PER_BLOCK (THREADS_PER_BLOCK / THREADS_PER_WARP)

// Warp-level reduction (warp shuffle reduction)
__device__ __forceinline__ int warp_reduce_sum(int val)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
#ifdef __HIP_PLATFORM_AMD__
        val += __shfl_down(val, offset); // HIP
#else
        /**
         * NOTE:
         * - Each iteration, thread[i] reads the register of thread[i + offset].
         *   e.g. offset=16: thread 0 reads thread 16, thread 1 reads thread 17, ...
         *   After 5 iterations (16->8->4->2->1), thread 0 holds the sum of all 32 threads.
         * - `0xffffffff` is the participation mask: each bit corresponds to one thread in the warp.
         *   0xffffffff = all 32 bits set -> all 32 threads must participate in this instruction.
         */
        val += __shfl_down_sync(0xffffffff, val, offset); // CUDA
#endif
    }

    return val;
}

__global__ void count_equal_kernel(const int4 *__restrict__ input,
                                   int *__restrict__ output,
                                   const int N,
                                   const int K)
{
    __shared__ int smem[WARPS_PER_BLOCK]; // 1 slot per warp (max 32 warps/block)

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int lane = threadIdx.x & 31;    // bitwise AND to `0b011111` to keep only 5 lowest bit -> current position of thread in warp
    int warp_id = threadIdx.x >> 5; // Right shift 5 bit, equivalent to `threadIdx.x / 32` -> warp id in block
    int N4 = N / 4;

    // Grid stride loop
    int localCount = 0;
    for (int i = tid; i < N4; i += stride)
    {
        // vectorized load (int4)
        int4 v = input[i];
        localCount += (v.x == K);
        localCount += (v.y == K);
        localCount += (v.z == K);
        localCount += (v.w == K);
    }
    // Handle last `N % 4` elements
    const int *arr = reinterpret_cast<const int *>(input);
    for (int i = N4 * 4 + tid; i < N; i += blockDim.x * gridDim.x)
        localCount += (arr[i] == K) ? 1 : 0;

    // warp reduce (register only, don't need smem)
    localCount = warp_reduce_sum(localCount);

    // lane 0 of each warp
    if (0 == lane)
        smem[warp_id] = localCount;
    __syncthreads();

    if (warp_id == 0)
    {
        // First warp reduce smem
        int warps_in_block = blockDim.x >> 5; // Total warps in block, equivalent to `blockDim.x / 32`
        localCount = (threadIdx.x < warps_in_block) ? smem[lane] : 0;
        localCount = warp_reduce_sum(localCount);

        if (threadIdx.x == 0)
            atomicAdd(output, localCount);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int *input, int *output, int N, int K)
{
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // int numSMs;
    // cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    // int blocksPerGrid = std::min((N + threadsPerBlock - 1) / threadsPerBlock, 32 * numSMs);

    count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(reinterpret_cast<const int4 *>(input), output, N, K);
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
    cudaMemset(count, 0, sizeof(int));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

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