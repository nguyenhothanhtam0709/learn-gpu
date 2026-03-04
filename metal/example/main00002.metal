/**
 * Matrix multiplication
 */

#include <metal_stdlib>

kernel void matrix_multiplication(
    const device float *inA [[buffer(0)]],
    const device float *inB [[buffer(1)]],
    device float *result [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    metal::uint2 id [[thread_position_in_grid]])
{
    if (
        (id.y < M)    // C rows
        && (id.x < K) // C cols
    )
    {
        float sum = 0.0f;
        for (uint i = 0; i < N; ++i)
            sum += inA[id.y * N + i] * inB[i * K + id.x];

        result[id.y * K + id.x] = sum;
    }
}