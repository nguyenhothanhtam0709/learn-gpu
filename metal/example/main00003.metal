/**
 * ReLU
 */

#include <metal_stdlib>

kernel void relu(
    const device float *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    metal::uint id [[thread_position_in_grid]])
{
    if (id < N)
        out[id] = metal::fmax(in[id], 0.0f);
}