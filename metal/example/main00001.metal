#include <metal_stdlib>

kernel void add_arrays(
    const device float *inA [[buffer(0)]],
    const device float *inB [[buffer(1)]],
    device float *result [[buffer(2)]],
    metal::uint id [[thread_position_in_grid]])
{
    result[id] = inA[id] + inB[id];
}