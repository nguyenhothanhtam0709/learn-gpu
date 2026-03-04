# Overview

CUDA GPUs run kernels using blocks of threads that are a multiple of 32 in size.

# Grids, blocks and threads

CUDA GPUs have many parallel processors grouped into Streaming Multiprocessors (SMs). Each SM can run multiple concurrent thread blocks, but each threa block runson a single SM.

<figure>
  <img src="imgs/image-00001.png"width="700">
  <figcaption>Grid, Block and Thread indexing in CUDA kernels (one-dimensional).</figcaption>
</figure>

`gridDim.x` contains the number of blocks in the grid. `blockIdx.x` contains the index of the current thread block in the grid.

The idea is that each thread gets its index by computing the offet to the beginning of its block (`blockIdx.x * blockDim.x`) and adding the thread’s index within the block (`threadIdx.x`).
