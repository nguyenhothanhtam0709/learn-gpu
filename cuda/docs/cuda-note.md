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

# Tile programming

Tile programming utilizes GPU Shared Memory to cache data locally within a block, drastically reducing high-latency Global Memory (VRAM) access. In this pattern, all threads in a block cooperatively load a tile (e.g., $16 \times 16$ or $32 \times 32$ elements) into Shared Memory. During the loading phase, because threads access contiguous memory addresses, the hardware performs **Memory Coalescing**, grouping up to 32 requests into a single high-bandwidth 'burst' transfer. Once the tile is loaded, the data is reused multiple times by all threads in the block at near-register speeds.

# Bank Conflicts
