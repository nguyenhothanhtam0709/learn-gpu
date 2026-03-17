/**
 * Matrix multiplication on WebGPU: C = A × B
 * Uses tiled shared-memory algorithm for optimal performance.
 *
 *   A: Float32 [M × K]
 *   B: Float32 [K × N]
 *   C: Float32 [M × N]  (output)
 *
 * Buffer layout: [rows: u32, cols: u32, ...data: f32[]]
 */

import { create as createGPU, globals } from "webgpu";

//#region Init WebGPU for Node.js
Object.assign(globalThis, globals);
Object.assign(globalThis.navigator, {
  gpu: createGPU([]),
});

/** @description Tile dimension. Each workgroup is TILE x TILE threads. */
const TILE_SIZE = 16;
/**
 * @description Byte offset where the float data starts inside a Matrix buffer.
 * @note [rows, cols]
 */
const MATRIX_HEADER = 2 * Uint32Array.BYTES_PER_ELEMENT;
//#endregion

/**
 * @description Allocate a GPU buffer with the Matrix layout:
 *      [rows: u32, cols: u32, ...data: f32[]]
 */
function createBuffer(
  device: GPUDevice,
  rows: number,
  cols: number,
  data: Float32Array,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  const buffer = device.createBuffer({
    size: MATRIX_HEADER + data.byteLength, // @note: for layout of Matrix struct - [rows: u32, cols: u32, ...data: f32[]]
    usage,
    mappedAtCreation: true,
  });

  const mapped = buffer.getMappedRange();
  new Uint32Array(mapped, 0, 2 /** 2 elements */).set([rows, cols]);
  new Float32Array(
    mapped,
    MATRIX_HEADER /** @note bytelength of the first 2 elements */,
  ).set(data);
  buffer.unmap();

  return buffer;
}

//#region Entrypoint
async function main(): Promise<void> {
  //#region Init WebGPU device
  const gpuAdapter = await navigator.gpu.requestAdapter();
  if (!gpuAdapter) throw new Error("GPU adapter is not available!");

  const gpuDevice = await gpuAdapter.requestDevice();
  if (!gpuDevice) throw new Error("GPU device is not available!");

  const gpuCmdQueue = gpuDevice.queue;
  //#endregion

  //#region Create buffer
  const M = 512;
  const K = 256;
  const N = 512;

  const dataA = new Float32Array(M * K).map((_, i) => i / (M * K));
  const dataB = new Float32Array(K * N).map((_, i) => (2 * i) / (K * N));

  const bufferA = createBuffer(gpuDevice, M, K, dataA, GPUBufferUsage.STORAGE);
  const bufferB = createBuffer(gpuDevice, K, N, dataB, GPUBufferUsage.STORAGE);

  const bufferSizeC = M * N * Float32Array.BYTES_PER_ELEMENT;
  // C only needs the header pre-set; data is written by the shader.
  // WebGPU zero-initialises buffers by default so header = [0, 0] initially.
  // We don't read C.rows/C.cols inside the shader, so this is fine.
  const bufferC = gpuDevice.createBuffer({
    size: MATRIX_HEADER + bufferSizeC,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const outputBuffer = gpuDevice.createBuffer({
    size: bufferSizeC,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  //#endregion

  try {
    //#region Shader
    const shader = /* wgsl */ `
struct Matrix {
    rows: u32,
    cols: u32,
    data: array<f32>, // runtime-sized; must be last member
}

@group(0) @binding(0) var<storage, read>            A : Matrix;
@group(0) @binding(1) var<storage, read>            B : Matrix;
@group(0) @binding(2) var<storage, read_write>      C : Matrix;

// Shared tiles - one per axis, loaded cooperatively by all threads in the workgroup
var<workgroup> tileA : array<array<f32, ${TILE_SIZE}>, ${TILE_SIZE}>;
var<workgroup> tileB : array<array<f32, ${TILE_SIZE}>, ${TILE_SIZE}>;

@compute @workgroup_size(${TILE_SIZE}, ${TILE_SIZE})
fn main(
  @builtin(global_invocation_id) globalId : vec3<u32>,
  @builtin(local_invocation_id)  localId  : vec3<u32>,
) {
  let row = globalId.y;     // which row of C this thread owns
  let col = globalId.x;     // which col of C this thread owns
  let lRow = localId.y;     // local row within tile
  let lCol = localId.x;    // local col within tile

  let M = A.rows;
  let K = A.cols; // == B.rows
  let N = B.cols;

  var sum: f32 = 0.0;
  let numTiles = (K + ${TILE_SIZE}u - 1u) / ${TILE_SIZE}u;

  for (var t : u32 = 0u; t < numTiles; t++) {
    // --- Load tile of A (row-major) ---
    let aCol = t * ${TILE_SIZE}u + lCol;
    if (row < M && aCol < K) {
      tileA[lRow][lCol] = A.data[row * K + aCol];
    } else {
      tileA[lRow][lCol] = 0.0; // zero-pad for non-tile-aligned dimensions
    }
    
    // --- Load tile of B (row-major) ---
    let bRow = t * ${TILE_SIZE}u + lRow;
    if (bRow < K && col < N) {
      tileB[lRow][lCol] = B.data[bRow * N + col];
    } else {
      tileB[lRow][lCol] = 0.0;
    }

    workgroupBarrier();  // all threads must finish loading before any thread reads

    // --- Accumulate partial dot product from this tile ---
    for (var k : u32 = 0u; k < ${TILE_SIZE}u; k++) {
      sum += tileA[lRow][k] * tileB[k][lCol];
    }

    workgroupBarrier();  // all threads must finish reading before next tile overwrites
  }

  if (row < M && col < N) {
    C.data[row * N + col] = sum;
  }
}
`;
    //#endregion

    //#region Compile shader
    const module = gpuDevice.createShaderModule({ code: shader });
    //#region Handle compile error
    const compilationInfo = await module.getCompilationInfo();
    const errors = compilationInfo.messages.filter((m) => m.type === "error");
    if (errors.length > 0) {
      throw new Error(
        `Shader compilation failed:\n${errors.map((e) => e.message).join("\n")}`,
      );
    }
    //#endregion
    //#endregion

    const pipeline = await gpuDevice.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module,
        entryPoint: "main",
      },
    });
    const bindGroup = gpuDevice.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufferA } },
        { binding: 1, resource: { buffer: bufferB } },
        { binding: 2, resource: { buffer: bufferC } },
      ],
    });

    //#region Dispatch compute
    const encoder = gpuDevice.createCommandEncoder();
    const pass = encoder.beginComputePass();

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(N / TILE_SIZE), // x-axis covers columns of C
      Math.ceil(M / TILE_SIZE), // y-axis covers rows of C
    );
    pass.end();
    //#endregion

    //#region Get result back
    encoder.copyBufferToBuffer(
      bufferC,
      MATRIX_HEADER,
      outputBuffer,
      0,
      bufferSizeC,
    );
    //#endregion

    //#region Submit and wait for result
    gpuCmdQueue.submit([encoder.finish()]);
    //#endregion

    //#region Read result
    await outputBuffer.mapAsync(GPUMapMode.READ);
    const output = Float32Array.from(
      new Float32Array(outputBuffer.getMappedRange()),
    ); // Safe copy
    outputBuffer.unmap();
    //#endregion

    // Spot-check a few values
    console.log(`Result shape : ${M} × ${N}`);
    console.log(`C[0][0]      = ${output[0]?.toFixed(8)}`);
    console.log(`C[0][1]      = ${output[1]?.toFixed(8)}`);
    console.log(`C[1][0]      = ${output[N]?.toFixed(8)}`);
  } finally {
    bufferA.destroy();
    bufferB.destroy();
    bufferC.destroy();
    outputBuffer.destroy();
    gpuDevice.destroy();
  }
}

void main().finally(() => {
  // @ts-ignore
  delete globalThis.navigator.gpu;
});
//#endregion
