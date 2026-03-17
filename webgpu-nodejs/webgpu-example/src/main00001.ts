/**
 * Elemental-wise add on WebGPU
 */

import { create as createGPU, globals } from "webgpu";

//#region Init WebGPU for Node.js
Object.assign(globalThis, globals);
Object.assign(globalThis.navigator, {
  gpu: createGPU([]),
});

const THREAD_WORKGROUP_SIZE = 64;
//#endregion

function createBuffer(
  device: GPUDevice,
  arr: Float32Array,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  const buffer = device.createBuffer({
    size: arr.byteLength,
    usage,
    mappedAtCreation: true,
  });
  new Float32Array(buffer.getMappedRange()).set(arr);
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
  const N = 10_000;
  const bufferSizeC = N * Float32Array.BYTES_PER_ELEMENT;

  const arrA = new Float32Array(N).map((_, i) => i);
  const arrB = new Float32Array(N).map((_, i) => i * 2);

  const bufferA = createBuffer(gpuDevice, arrA, GPUBufferUsage.STORAGE);
  const bufferB = createBuffer(gpuDevice, arrB, GPUBufferUsage.STORAGE);
  const bufferC = gpuDevice.createBuffer({
    size: bufferSizeC,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const outputBuffer = gpuDevice.createBuffer({
    size: bufferSizeC,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  //#endregion

  try {
    const shader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read> B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;

@compute @workgroup_size(${THREAD_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&A)) {return;}

  C[i] = A[i] + B[i];
}
`;
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

    const numWorkgroups = Math.ceil(N / THREAD_WORKGROUP_SIZE);

    pass.dispatchWorkgroups(numWorkgroups);
    pass.end();
    //#endregion

    //#region Get result back
    encoder.copyBufferToBuffer(bufferC, 0, outputBuffer, 0, bufferSizeC);
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

    console.log(output);
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
