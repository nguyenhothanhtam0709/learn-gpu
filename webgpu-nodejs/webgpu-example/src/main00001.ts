/**
 * Elemental-wise add on WebGPU
 */

import { create as createGPU, globals } from "webgpu";

//#region Init WebGPU for Node.js
Object.assign(globalThis, globals);
Object.assign(globalThis.navigator, {
  gpu: createGPU([]),
});
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
  //#endregion

  //#region Create buffer
  const N = 10_000;
  const arrA = new Float32Array(N).map((_, i) => i);
  const arrB = new Float32Array(N).map((_, i) => i * 2);

  const bufferA = createBuffer(gpuDevice, arrA, GPUBufferUsage.STORAGE);
  const bufferB = createBuffer(gpuDevice, arrB, GPUBufferUsage.STORAGE);
  const bufferC = gpuDevice.createBuffer({
    size: arrA.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const bufferN = gpuDevice.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  gpuDevice.queue.writeBuffer(bufferN, 0, new Uint32Array([N]));
  //#endregion

  const shader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read> B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;
@group(0) @binding(3) var<uniform> N: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  let i = id.x;
  if (i >= N) {return;}

  C[i] = A[i] + B[i];
}
`;
  const module = gpuDevice.createShaderModule({ code: shader });
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
      { binding: 3, resource: { buffer: bufferN } },
    ],
  });

  //#region Dispatch compute
  const encoder = gpuDevice.createCommandEncoder();
  const pass = encoder.beginComputePass();

  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);

  const workgroupSize = 64;
  const numWorkgroups = Math.ceil(N / workgroupSize);

  pass.dispatchWorkgroups(numWorkgroups);
  pass.end();

  gpuDevice.queue.submit([encoder.finish()]);
  await gpuDevice.queue.onSubmittedWorkDone();
  //#endregion

  //#region Get result back
  const readBuffer = gpuDevice.createBuffer({
    size: arrA.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const copyEncoder = gpuDevice.createCommandEncoder();
  copyEncoder.copyBufferToBuffer(bufferC, 0, readBuffer, 0, arrA.byteLength);
  gpuDevice.queue.submit([copyEncoder.finish()]);
  await gpuDevice.queue.onSubmittedWorkDone();

  await readBuffer.mapAsync(GPUMapMode.READ);
  const result = Float32Array.from(
    new Float32Array(readBuffer.getMappedRange()),
  );
  readBuffer.unmap();
  //#endregion

  console.log(result);
}

void main().finally(() => {
  // @ts-ignore
  delete globalThis.navigator.gpu;
});
//#endregion
