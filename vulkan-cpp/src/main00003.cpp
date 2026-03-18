#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cassert>

#include <vulkan/vulkan.h>

/** @note Number of vector elements: 2^20 = 1,048,576 (~1 million) */
constexpr uint32_t N = 1 << 20;
/**
 *  @note Number of threads running in parallel within a single workgroup (must match shader.comp).
 *  Equivalent to "threads per block" in CUDA
 */
constexpr uint32_t LOCAL_SIZE = 256;

/**
 * @note MACRO: Wraps every Vulkan call to automatically throw on failure.
 * Vulkan does not throw exceptions — all calls return a VkResult (VK_SUCCESS = 0).
 * Without checking, errors silently propagate and crash elsewhere, making them
 * very hard to debug.
 */
#define VK_CHECK(call)                                                       \
    do                                                                       \
    {                                                                        \
        VkResult _r = (call);                                                \
        if (_r != VK_SUCCESS)                                                \
        {                                                                    \
            throw std::runtime_error(                                        \
                std::string(#call) + " failed, code=" + std::to_string(_r)); \
        }                                                                    \
    } while (0)

/**
 * @note Read a SPIR-V file (binary) — this is the bytecode the GPU understands.
 * Unlike OpenGL (which accepts GLSL strings at runtime), Vulkan requires
 * compiling GLSL → SPIR-V ahead of time (using glslc), then loading this binary.
 *
 * @note Read as uint32_t because the Vulkan spec requires SPIR-V to be 4-byte aligned.
 */
static std::vector<uint32_t> readSPIRV(const std::string &path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error("Cannot open shader: " + path);
    size_t size = f.tellg();
    f.seekg(0);
    std::vector<uint32_t> buf(size / 4);
    f.read(reinterpret_cast<char *>(buf.data()), size);
    return buf;
}

// MARK: VulkanContext
/**
 * @note VulkanContext: groups all fundamental Vulkan objects into one struct.
 * @note Initialization order matters (must be created in this sequence):
 * Instance → PhysicalDevice → Device → Queue → CommandPool
 */
struct VulkanContext
{
    /** @note the Vulkan application */
    VkInstance instance;
    /** @note the actual GPU hardware */
    VkPhysicalDevice physDevice;
    /** @note a logical connection to the GPU */
    VkDevice device;
    /** @note a submission queue for GPU commands */
    VkQueue computeQueue;
    /** @note index of the queue family that supports compute */
    uint32_t computeFamily;
    /** @note memory allocator for Command Buffers */
    VkCommandPool cmdPool;
};

/**
 * @note Find the index of a Queue Family that supports Compute operations.
 * A GPU exposes multiple "queue families" for different work types (graphics, compute, transfer, etc.).
 * We only need compute since we are not rendering anything.
 * Queue family ≈ "the category of work a queue can perform"
 */
static uint32_t findComputeFamily(VkPhysicalDevice gpu)
{
    uint32_t count = 0;
    // Calling twice is the standard Vulkan pattern:
    //   Call 1: pass nullptr to retrieve the count
    //   Call 2: pass a pointer to fill the data
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, props.data());

    for (uint32_t i = 0; i < count; ++i)
        if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) // check the compute bit flag
            return i;

    throw std::runtime_error("No compute queue family found");
}

VulkanContext createContext()
{
    VulkanContext ctx{};

    // ── 1. CREATE INSTANCE ───────────────────────────────────────────────────
    // Instance = the Vulkan entry point. Must be created before everything else.
    // No extensions or validation layers are enabled here for simplicity.
    VkApplicationInfo appInfo{};
    // ⚠️  sType MUST be set correctly — Vulkan uses this field to identify the struct.
    //     This is a mandatory pattern for EVERY Vulkan struct that has an sType field.
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instCI{};
    instCI.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instCI.pApplicationInfo = &appInfo;
    VK_CHECK(vkCreateInstance(&instCI, nullptr, &ctx.instance)); // The second nullptr = custom allocator (not used → nullptr)

    // ── 2. SELECT PHYSICAL DEVICE (GPU) ─────────────────────────────────────
    // ⚠️  The two-step pattern is mandatory: first get the count, then get the data.
    //     Hard-coding count=1 causes VK_INCOMPLETE when multiple GPUs are present.
    uint32_t deviceCount = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(ctx.instance, &deviceCount, nullptr));
    if (deviceCount == 0)
        throw std::runtime_error("No Vulkan-capable device found");

    std::vector<VkPhysicalDevice> devices(deviceCount);
    VK_CHECK(vkEnumeratePhysicalDevices(ctx.instance, &deviceCount, devices.data()));

    // Default fallback = first GPU in the list
    ctx.physDevice = devices[0];
    // Prefer a Discrete GPU (dedicated) over an Integrated GPU
    // because discrete GPUs have dedicated VRAM and better compute performance.
    for (auto &device : devices)
    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            ctx.physDevice = device;
            break;
        }
    }

    // Log the selected GPU name for debugging
    VkPhysicalDeviceProperties chosen;
    vkGetPhysicalDeviceProperties(ctx.physDevice, &chosen);
    std::cout << "Using device: " << chosen.deviceName << "\n";

    ctx.computeFamily = findComputeFamily(ctx.physDevice);

    // ── 3. CREATE LOGICAL DEVICE ─────────────────────────────────────────────
    // Device = the "usable connection" to the GPU.
    // PhysicalDevice only holds information; Device is what you use to call the API.
    float priority = 1.0f; // queue priority in range [0.0, 1.0]
    VkDeviceQueueCreateInfo qCI{};
    qCI.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qCI.queueFamilyIndex = ctx.computeFamily;
    qCI.queueCount = 1;
    qCI.pQueuePriorities = &priority;

    VkDeviceCreateInfo devCI{};
    devCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devCI.queueCreateInfoCount = 1;
    devCI.pQueueCreateInfos = &qCI;
    VK_CHECK(vkCreateDevice(ctx.physDevice, &devCI, nullptr, &ctx.device));

    // Retrieve the handle to the compute queue we just created (index 0, only 1 queue)
    vkGetDeviceQueue(ctx.device, ctx.computeFamily, 0, &ctx.computeQueue);

    // ── 4. CREATE COMMAND POOL ───────────────────────────────────────────────
    // CommandPool = memory allocator for Command Buffers.
    // Command Buffer = where GPU commands are "recorded" (bind, dispatch, barrier...)
    // The GPU does not execute commands immediately — you record them into a
    // command buffer first, then submit it to a queue.
    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.queueFamilyIndex = ctx.computeFamily;
    poolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    // This flag allows resetting individual command buffers (without resetting the whole pool)
    VK_CHECK(vkCreateCommandPool(ctx.device, &poolCI, nullptr, &ctx.cmdPool));

    return ctx;
}

/**
 * @note Buffer: a wrapper for VkBuffer + VkDeviceMemory.
 *
 * Vulkan separates the Buffer (handle/metadata) from the Memory (actual storage).
 * You must create both and then bind them together —
 * unlike malloc() which handles everything in one step.
 */
struct Buffer
{
    /** @note handle representing the buffer (metadata) */
    VkBuffer handle;
    /** @note the actual memory region on GPU/RAM */
    VkDeviceMemory memory;
    /** @note size in bytes */
    VkDeviceSize size;
};

/**
 * @note Find a memory type that satisfies the given requirements.
 *
 * The GPU exposes multiple memory "heaps" with different properties:
 * - DEVICE_LOCAL:  GPU VRAM, fastest, not directly CPU-readable
 * - HOST_VISIBLE:  CPU can map/read/write via vkMapMemory
 * - HOST_COHERENT: No manual flush required after CPU writes
 */
static uint32_t findMemType(VkPhysicalDevice gpu,
                            uint32_t typeBits,
                            VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(gpu, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
        // (typeBits & (1 << i)) : checks whether memory type i is acceptable for this buffer
        if ((typeBits & (1 << i)) &&
            // (memProps & flags) == flags : checks whether memory type i has ALL required flags
            // Must use == flags (not just != 0) to avoid selecting a type with missing flags.
            (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;

    throw std::runtime_error("No suitable memory type");
}

/**
 * @note Create a buffer, allocate memory for it, and bind them together.
 *
 * Mandatory Vulkan workflow (very different from OpenGL/CUDA):
 *    1. vkCreateBuffer              → create the buffer object (no memory yet)
 *    2. vkGetBufferMemoryRequirements → query how many bytes and what type the GPU needs
 *    3. vkAllocateMemory            → allocate memory of the correct type
 *    4. vkBindBufferMemory          → attach memory to the buffer
 */
Buffer createBuffer(const VulkanContext &ctx,
                    VkDeviceSize size,
                    VkBufferUsageFlags usage,       // intended use: STORAGE, UNIFORM, etc.
                    VkMemoryPropertyFlags memProps) // memory type: HOST_VISIBLE, etc.
{
    Buffer buf{};
    buf.size = size;

    VkBufferCreateInfo bCI{};
    bCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bCI.size = size;
    bCI.usage = usage;
    bCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;                      // only one queue uses this buffer at a time
    VK_CHECK(vkCreateBuffer(ctx.device, &bCI, nullptr, &buf.handle)); // create the buffer object (no memory yet)

    // Ask the GPU: what memory type does this buffer need, and what is the required alignment?
    VkMemoryRequirements memReq;
    // memReq.size         >= size (may be larger due to alignment padding)
    // memReq.memoryTypeBits: bitmask of acceptable memory types
    vkGetBufferMemoryRequirements(ctx.device, buf.handle, &memReq); // query how many bytes and what type the GPU needs

    VkMemoryAllocateInfo allocI{};
    allocI.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocI.allocationSize = memReq.size;
    allocI.memoryTypeIndex = findMemType(ctx.physDevice, memReq.memoryTypeBits, memProps);
    VK_CHECK(vkAllocateMemory(ctx.device, &allocI, nullptr, &buf.memory)); // allocate memory of the correct type
    // 0 = offset within the memory allocation (start from the beginning)
    VK_CHECK(vkBindBufferMemory(ctx.device, buf.handle, buf.memory, 0)); // attach memory to the buffer

    return buf;
}

/** @note Destroy in the reverse order of creation: Buffer first, then Memory */
void destroyBuffer(VkDevice device, Buffer &buf)
{
    vkDestroyBuffer(device, buf.handle, nullptr);
    vkFreeMemory(device, buf.memory, nullptr);
}

/**
 * @note ComputePipeline: groups all objects related to the compute pipeline.
 *
 * A Pipeline in Vulkan = the fully compiled GPU configuration for running a shader.
 * The layout (descriptors, push constants) must be declared before creating the pipeline.
 */
struct ComputePipeline
{
    /** @note layout: "which bindings does the shader need" */
    VkDescriptorSetLayout descLayout;
    /** @note layout: descriptors + push constants combined */
    VkPipelineLayout pipeLayout;
    /** @note the actual compiled pipeline */
    VkPipeline pipeline;
    /** @note pool used to allocate descriptor sets */
    VkDescriptorPool descPool;
    /** @note set that holds bindings to the real buffers */
    VkDescriptorSet descSet;
    /** @note loaded SPIR-V bytecode */
    VkShaderModule shaderModule;
};

/**
 * @note Build the full pipeline:
 *    ShaderModule → DescriptorSetLayout → PipelineLayout → Pipeline
 *    → DescriptorPool → DescriptorSet
 *
 * Descriptor concepts:
 *   - DescriptorSetLayout : schema ("binding 0 is SSBO, binding 1 is SSBO...")
 *   - DescriptorSet       : instance ("binding 0 → bufA, binding 1 → bufB...")
 *   - DescriptorPool      : memory pool used to allocate DescriptorSets
 */
ComputePipeline createPipeline(const VulkanContext &ctx,
                               const std::string &spirvPath)
{
    ComputePipeline cp{};

    // ── 1. LOAD SHADER ───────────────────────────────────────────────────────
    auto code = readSPIRV(spirvPath);
    VkShaderModuleCreateInfo smCI{};
    smCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smCI.codeSize = code.size() * 4; // SPIR-V is stored as uint32[], so byte count = element count × 4
    smCI.pCode = code.data();
    VK_CHECK(vkCreateShaderModule(ctx.device, &smCI, nullptr, &cp.shaderModule));

    // ── 2. DESCRIPTOR SET LAYOUT ─────────────────────────────────────────────
    // Declare 3 SSBOs (Storage Buffer Objects) for buffers A, B, and C.
    // SSBO = arbitrary-size read/write buffer accessible from the shader,
    //        ideal for compute workloads.
    VkDescriptorSetLayoutBinding bindings[3]{};
    for (uint32_t i = 0; i < 3; ++i)
    {
        bindings[i].binding = i;                                        // matches layout(binding = i) in GLSL
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // SSBO
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; // used only in compute shader
    }

    VkDescriptorSetLayoutCreateInfo dslCI{};
    dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslCI.bindingCount = 3;
    dslCI.pBindings = bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(ctx.device, &dslCI, nullptr, &cp.descLayout));

    // ── 3. PUSH CONSTANT ─────────────────────────────────────────────────────
    // Push constant = a way to pass small data (≤128 bytes) from CPU → shader
    // extremely fast, no buffer needed, no descriptor required.
    // Used here to pass N (element count) so the shader knows the boundary.
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(uint32_t); // passing just one uint32

    // ── 4. PIPELINE LAYOUT ───────────────────────────────────────────────────
    // PipelineLayout = declares "which descriptor sets and push constants this
    // pipeline uses". Must be set up before creating the pipeline itself.
    VkPipelineLayoutCreateInfo plCI{};
    plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plCI.setLayoutCount = 1;
    plCI.pSetLayouts = &cp.descLayout;
    plCI.pushConstantRangeCount = 1;
    plCI.pPushConstantRanges = &pcRange;
    VK_CHECK(vkCreatePipelineLayout(ctx.device, &plCI, nullptr, &cp.pipeLayout));

    // ── 5. COMPUTE PIPELINE ──────────────────────────────────────────────────
    // Pipeline = a fully compiled GPU state object.
    // ⚠️  Pipeline creation is expensive (can take a few ms) —
    //     cache it and reuse; never recreate it every frame.
    VkComputePipelineCreateInfo pCI{};
    pCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pCI.stage.module = cp.shaderModule;
    pCI.stage.pName = "main";
    pCI.layout = cp.pipeLayout;
    VK_CHECK(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pCI, nullptr, &cp.pipeline)); // VK_NULL_HANDLE = no pipeline cache (using a cache would speed up subsequent runs)

    // ── 6. DESCRIPTOR POOL + SET ─────────────────────────────────────────────
    // Pool: pre-allocates memory for descriptors (maxSets=1, 3 SSBOs)
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpCI{};
    dpCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpCI.maxSets = 1; // at most 1 descriptor set
    dpCI.poolSizeCount = 1;
    dpCI.pPoolSizes = &poolSize;
    VK_CHECK(vkCreateDescriptorPool(ctx.device, &dpCI, nullptr, &cp.descPool));

    // Allocate 1 descriptor set from the pool, using the layout defined above
    VkDescriptorSetAllocateInfo dsAI{};
    dsAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsAI.descriptorPool = cp.descPool;
    dsAI.descriptorSetCount = 1;
    dsAI.pSetLayouts = &cp.descLayout;
    // Note: descriptor sets do not need to be destroyed individually —
    //       they are freed automatically when the pool is destroyed.
    VK_CHECK(vkAllocateDescriptorSets(ctx.device, &dsAI, &cp.descSet));

    return cp;
}

/**
 * @note Bind real buffers into the descriptor set.
 * After this call, the shader knows: binding 0→bufA, binding 1→bufB, binding 2→bufC.
 *
 * ⚠️  Must be called AFTER both the pipeline and the buffers have been created.
 */
void bindBuffers(const VulkanContext &ctx,
                 ComputePipeline &cp,
                 Buffer &a, Buffer &b, Buffer &c)
{
    VkDescriptorBufferInfo infos[3] = {
        {a.handle, 0, a.size}, // {buffer, offset, range}
        {b.handle, 0, b.size},
        {c.handle, 0, c.size},
    };
    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 3; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = cp.descSet; // which set to write into
        writes[i].dstBinding = i;      // at which binding slot
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    // Update the descriptor set — no return value, cannot fail
    vkUpdateDescriptorSets(ctx.device, 3, writes, 0, nullptr);
}

// MARK: Entrypoint
int main(void)
{
    try
    {
        auto ctx = createContext();

        const VkDeviceSize bufSize = N * sizeof(float);
        // HOST_VISIBLE  = CPU can call vkMapMemory to read/write this memory
        // HOST_COHERENT = CPU writes are automatically visible to the GPU without manual flushing
        // ⚠️  This memory lives in system RAM (not VRAM), which is fine for prototyping
        //     but slower than DEVICE_LOCAL memory in production.
        const VkMemoryPropertyFlags hostFlags =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        auto bufA = createBuffer(ctx, bufSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostFlags);
        auto bufB = createBuffer(ctx, bufSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostFlags);
        auto bufC = createBuffer(ctx, bufSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostFlags);

        // Upload data: map memory → write → unmap
        // vkMapMemory returns a CPU pointer that points directly into GPU-accessible memory
        auto upload = [&](Buffer &buf, float val)
        {
            float *ptr;
            vkMapMemory(ctx.device, buf.memory, 0, bufSize, 0, (void **)&ptr);
            for (uint32_t i = 0; i < N; ++i)
                ptr[i] = val;
            vkUnmapMemory(ctx.device, buf.memory); // always unmap when done
        };
        upload(bufA, 1.0f); // A[i] = 1.0
        upload(bufB, 2.0f); // B[i] = 2.0

        // Pipeline
        auto cp = createPipeline(ctx, "shader.spv");
        bindBuffers(ctx, cp, bufA, bufB, bufC);

        // ── RECORD COMMAND BUFFER ─────────────────────────────────────────────
        // Vulkan does not allow calling GPU commands directly. You must:
        //   1. Begin the command buffer
        //   2. Record commands (bind, dispatch, barrier...)
        //   3. End the command buffer
        //   4. Submit to a queue → the GPU executes it
        VkCommandBufferAllocateInfo cbAI{};
        cbAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbAI.commandPool = ctx.cmdPool;
        cbAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; // PRIMARY = submitted directly to a queue
        cbAI.commandBufferCount = 1;

        VkCommandBuffer cmd;
        VK_CHECK(vkAllocateCommandBuffers(ctx.device, &cbAI, &cmd));

        VkCommandBufferBeginInfo beginI{};
        beginI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginI.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // ONE_TIME = this command buffer is submitted only once; the driver can optimize accordingly
        VK_CHECK(vkBeginCommandBuffer(cmd, &beginI));

        // Activate the compute pipeline
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cp.pipeline);
        // Attach the descriptor set (bufA, bufB, bufC) to the pipeline
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                cp.pipeLayout, 0, 1, &cp.descSet, 0, nullptr);

        // Pass N via push constant (faster than UBO/SSBO for small data)
        uint32_t n = N;
        vkCmdPushConstants(cmd, cp.pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(n), &n);

        // DISPATCH: "run the shader with X workgroups"
        // Total threads = groups * LOCAL_SIZE = groups * 256
        // ⚠️  Ceiling division formula: (N + LOCAL_SIZE - 1) / LOCAL_SIZE
        //     ensures enough workgroups to cover all N elements,
        //     even when N is not evenly divisible by LOCAL_SIZE.
        uint32_t groups = (N + LOCAL_SIZE - 1) / LOCAL_SIZE;
        vkCmdDispatch(cmd, groups, 1, 1); // 3D dispatch: x=groups, y=1, z=1

        // ── MEMORY BARRIER ────────────────────────────────────────────────────
        // A barrier is a synchronization fence within the GPU pipeline.
        // Without it, the GPU may not have finished writing before the CPU reads.
        //
        // srcStageMask  = "wait for this stage to finish"      → COMPUTE_SHADER
        // dstStageMask  = "before this stage begins"           → HOST (CPU read)
        // srcAccessMask = type of access to flush              → SHADER_WRITE
        // dstAccessMask = type of access that must be visible  → HOST_READ
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;

        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // src stage
            VK_PIPELINE_STAGE_HOST_BIT,           // dst stage
            0,                                    // dependency flags (0 = none)
            1, &barrier,                          // memory barriers
            0, nullptr,                           // buffer memory barriers (unused)
            0, nullptr);                          // image memory barriers (unused)

        VK_CHECK(vkEndCommandBuffer(cmd));

        // ── SUBMIT & WAIT ─────────────────────────────────────────────────────
        VkSubmitInfo submitI{};
        submitI.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitI.commandBufferCount = 1;
        submitI.pCommandBuffers = &cmd;

        // Fence = CPU-GPU synchronization primitive.
        // After submit, the GPU runs asynchronously. The CPU calls vkWaitForFences
        // to block until the GPU finishes.
        // ⚠️  Unlike Semaphores (GPU-GPU sync), Fences are used for CPU-GPU sync.
        VkFenceCreateInfo fCI{};
        fCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fCI.flags = 0;
        VkFence fence;
        VK_CHECK(vkCreateFence(ctx.device, &fCI, nullptr, &fence));
        VK_CHECK(vkQueueSubmit(ctx.computeQueue, 1, &submitI, fence));
        VK_CHECK(vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX)); // VK_TRUE = wait for ALL fences; UINT64_MAX = no timeout

        // ── VERIFY RESULT ─────────────────────────────────────────────────────
        float *result;
        vkMapMemory(ctx.device, bufC.memory, 0, bufSize, 0, (void **)&result);
        bool ok = true;
        for (uint32_t i = 0; i < N; ++i)
            if (result[i] != 3.0f)
            {
                ok = false;
                break;
            }
        vkUnmapMemory(ctx.device, bufC.memory);

        std::cout << (ok ? "✅ Vector add PASSED (C[i] = 3.0)" : "❌ FAILED") << "\n";

        // ── CLEANUP ───────────────────────────────────────────────────────────
        // ⚠️  Destroy in REVERSE order of creation.
        //     Always destroy child objects before parent objects.
        //     Wrong order → validation layer errors or crash.
        vkDestroyFence(ctx.device, fence, nullptr);
        destroyBuffer(ctx.device, bufA);
        destroyBuffer(ctx.device, bufB);
        destroyBuffer(ctx.device, bufC);
        vkDestroyShaderModule(ctx.device, cp.shaderModule, nullptr);
        vkDestroyDescriptorPool(ctx.device, cp.descPool, nullptr); // frees descriptor sets inside
        vkDestroyPipeline(ctx.device, cp.pipeline, nullptr);
        vkDestroyPipelineLayout(ctx.device, cp.pipeLayout, nullptr);
        vkDestroyDescriptorSetLayout(ctx.device, cp.descLayout, nullptr);
        vkDestroyCommandPool(ctx.device, ctx.cmdPool, nullptr); // frees command buffers inside
        vkDestroyDevice(ctx.device, nullptr);
        vkDestroyInstance(ctx.instance, nullptr);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}