/**
 * ReLU
 */

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <cstdlib>
#include <iostream>
#include <vector>

int main()
{
    // Create Metal device & command queue
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    MTL::CommandQueue *queue = device->newCommandQueue();

    // Load compiled metallib
    NS::Error *error = nullptr;
    auto library = device->newLibrary(NS::String::string("main00003.metallib",
                                                         NS::ASCIIStringEncoding),
                                      &error);
    if (!library)
    {
        std::cerr << "Failed to load metallib: "
                  << error->localizedDescription()->utf8String()
                  << std::endl;
        return EXIT_FAILURE;
    }

    // Create compute pipeline
    auto func = library->newFunction(NS::String::string("relu",
                                                        NS::ASCIIStringEncoding));
    auto pipeline = device->newComputePipelineState(func, &error);

    // Prepare input data
    const int N = 8;
    std::vector<float> A(N), B(N);
    for (int i = 0; i < N; i++)
        A[i] = i;

    auto bufA = device->newBuffer(A.data(),
                                  N * sizeof(float),
                                  MTL::ResourceStorageModeManaged);
    auto bufB = device->newBuffer(B.data(),
                                  N * sizeof(float),
                                  MTL::ResourceStorageModeManaged);
    auto bufN = device->newBuffer(&N, sizeof(uint32_t), MTL::ResourceStorageModeShared);

    // Execute compute command
    auto cmdBuffer = queue->commandBuffer();
    auto encoder = cmdBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(bufA, 0, 0);
    encoder->setBuffer(bufB, 0, 1);
    encoder->setBytes(bufN, 0, 2);

    MTL::Size gridSize(N, 1, 1);
    MTL::Size threadGroupSize(1, 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    encoder->endEncoding();

    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();

    // Read results
    memcpy(B.data(), bufB->contents(), N * sizeof(float));

    std::cout << "Result: ";
    for (auto v : B)
        std::cout << v << " ";
    std::cout << std::endl;

    encoder->release();
    cmdBuffer->release();
    bufA->release();
    bufB->release();
    bufN->release();
    pipeline->release();
    func->release();
    library->release();
    queue->release();
    device->release();

    return EXIT_SUCCESS;
}