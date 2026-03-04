/**
 * Matrix multiplication
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
    auto library = device->newLibrary(NS::String::string("main00002.metallib",
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
    auto func = library->newFunction(NS::String::string("matrix_multiplication",
                                                        NS::ASCIIStringEncoding));
    auto pipeline = device->newComputePipelineState(func, &error);

    // Prepare input data
    uint32_t M = 1024, N = 512, K = 1024;
    std::vector<float> A(M * N), B(N * K), C(M * K);
    for (int i = 0; i < M * N; i++)
        A[i] = i;
    for (int i = 0; i < N * K; i++)
        B[i] = i;

    auto bufA = device->newBuffer(A.data(),
                                  M * N * sizeof(float),
                                  MTL::ResourceStorageModeShared);
    auto bufB = device->newBuffer(B.data(),
                                  N * K * sizeof(float),
                                  MTL::ResourceStorageModeShared);
    auto bufC = device->newBuffer(M * K * sizeof(float),
                                  MTL::ResourceStorageModeShared);
    auto bufM = device->newBuffer(&M, sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto bufN = device->newBuffer(&N, sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto bufK = device->newBuffer(&K, sizeof(uint32_t), MTL::ResourceStorageModeShared);

    // Execute compute command
    auto cmdBuffer = queue->commandBuffer();
    auto encoder = cmdBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(bufA, 0, 0);
    encoder->setBuffer(bufB, 0, 1);
    encoder->setBuffer(bufC, 0, 2);
    encoder->setBuffer(bufM, 0, 3);
    encoder->setBuffer(bufN, 0, 4);
    encoder->setBuffer(bufK, 0, 5);

    MTL::Size gridSize(M, K, 1);
    MTL::Size threadGroupSize(16, 16, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    encoder->endEncoding();

    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();

    // Read results
    memcpy(C.data(), bufC->contents(), M * K * sizeof(float));

    std::cout << "Result: ";
    for (auto v : C)
        std::cout << v << " ";
    std::cout << std::endl;

    encoder->release();
    cmdBuffer->release();
    bufA->release();
    bufB->release();
    bufC->release();
    bufM->release();
    bufN->release();
    bufK->release();
    pipeline->release();
    func->release();
    library->release();
    queue->release();
    device->release();

    return EXIT_SUCCESS;
}