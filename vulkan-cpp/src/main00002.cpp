#include <cstdlib>
#include <iostream>
#include <vector>

#include <vulkan/vulkan.h>

int main(void)
{
    // Create instance
    VkInstance instance;
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
    {
        std::cerr << "Failed to create instance\n";
        return EXIT_FAILURE;
    }

    // Enumerate physical devices (GPU/CPU)
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0)
    {
        std::cerr << "No Vulkan devices found\n";
        return EXIT_FAILURE;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const auto &device : devices)
    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);

        VkPhysicalDeviceFeatures features;
        vkGetPhysicalDeviceFeatures(device, &features);

        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(device, &memProps);

        std::cout << "=== Device ===\n";
        std::cout << "Name: " << props.deviceName << "\n";
        std::cout << "API Version: "
                  << VK_VERSION_MAJOR(props.apiVersion) << "."
                  << VK_VERSION_MINOR(props.apiVersion) << "."
                  << VK_VERSION_PATCH(props.apiVersion) << "\n";

        std::cout << "Driver Version: " << props.driverVersion << "\n";
        std::cout << "Vendor ID: " << props.vendorID << "\n";
        std::cout << "Device ID: " << props.deviceID << "\n";

        std::cout << "Max Image Dimension 2D: "
                  << props.limits.maxImageDimension2D << "\n";

        std::cout << "Geometry Shader support: "
                  << (features.geometryShader ? "YES" : "NO") << "\n";

        std::cout << "Memory Heaps: " << memProps.memoryHeapCount << "\n";
        for (uint32_t i = 0; i < memProps.memoryHeapCount; i++)
        {
            std::cout << "  Heap " << i
                      << " Size: "
                      << memProps.memoryHeaps[i].size / (1024 * 1024)
                      << " MB\n";
        }

        std::cout << "------------------------\n";
    }

    // Cleanup
    vkDestroyInstance(instance, nullptr);

    return EXIT_SUCCESS;
}