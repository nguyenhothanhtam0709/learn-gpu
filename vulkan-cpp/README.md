# Overview

C++ GPGPU computing using [Vulkan](https://www.vulkan.org/).

## Installation

### Ubuntu

```sh
sudo apt update
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools libvulkan-dev
# Install Vulkan sdk with Glsl tool
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools libvulkan-dev spirv-tools glslc
# Install Glsl tool
sudo apt install glslang-tools

# Check Vulkan installation
vulkaninfo --summary
```
