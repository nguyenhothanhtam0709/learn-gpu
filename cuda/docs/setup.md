## Notes

Run C++ Cuda with NVCC (Nvidia CUDA Compiler) on Google Colab.

```python
# Check cuda version
!nvidia-smi

# Install compatible cuda-toolkit
!apt-get update
!apt-get install -y cuda-toolkit-12-4

# Export env
import os
os.environ["PATH"] = "/usr/local/cuda-12.4/bin:" + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.4/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

# Check CUDA toolchain
!nvcc --version

# Install and load jupyter nvcc extension
!pip install nvcc4jupyter
%load_ext nvcc4jupyter
```
