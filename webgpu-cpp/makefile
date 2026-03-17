DAWN_PATH=./externals/dawn/install/Release

build:
	g++ src/main00001.cpp \
		-I$(DAWN_PATH)/include \
		-L$(DAWN_PATH)/lib \
		-lwebgpu_dawn \
		-std=c++17 \
		-o hello_webgpu