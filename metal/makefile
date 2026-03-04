CXX=/usr/bin/clang++ # use Apple Clang instead of LLVM Clang
SDKROOT=$(xcrun --show-sdk-path)
TARGET = main00001

dev: build_kernel build_cpp
	./main

build_cpp:
	$(CXX) example/$(TARGET).cpp -std=c++17 -fobjc-arc \
		-framework Metal -framework Foundation \
		-I third_party/metal-cpp \
		-o main

build_kernel:
	xcrun -sdk macosx metal -c example/$(TARGET).metal -o example/$(TARGET).air
	xcrun -sdk macosx metallib example/$(TARGET).air -o $(TARGET).metallib


clean:
	rm -rf example/*.air *.metallib main