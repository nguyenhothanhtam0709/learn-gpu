
comp:
	g++ src/main00003.cpp \
	-lvulkan \
	-std=c++17 \
	-o main

comp_shader:
	glslangValidator -V src/main00003-shader.comp.glsl -o shader.spv

dev: comp
	./main