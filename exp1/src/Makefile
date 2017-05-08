CC=g++
LINKER_DIRS=-L/usr/local/cuda-6.0/lib64/stubs -L/usr/local/cuda-6.0/lib64
LINKER_FLAGS=-lcudart -lcuda
NVCC=nvcc
CUDA_ARCHITECTURE=20
OCELOT=

all: main

main: main.o cuda_wrapper.o
	$(CC) main.o cuda_wrapper.o -o main $(LINKER_DIRS) $(LINKER_FLAGS) $(OCELOT)

main.o: main.cpp
	$(CC) main.cpp -c -I .

cuda_wrapper.o: cuda_wrapper.cu
	$(NVCC) -c cuda_wrapper.cu -arch=sm_$(CUDA_ARCHITECTURE)

clean:
	rm -f *.o main
