CC = g++
NVCC = nvcc
CFLAGS = -Wall -std=c++11 -O3
NVCCFLAGS = -O3 -std=c++11
LDFLAGS = -lcudart

EXECUTABLE = cuda_seam_carving

all: $(EXECUTABLE)

$(EXECUTABLE): cuda_seam_carving.o cuda_kernels.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

cuda_seam_carving.o: cuda_seam_carving.cpp cuda_kernels.cuh
	$(CC) $(CFLAGS) -c $<

cuda_kernels.o: cuda_kernels.cu cuda_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -c $<

clean:
	rm -f *.o $(EXECUTABLE)

.PHONY: all clean 