CC = nvcc
DEBUG = -g
CUDA_DIR = /gpfs/software/cuda/9.1/toolkit/lib64
MAGMA_DIR = /gpfs/home/shyyuan/local/magma-2.4.0/lib/
LIBS = -L $(CUDA_DIR) -L $(MAGMA_DIR) 
INCS = -I /gpfs/home/shyyuan/local/magma-2.4.0/include/

CFLAGS = $(DEBUG) $(LIBS) $(INCS) -DADD_ -c
UPFLAGS = $(DEBUG) $(LIBS) $(INCS)
OBJS = matrix_build.o main.o

all: upwind
	
upwind: $(OBJS)
		$(CC) $(UPFLAGS) $(OBJS) -o upwind -lcudart -lcublas -lcusolver -lmagma

matrix_build.o: matrix_build.h
		$(CC) $(CFLAGS) matrix_build.cu
main.o: matrix_build.h
		$(CC) $(CFLAGS) main.cu




clean:
	rm *.o upwind 
