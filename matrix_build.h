#ifndef MATRIX_BUILD
#define MATRIX_BUILD
#include <cuda.h>
#include <cuda_runtime_api.h>
//compute the number of rows and columns for each matrix
__global__ void computeRowandCol(const int*neighboursize, int*numRow, int*numCol, int* LPFOrder,const int numParticle);



//build the matrix A in parallel
__global__ void computeA2D(const int*neighbourList,const int*LPFOrder,const int* numRow,const double*x,const double*y,
        const int numParticle,const int maxNeighbourOneDir, double**A,double*dis);





















#endif
