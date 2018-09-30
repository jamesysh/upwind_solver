#include <iostream>
#include "matrix_build.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdio.h>
__global__ void computeRowandCol(const int*neighboursize, int*numRow, int*numCol,int* LPFOrder,const int numParticle){
   int tid = threadIdx.x + blockIdx.x*blockDim.x;
   int offset = blockDim.x*gridDim.x;
   int numrow2nd = 36;
   int numrow1st = 3;
   int numcol2nd = 5;
   int numcol1st = 2;
   while(tid<numParticle){
    
   int numNeisize =  neighboursize[tid];
       if(LPFOrder[tid]==2){
           if(numNeisize >= numrow2nd){
               numRow[tid] = numrow2nd;
               numCol[tid] = numcol2nd;
           }
            else LPFOrder[tid] = 1;
       }
    
       if(LPFOrder[tid]==1){
           if(numNeisize >= numrow1st){
               numRow[tid] = numrow1st;
               numCol[tid] = numcol1st;
           }
            else LPFOrder[tid] = 0;


       }
        
       if(LPFOrder[tid]==0){
            numRow[tid] = 0;
            numCol[tid] = 0;

       }
        tid = tid + offset;
}

    __syncthreads();
}



__global__ void computeA2D(const int*neighbourList,const int*LPFOrder,const int* numRow,const double*x,const double*y, const int numParticle,const int maxNeighbourOneDir,double**A,double*dis)
    {
        
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int offset = blockDim.x*gridDim.x;
    //printf("runing form %d\n",tid);
    while(tid<numParticle){
        int numOfRow = numRow[tid];
        double distance = sqrt((x[neighbourList[tid*maxNeighbourOneDir+numOfRow/4]]-x[tid])*(x[neighbourList[tid*maxNeighbourOneDir+numOfRow/4]]-x[tid])+(y[neighbourList[tid*maxNeighbourOneDir+numOfRow/4]]-y[tid])*(y[neighbourList[tid*maxNeighbourOneDir+numOfRow/4]]-y[tid]));
        if(LPFOrder[tid] == 1){
            
            for(int i=0;i<numOfRow;i++){
            
                int neiIndex = neighbourList[tid*maxNeighbourOneDir+i];
                    
                double h = (x[neiIndex]-x[tid])/distance;
                
                double k = (y[neiIndex]-y[tid])/distance;
                A[tid][i] = h;
                A[tid][i+numOfRow] = k;
            }   
    
        }
        else if(LPFOrder[tid] == 2){
            for(int i=0;i<numOfRow;i++){
                int neiIndex = neighbourList[tid*maxNeighbourOneDir+i];
                double h = (x[neiIndex]-x[tid])/distance;
                double k = (y[neiIndex]-y[tid])/distance;
                A[tid][i] = h;
                A[tid][i + numOfRow] = k;
                A[tid][i + 2*numOfRow] = 0.5*h*h;
                A[tid][i + 3*numOfRow] = 0.5*k*k;
                A[tid][i + 4*numOfRow] = h*k;


            }
        
        } 
    dis[tid] = distance;
    tid = tid + offset;
    }
}



























