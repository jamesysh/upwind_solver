#include <iostream>
#include "matrix_build.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdio.h>

__global__ void computeRowandCol(const int*neighboursize, int*numRow, int*numCol,int* LPFOrder,const int numParticle)
{
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

__global__ void computeB(const int* neighbourList, const int* numRow, const double* inData, const int maxNumNeighbourOne, const int numParticle,
                        double** b)//output vector b

{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int offset = blockDim.x*gridDim.x;
    while(tid<numParticle){
        for(int i=0;i<numRow[tid];i++){
            int neiIndex = neighbourList[tid*maxNumNeighbourOne + i];
            b[tid][i] = inData[neiIndex] - inData[tid];
        }
    
        tid = tid + offset;
    }
}


__global__ void computeLS(double**A,double**B,double**Tau, const int* numRow,const int* numCol ,const int numFluid, 
                        double**Result)//output result
{

    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int offset = blockDim.x*gridDim.x;
    while(tid < numFluid){
        int nrow = numRow[tid];
        int ncol = numCol[tid];
        for(int i=0;i<ncol;i++){
//cant build v here cauz we need a fixed size in cuda kernal    no double v[ncol]
            double v_times_b = 0.;
            for(int j=0;j<nrow;j++){
                if(j < i) continue;
                if(j == i) v_times_b += 1*B[tid][j];
                else v_times_b += A[tid][j+i*nrow]*B[tid][j];
            }
            v_times_b *= Tau[tid][i];

            for(int j=0;j<nrow;j++){
                if(j < i) continue;
                if(j == i) B[tid][j] -= v_times_b;
                else
                B[tid][j] -= v_times_b*A[tid][j+i*nrow];
            }

        }

//compute QTB complete

//Backsubstitution
        for(int i=ncol-1;i>=0;i--){
            Result[tid][i] = B[tid][i]/A[tid][i*nrow+i];
            for(int j=0;j<i;j++){
                
                B[tid][j] -= A[tid][j+i*nrow]*Result[tid][i];
            }


        }
    tid += offset;

    }
}























