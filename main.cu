#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include "matrix_build.h"
#include <cublas_v2.h>
#include <magma_lapack.h>
#include <magma_v2.h>
using namespace std;



int main() {
    
    

    magma_init();
    
    magma_int_t dev_t = 0;
    magma_queue_t queue_qr = NULL;
    magma_queue_create(dev_t,&queue_qr);
   
   //set up device and queue

    
    
    int numFluid = 2268;
    int numBoundary = 1903;
    int numGhost = 0;

    int numParticle = numFluid + numBoundary + numGhost;
    
    int numNeighbourone = 177;
    int numNeighbour = 177;
// read data from txt file
    double* inPressure = new double[numParticle];
    double* inVolume = new double[numParticle];
    double* inSoundSpeed = new double[numParticle];
    double* inVelocity = new double[numParticle];
    int* neighbourlist0 = new int[numFluid*numNeighbourone];
    int* neighbourlist1 = new int[numFluid*numNeighbourone];
    int* neighboursize0 =new int[numFluid];
    int* neighboursize1 = new int[numFluid];
    int* LPFOrder0 = new int[numFluid];
    int* LPFOrder1 = new int[numFluid];
    double* xPosition = new double[numParticle];
    double* yPosition = new double[numParticle];
    
    
    //store data into array

    ifstream myfile;
    
    myfile.open("xPosition.txt");
    for(int i=0;i<numParticle;i++){
        double tem;
        myfile>>tem;
        xPosition[i] = tem;
    }
    myfile.close();

    myfile.open("yPosition.txt");
    for(int i=0;i<numParticle;i++){
        double tem;
        myfile>>tem;
        yPosition[i] = tem;
    }
    myfile.close();


   myfile.open("inPressure.txt");
    for(int i=0;i<numParticle;i++){
        double tem;
        myfile>>tem;
        inPressure[i]=tem;
    }
    myfile.close();
    
    myfile.open("inVelocity.txt");
    for(int i=0;i<numParticle;i++){
        double tem;
        myfile>>tem;
        inVelocity[i]=tem;
    }
    myfile.close();
    
    myfile.open("inSoundSpeed.txt");
    for(int i=0;i<numParticle;i++){
       double tem;
        myfile>>tem;
        inSoundSpeed[i]=tem;
    }
    myfile.close();
   
    myfile.open("inVolume.txt");
    for(int i=0;i<numParticle;i++){
       double tem;
        myfile>>tem;
        inVolume[i]=tem;
    }
    myfile.close();
   
  myfile.open("neighbourlist0.txt");
    for(int i=0;i<numFluid*numNeighbourone;i++){
        int tem;
        myfile>>tem;
        neighbourlist0[i]=tem;
    }
    myfile.close();
    myfile.open("neighbourlist1.txt");
    for(int i=0;i<numFluid*numNeighbourone;i++){
        int tem;
        myfile>>tem;
        neighbourlist1[i]=tem;
    }
    myfile.close();
      myfile.open("neighboursize0.txt");
    for(int i=0;i<numFluid;i++){
       double tem;
        myfile>>tem;
        neighboursize0[i]=tem;
    }
    myfile.close();
     myfile.open("neighboursize1.txt");
    for(int i=0;i<numFluid;i++){
       double tem;
        myfile>>tem;
        neighboursize1[i]=tem;
    }
    myfile.close();


    fill_n(LPFOrder0,numFluid,1);
    fill_n(LPFOrder1,numFluid,1);

//device arrays which need copy
    
    double* d_xPosition;
    double* d_yPosition;
    double* d_inPressure;
    double* d_inVolume;
    double* d_inSoundSpeed;
    double* d_inVelocity;
    int* d_neighbourlist0;
    int* d_neighbourlist1;
    int* d_neighboursize0;
    int* d_neighboursize1;
    int* d_LPFOrder0;
    int* d_LPFOrder1;
   
// device arrays which dont need memcopy
    int* d_numRow;
    int* d_numCol;
   
   
   
   
    cudaMalloc((void**)&d_xPosition,sizeof(double)*numParticle);
    cudaMalloc((void**)&d_yPosition,sizeof(double)*numParticle);
    cudaMalloc((void**)&d_inPressure,sizeof(double)*numParticle);
    cudaMalloc((void**)&d_inVolume,sizeof(double)*numParticle );
    cudaMalloc((void**)&d_inVelocity, sizeof(double)*numParticle);
    cudaMalloc((void**)&d_inSoundSpeed,sizeof(double)*numParticle);
    cudaMalloc((void**)&d_neighbourlist0,sizeof(int)*numFluid*numNeighbourone);
    cudaMalloc((void**)&d_neighbourlist1,sizeof(int)*numFluid*numNeighbourone);
    cudaMalloc((void**)&d_neighboursize0,sizeof(int)*numFluid);
    cudaMalloc((void**)&d_neighboursize1,sizeof(int)*numFluid);
    cudaMalloc((void**)&d_LPFOrder0,sizeof(int)*numFluid);
    cudaMalloc((void**)&d_LPFOrder1,sizeof(int)*numFluid);
    cudaMalloc((void**)&d_numRow,sizeof(int)*numFluid);
    cudaMalloc((void**)&d_numCol,sizeof(int)*numFluid);
cout<<"-------------------------cuda allocate done----------------------------------"<<endl;

//memory copy

    cudaMemcpy(d_xPosition,xPosition,sizeof(double)*numParticle,cudaMemcpyHostToDevice);
    cudaMemcpy(d_yPosition,yPosition,sizeof(double)*numParticle,cudaMemcpyHostToDevice);
    cudaMemcpy(d_inPressure,inPressure,sizeof(double)*numParticle,cudaMemcpyHostToDevice);
    cudaMemcpy(d_inVolume,inVolume,sizeof(double)*numParticle,cudaMemcpyHostToDevice);
    cudaMemcpy(d_inVelocity,inVelocity,sizeof(double)*numParticle,cudaMemcpyHostToDevice);
    cudaMemcpy(d_inSoundSpeed,inSoundSpeed,sizeof(double)*numParticle,cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbourlist0,neighbourlist0,sizeof(int)*numFluid*numNeighbourone,cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbourlist1,neighbourlist1,sizeof(int)*numFluid*numNeighbourone,cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighboursize0,neighboursize0,sizeof(int)*numFluid,cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbourlist1,neighboursize1,sizeof(int)*numFluid,cudaMemcpyHostToDevice);
    cudaMemcpy(d_LPFOrder0,LPFOrder0,sizeof(int)*numFluid,cudaMemcpyHostToDevice);
    cudaMemcpy(d_LPFOrder1,LPFOrder1,sizeof(int)*numFluid,cudaMemcpyHostToDevice);



    
    
   cout<<"----------------------------mem allocate and copy done--------------------------------"<<endl; 
    
    
    //----------------OUTPUT-------------------------
/*    
    double* outVelocity = new double[numParticle];
    double* outPressure = new double[numParticle];
    double* outSoundSpeed = new double[numParticle];
    double* outVolume = new double[numParticle];

*/



  cout<<"--------------------------------Testing---------------------------------------"<<endl;  

    dim3 blocks(128,1);
    dim3 threads(128,1);
    computeRowandCol<<<blocks,threads>>>(d_neighboursize0,d_numRow,d_numCol,d_LPFOrder0,numFluid);


//build double device pointer A
    double** A;
    cudaMalloc((void**)&A,sizeof(double*)*numFluid);
    double** A_temp = new double*[numFluid];
   for(int i=0;i<numFluid;i++){
        cudaMalloc((void**)&A_temp[i],sizeof(double)*5*numNeighbourone);
    }
    cudaMemcpy(A, A_temp,sizeof(double*)*numFluid,cudaMemcpyHostToDevice);
//build distance array

    double* d_distance;
    cudaMalloc((void**)&d_distance,sizeof(double)*numFluid);
  
    cout<<"------------------------------Testing2---------------------------"<<endl;
    dim3 blocks1(128,1);
    dim3 threads1(128,1);
 


 
  
    computeA2D<<<blocks1,threads1>>>(d_neighbourlist0,d_LPFOrder0,d_numRow,d_xPosition,d_yPosition, numFluid,numNeighbourone,A,d_distance);

    
    cout<<"-----------------------Testing Done------------------------------"<<endl;
/*    
for(int i=0;i<numFluid;i++){
   cout<<"A of number: "<<i<<endl;
   magma_dprint_gpu(6,1,A_temp[i],6,queue_qr);
   }
*/
    

    

//Process QR batched mode
   
   
   
    magma_int_t m = 3;
    magma_int_t n = 2;
    magma_int_t lda = 3;
    magma_int_t min_mn = min(m,n);
    double **Tau;
    cudaMalloc((void**)&Tau,numFluid*sizeof(double*));
    double** Tau_temp = new double*[numFluid];
    for(int i=0;i<numFluid;i++){
        cudaMalloc((void**)&Tau_temp[i],sizeof(double)*min_mn);
    }



    cudaMemcpy(Tau, Tau_temp, sizeof(double*)*numFluid, cudaMemcpyHostToDevice);  
    magma_int_t* info;

    cudaMalloc((void**)&info,numFluid*sizeof(magma_int_t));
    
    magma_int_t batchid = numFluid;

    //Start QR
  
    magma_dgeqrf_batched(m,n,A,lda,Tau,info,batchid,queue_qr);
  

  
  
    cout<<"-------------------------QR DONE----------------------------------"<<endl;





    //build right hand side B
    double** B;
    cudaMalloc((void**)&B,sizeof(double*)*numFluid);
    double** B_temp = new double*[numFluid];
    for(int i=0;i<numFluid;i++){
        cudaMalloc((void**)&B_temp[i],sizeof(double)*numNeighbourone);
    }
    cudaMemcpy(B,B_temp,sizeof(double*)*numFluid,cudaMemcpyHostToDevice);
    computeB<<<blocks,threads>>>(d_neighbourlist0, d_numRow, d_inPressure, numNeighbourone, numFluid, B);
    /*
for(int i=0;i<numFluid;i++){
   cout<<"number: "<<i<<endl;
   magma_dprint_gpu(3,1,B_temp[i],3,queue_qr);
   }
*/


//solver linear system

    double **result;
    cudaMalloc((void**)&result,numFluid*sizeof(double*));
    double** result_temp = new double*[numFluid];
    for(int i=0;i<numFluid;i++){
        cudaMalloc((void**)&result_temp[i],sizeof(double)*min_mn);
    }



    cudaMemcpy(result, result_temp, sizeof(double*)*numFluid, cudaMemcpyHostToDevice);  
   
    computeLS<<<blocks,threads>>>(A,B,Tau,d_numRow,d_numCol,numFluid, result);

cout<<"BIG SUCCESS!!"<<endl;
for(int i=0;i<numFluid;i++){
   cout<<"number: "<<i<<endl;
   magma_dprint_gpu(2,1,result_temp[i],2,queue_qr);
}   
    magma_queue_destroy(queue_qr);
    magma_finalize();
//release memory


    delete[] inPressure;
    delete[] inVolume;
    delete[] inVelocity;
    delete[] inSoundSpeed;
    delete[] neighbourlist0;
    delete[] neighbourlist1;
    delete[] neighboursize0;
    delete[] neighboursize1;
    delete[] LPFOrder0;
    delete[] LPFOrder1;
    delete[] A_temp;
    delete[] xPosition;
    delete[] yPosition;
    delete[] B_temp;
    delete[] result_temp;
    
    cudaFree(d_neighboursize0);
    cudaFree(d_neighboursize1);
    cudaFree(d_neighbourlist0);
    cudaFree(d_neighbourlist1);
    cudaFree(d_LPFOrder0);
    cudaFree(d_LPFOrder1);
    cudaFree(d_inPressure);
    cudaFree(d_inVolume);
    cudaFree(d_inSoundSpeed);
    cudaFree(d_inVelocity);
    cudaFree(d_numRow);
    cudaFree(d_numCol);
    cudaFree(A);
    cudaFree(result);
    for(int i=0;i<numParticle;i++){
        cudaFree(A_temp[i]);
        cudaFree(B_temp[i]);
        cudaFree(result_temp[i]);
    }
    cudaFree(d_distance);
    cudaFree(B);
    

// QR
    delete[] Tau_temp;
    cudaFree(Tau);
    for(int i=0;i<numParticle;i++){
        cudaFree(Tau_temp[i]);
    }
    cudaFree(info);
}

    

