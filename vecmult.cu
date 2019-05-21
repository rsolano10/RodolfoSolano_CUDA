#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>

#define BLOCK 4

//Cuda error checking - non mandatory
void cudaCheckError() {
    cudaError_t e=cudaGetLastError();
    if(e!=cudaSuccess) {
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
      exit(0); 
    }
}

//GPU kernel 
__global__ 
void multPGU(int *matAPtr,int *matBPtr, int *resMat,int colAPtr, int rowBPtr, int colBPtr)
{
    //cuenta iteraciones de filas y columnas
    int rowCount = blockIdx.y * blockDim.y + threadIdx.y;
    int colCount = blockIdx.x * blockDim.x + threadIdx.x;

    int res = 0; // resultado de la operacion

    if( colCount < colBPtr && rowCount < colAPtr)
    {
        for(int i = 0; i < rowBPtr; i++)
        {
            res += matAPtr[rowCount * rowBPtr + i] * matBPtr[i * colBPtr + colCount];
        }
        resMat[rowCount * colBPtr + colCount] = res;
    }
}

//CPU function
void multPC(int *matAPtr, int *matBPtr, int *resMat, int colAPtr, int rowBPtr, int colBPtr) {
    for (int i = 0; i < colAPtr; ++i)
    {
        for (int j = 0; j < colBPtr; ++j)
        {
            int res = 0.0;
            for (int h = 0; h < rowBPtr; ++h)
            {
                res += matAPtr[i * rowBPtr + h] * matBPtr[h * colBPtr + j];
            }
            resMat[i * colBPtr + j] = res;
        }
    }
}

int main(int argc, char const *argv[]){
    int colA = 4, rowB = 4, colB = 4;
    srand(123456987);
    int *mA, *mB, *mC, *mD;

    //Malloc some mem for host
    cudaMallocHost((void **) &mA, sizeof(int)*colA*rowB);
    cudaMallocHost((void **) &mB, sizeof(int)*rowB*colB);
    cudaMallocHost((void **) &mC, sizeof(int)*colA*colB);
    cudaMallocHost((void **) &mD, sizeof(int)*colA*colB);

    //Fill  matrix
    for (int i = 0; i < colA; ++i) {
        for (int j = 0; j < rowB; ++j) {
            mA[i * n + j] = rand() % 1024;
            mB[i * k + j] = rand() % 1024;
        }
    }
    
    float GPUTime, CPUTime;
    
    //Define event time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    //Start time lapse
    cudaEventRecord(start, 0);

    //Mem allocs ptrs
    int *d_a, *d_b, *d_c;

    //Malloc mem
    cudaMalloc((void **) &d_a, sizeof(int)*colA*rowB);
    cudaMalloc((void **) &d_b, sizeof(int)*rowB*colB);
    cudaMalloc((void **) &d_c, sizeof(int)*colA*colB);

    //mem copy from host
    cudaMemcpy(d_a, mA, sizeof(int)*colA*rowB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, mB, sizeof(int)*rowB*colB, cudaMemcpyHostToDevice);

    unsigned int totalRows = (colA + BLOCK - 1) / BLOCK;
    unsigned int totalCols = (colB + BLOCK - 1) / BLOCK;

    //dim3 definitions
    dim3 dimGrid(totalCols, totalRows);
    dim3 dimBlock(BLOCK, BLOCK);
    
    //Call GPU Func
    multPGU<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, colA, rowB, colB);
    
    //Copy to host
    cudaMemcpy(mC, d_c, sizeof(int)*colA*colB, cudaMemcpyDeviceToHost);
    
    //Sinchronize
    cudaThreadSynchronize();
    
    //Stop clock
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    // get GPU Time
    cudaEventElapsedTime(&GPUTime, start, stop);
    printf("GPU Time: %f ms.\n\n", GPUTime);
    
    // start CPU clock
    cudaEventRecord(start, 0);
    
    //Call cpu func
    multPC(mA, mB, mD, colA, rowB,colB);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&CPUTime, start, stop);

    printf("CPU Time: %f ms.\n\n", CPUTime);
    
    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(mA);
    cudaFreeHost(mB);
    cudaFreeHost(mC);
    cudaFreeHost(mD);
    
    return 0;
}
    