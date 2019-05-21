#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#define XINICIAL 0
#define XFINAL 10
#define PASOS 1000000000

int *res;  // host data
int *c;  // results

//Cuda error checking - non mandatory
void cudaCheckError() {
 cudaError_t e=cudaGetLastError();
 if(e!=cudaSuccess) {
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
   exit(0); 
 }
}

float calcularFuncion(float x)
{
    return x * x;
}

//GPU kernel 
__global__
void vecAdd(int *res,int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i] + B[i];
}

//CPU function
void integral_h(float *res, int N){
    float incremento = ((float)XFINAL - (float)XINICIAL) / (float)PASOS;
    float x = XINICIAL;
    for (int i = 0; i < PASOS; i++)
    {
        res += incremento * (calcularFuncion(x) + calcularFuncion(x + incremento)) / (float)2;
        x += incremento;
    }
}

int main(int argc,char **argv)
{
    printf("Begin \n");
    //Iterations
    int n=1000000000;
    //Number of blocks
    int nBytes = n*sizeof(float);
    //Block size and number
    int block_size, block_no;

    //memory allocation	
    res = (float *) malloc(nBytes);
    c = (float *) malloc(nBytes);

    int *a_d,*b_d,*c_d;
    block_size = 250; //threads per block
    block_no = n/block_size;
        
    //Work definition
    dim3 dimBlock(block_size, 1, 1);
    dim3 dimGrid(block_no, 1, 1);

    // Data filling
    for(int i=0;i<n;i++)
    a[i]=i,b[i]=i;


    printf("Allocating device memory on host..\n");
   //GPU memory allocation
    cudaMalloc((void **) &a_d, n*sizeof(float));
    cudaMalloc((void **) &b_d, n*sizeof(float));
    cudaMalloc((void **) &c_d, n*sizeof(float));

    printf("Copying to device..\n");
    cudaMemcpy(a_d, a, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, n*sizeof(float), cudaMemcpyHostToDevice);

    clock_t start_d=clock();
    printf("Doing GPU Vector add\n");
    vecAdd<<<block_no,block_size>>>(a_d, b_d, c_d, n);
    cudaCheckError();

    //Wait for kernel call to finish
    cudaThreadSynchronize();

    clock_t end_d = clock();
    

    printf("Doing CPU Vector add\n");
    clock_t start_h = clock();
    vecAdd_h(a, b, c2, n);
    clock_t end_h = clock();
	
    //Time computing
    double time_d = (double)(end_d-start_d)/CLOCKS_PER_SEC;
    double time_h = (double)(end_h-start_h)/CLOCKS_PER_SEC;

    //Copying data back to host, this is a blocking call and will not start until all kernels are finished
    cudaMemcpy(c, c_d, n*sizeof(int), cudaMemcpyDeviceToHost);
    printf("n = %d \t GPU time = %fs \t CPU time = %fs\n", n, time_d, time_h);

    //Free GPU memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    return 0;
}
