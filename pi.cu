// Using CUDA device to calculate pi
#include <stdio.h>
#include <cuda.h>

#define NBIN  10000000  // Number of bins
#define NUM_BLOCK   13  // Number of thread blocks
#define NUM_THREAD 192  // Number of threads per block
#define NUM_STEPS = 10000000;
int tid;
float pi = 0;

// Kernel that executes on the CUDA device
__global__ void cal_pi(float *sum, int nbin, float step, int nthreads, int nblocks) {
	int i;
	float x;
	int idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the blocks
	for (i=idx; i< nbin; i+=nthreads*nblocks) {
		x = (i+0.5)*step;
		sum[idx] += 4.0/(1.0+x*x);
	}
}

// Main routine that executes on the host
int main(void) {
	dim3 dimGrid(NUM_BLOCK,1,1);  // Grid dimensions
	dim3 dimBlock(NUM_THREAD,1,1);  // Block dimensions
	float *sumHost, *sumDev,*sumCPU;  // Pointer to host & device arrays

	float step = 1.0/NBIN;  // Step size
	size_t size = NUM_BLOCK*NUM_THREAD*sizeof(float);  //Array memory size
    sumHost = (float *)malloc(size);  //  Allocate array on host
    sumCPU = (float *)malloc(size);  //  Allocate array on host
	cudaMalloc((void **) &sumDev, size);  // Allocate array on device
	// Initialize array in device to 0
    cudaMemset(sumDev, 0, size);
    printf("Doing GPU Vector add\n");
    clock_t start_d=clock();
	// Do calculation on device
    cal_pi <<<dimGrid, dimBlock>>> (sumDev, NBIN, step, NUM_THREAD, NUM_BLOCK); // call CUDA kernel
    clock_t end_d = clock();
	// Retrieve result from device and store it in host array
	cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);
	for(tid=0; tid<NUM_THREAD*NUM_BLOCK; tid++)
		pi += sumHost[tid];
    pi *= step;


    clock_t start_h=clock();
	// Do calculation on device
    //cal_pi_h(sumCPU,step);
    int i_h;
    float x_h;
    for (i_h=1;i<= NUM_STEPS; i_h++){
        x_h = (i-0.5)*step;
        sumCPU = sumCPU + 4.0/(1.0+x*x);
    }
    clock_t end_h = clock();

    //Time computing
    double time_d = (double)(end_d-start_d)/CLOCKS_PER_SEC;
    double time_h = (double)(end_h-start_h)/CLOCKS_PER_SEC;

	// Print results
    printf("PI = %f \t GPU time = %fs \t CPU time = %fs\n", pi, time_d, time_h);

	// Cleanup
	free(sumHost); 
	cudaFree(sumDev);

	return 0;
}