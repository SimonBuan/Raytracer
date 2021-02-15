#include <stdlib.h>
#include <cuda_runtime.h>

#include "kernel.h"


//Matrix multiplication CUDA GPU function
__global__ void mat_mult_points_kernel(
	double* m,
	double* x, double* y, double* z,
	int numpoints)
{
	double u, v, w;

	//Calculate global thread ID
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	//Boundary check
	if (i < numpoints) {
		u = m[0] * x[i] + m[1] * y[i] + m[2] * z[i] + m[3];
		v = m[4] * x[i] + m[5] * y[i] + m[6] * z[i] + m[7];
		w = m[8] * x[i] + m[9] * y[i] + m[10] * z[i] + m[11];

		x[i] = u;
		y[i] = v;
		z[i] = w;
	}
}

//Host CPU matrix multiplication thread spawner
void mat_mult_device(
	double* X, double* Y, double* Z,
	double m[4][4],
	double* x, double* y, double* z,
	int numpoints)
{
	size_t bytes = numpoints * sizeof(double);
	size_t mat_bytes = 16 * sizeof(double);

	//Allocates memory and copies contents of m onto device
	double* dev_m;
	cudaMalloc(&dev_m, mat_bytes);
	cudaMemcpy(dev_m, m, mat_bytes, cudaMemcpyHostToDevice);

	double* dev_x;
	double* dev_y;
	double* dev_z;

	//Allocate device memory
	cudaMalloc(&dev_x, bytes);
	cudaMalloc(&dev_y, bytes);
	cudaMalloc(&dev_z, bytes);

	//Copy data to device
	cudaMemcpy(dev_x, x, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, y, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_z, z, bytes, cudaMemcpyHostToDevice);

	//Threads per CTA (1024)
	int NUM_THREADS = 1 << 10;

	//CTAs per grid
	int NUM_BLOCKS = (numpoints + NUM_THREADS - 1) / NUM_THREADS;

	mat_mult_points_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(dev_m, dev_x, dev_y, dev_z, numpoints);

	//Copy data back from device to host
	cudaMemcpy(X, dev_x, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(Y, dev_y, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(Z, dev_z, bytes, cudaMemcpyDeviceToHost);

	//Free device memory
	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_z);
}
