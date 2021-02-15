#include <cuda_runtime.h>
#include "obj_reader.h"
#include "light_model.h"
#include "kernel.h"

__device__ double det_2x2_device(double A[2], double B[2]) {
    //            |A[0] B[0]|
    //Returns det |A[1] B[1]|
    double det = A[0] * B[1] - A[1] * B[0];
    return det;
}

__device__ double det_3x3_device(double A[3], double B[3], double C[3]) {
    //            |A[0] B[0] C[0]|
    //Returns det |A[1] B[1] C[1]|
    //            |A[2] B[2] C[2]|
    double det = 0;
    double tempA[2], tempB[2];
    tempA[0] = B[1]; tempA[1] = B[2];
    tempB[0] = C[1]; tempB[1] = C[2];
    det += A[0] * det_2x2_device(tempA, tempB);
    tempA[0] = A[1]; tempA[1] = A[2];
    tempB[0] = C[1]; tempB[1] = C[2];
    det += -B[0] * det_2x2_device(tempA, tempB);
    tempA[0] = A[1]; tempA[1] = A[2];
    tempB[0] = B[1]; tempB[1] = B[2];
    det += C[0] * det_2x2_device(tempA, tempB);
    return det;
}

__global__ void intersect_single_triangle_device(double S[3], double E[3], double* uv, double* dist, Triangle* tris, double* x, double* y, double* z, int* num_tris) {
    //Calculate global thread ID
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    //Boundary check
    if (i >= *num_tris) return;

    //Offset for the index of UV-values
    int uv_offset = i * 2;

    Triangle tri = tris[i];

    double AB[3], AC[3], ES[3], AS[3];
    double A[3], B[3], C[3];
    double t;

    //Get the indeces of the vertices in the triangle
    int index_A = tri.A;
    int index_B = tri.B;
    int index_C = tri.C;

    //Get the positions of the vertices from corresponding index
    A[0] = x[index_A]; A[1] = y[index_A]; A[2] = z[index_A];
    B[0] = x[index_B]; B[1] = y[index_B]; B[2] = z[index_B];
    C[0] = x[index_C]; C[1] = y[index_C]; C[2] = z[index_C];

    //Find vectors between points in triangle, 
    //as well as with start and end of ray
    AB[0] = B[0] - A[0]; AB[1] = B[1] - A[1]; AB[2] = B[2] - A[2];
    AC[0] = C[0] - A[0]; AC[1] = C[1] - A[1]; AC[2] = C[2] - A[2];
    ES[0] = S[0] - E[0]; ES[1] = S[1] - E[1]; ES[2] = S[2] - E[2];
    AS[0] = S[0] - A[0]; AS[1] = S[1] - A[1]; AS[2] = S[2] - A[2];


    double den = det_3x3_device(AB, AC, ES);

    if (den == 0) {
        dist[i] = -1;
        return;
    }
    double topt, topu, topv;
    topt = det_3x3_device(AB, AC, AS);
    t = topt / den;
    if (t < 0) {
        dist[i] = -1;
        return;
    }
    topu = det_3x3_device(AS, AC, ES);
    uv[uv_offset + 0] = topu / den;

    if ((uv[uv_offset + 0] < 0) || (uv[uv_offset + 0] > 1)) {
        dist[i] = -1;
        return;
    }
    topv = det_3x3_device(AB, AS, ES);
    uv[uv_offset + 1] = topv / den;
    if ((uv[uv_offset + 1] < 0) || (uv[uv_offset + 1] > 1)) {
        dist[i] = -1;
        return;
    }
    if (uv[uv_offset + 0] + uv[uv_offset + 1] > 1) {
        dist[i] = -1;
        return;
    }
    dist[i] = t;
}

int intersect_all_triangles_device(double S[3], double E[3],
    double uvt[3], double point[3], double normal[3], double obinv[4][4])
{
    //Allocate device memory for distance array
    size_t bytes = sizeof(double) * num_tris;
    double* d_dist;
    cudaMalloc(&d_dist, bytes);
    
    int i;
    int closest = -1;
    double tempUV[2];
    uvt[2] = 1e50;

    //Copy ray and UV array to device
    double* d_S, * d_E, * d_UV;
    cudaMalloc(&d_S, 3 * sizeof(double));
    cudaMalloc(&d_E, 3 * sizeof(double));
    
    cudaMalloc(&d_UV, num_tris * 2 * sizeof(double));

    cudaMemcpy(d_S, S, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E, E, 3 * sizeof(double), cudaMemcpyHostToDevice);

    //Copy triangle data to device
    Triangle* d_tris;
    cudaMalloc(&d_tris, num_tris * sizeof(Triangle));
    cudaMemcpy(d_tris, tris, num_tris * sizeof(Triangle), cudaMemcpyHostToDevice);

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

    //Copy number of triangles to device
    int* dev_num_tris;
    cudaMalloc(&dev_num_tris, 2*sizeof(int));
    cudaMemcpy(dev_num_tris, &num_tris, sizeof(int), cudaMemcpyHostToDevice);

    //Threads per CTA (1024)
    int NUM_THREADS = 1 << 10;

    //CTAs per grid
    int NUM_BLOCKS = (num_tris + NUM_THREADS - 1) / NUM_THREADS;

    intersect_single_triangle_device << <NUM_BLOCKS, NUM_THREADS >> > (d_S, d_E, d_UV, d_dist, d_tris, dev_x, dev_y, dev_z, dev_num_tris);

    //Free device memory
    cudaFree(d_tris);
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_z);
    cudaFree(dev_num_tris);

    //Copy result from device to host
    double* dist;
    dist = (double*)malloc(bytes);
    cudaMemcpy(dist, d_dist, bytes, cudaMemcpyDeviceToHost);
   
    //Free device memory
    cudaFree(d_dist);
    
    for (i = 0; i < num_tris; i++) {
        //Find the closest intersection
        if ((dist[i] > 0) && (dist[i] < uvt[2])) {
            //Copy UV array from device to host
            cudaMemcpy(tempUV, &d_UV[i * 2], 2 * sizeof(double), cudaMemcpyDeviceToHost);
            uvt[2] = dist[i];
            uvt[0] = tempUV[0];
            uvt[1] = tempUV[1];
            closest = i;

        }
    }
    
    free(dist);
    cudaFree(d_S);
    cudaFree(d_E);
    cudaFree(d_UV);

    if (closest != -1) {
        //Load point with coordinates of intersection between ray and object
        point[0] = S[0] + uvt[2] * (E[0] - S[0]);
        point[1] = S[1] + uvt[2] * (E[1] - S[1]);
        point[2] = S[2] + uvt[2] * (E[2] - S[2]);


        //Find normal vector information at the closest triangle
        double An[3], Bn[3], Cn[3];
        int index_A = tris[closest].An;
        int index_B = tris[closest].Bn;
        int index_C = tris[closest].Cn;

        An[0] = xnormal[index_A];
        An[1] = ynormal[index_A];
        An[2] = znormal[index_A];

        Bn[0] = xnormal[index_B];
        Bn[1] = ynormal[index_B];
        Bn[2] = znormal[index_B];

        Cn[0] = xnormal[index_C];
        Cn[1] = ynormal[index_C];
        Cn[2] = znormal[index_C];

        //Interpolate between the normal at each vertex to find normal at intersection point
        interpolate_normal_vector(normal, An, Bn, Cn, uvt, obinv);
    }
    return closest;
}

/*
__global__ void intersect_single_triangle(double S[3], double E[3], double *uv, double *dist, Triangle *tris, double *x, double *y, double *z, int* num_tris) {
    //Calculate global thread ID

    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    //Boundary check
    if (i >= *num_tris) return;

    //Offset for the index of UV-values
    int uv_offset = i * (*num_tris);

    Triangle tri = tris[i];

    double AB[3], AC[3], ES[3], AS[3];
    double A[3], B[3], C[3];
    double t;

    //Get the indeces of the vertices in the triangle
    int index_A = tri.A;
    int index_B = tri.B;
    int index_C = tri.C;

    //Get the positions of the vertices from corresponding index
    A[0] = x[index_A]; A[1] = y[index_A]; A[2] = z[index_A];
    B[0] = x[index_B]; B[1] = y[index_B]; B[2] = z[index_B];
    C[0] = x[index_C]; C[1] = y[index_C]; C[2] = z[index_C];

    //Find vectors between points in triangle, 
    //as well as with start and end of ray
    AB[0] = B[0] - A[0]; AB[1] = B[1] - A[1]; AB[2] = B[2] - A[2];
    AC[0] = C[0] - A[0]; AC[1] = C[1] - A[1]; AC[2] = C[2] - A[2];
    ES[0] = S[0] - E[0]; ES[1] = S[1] - E[1]; ES[2] = S[2] - E[2];
    AS[0] = S[0] - A[0]; AS[1] = S[1] - A[1]; AS[2] = S[2] - A[2];


    double den = det_3x3_device(AB, AC, ES);

    if (den == 0) {
        dist[i] = -1;
        return;
    }
    double topt, topu, topv;
    topt = det_3x3_device(AB, AC, AS);
    t = topt / den;
    if (t < 0) {
        dist[i] = -1;
        return;
    }
    topu = det_3x3_device(AS, AC, ES);
    uv[uv_offset + 0] = topu / den;

    if ((uv[uv_offset + 0] < 0) || (uv[uv_offset + 0] > 1)) {
        dist[i] = -1;
        return;
    }
    topv = det_3x3_device(AB, AS, ES);
    uv[uv_offset + 1] = topv / den;
    if ((uv[uv_offset + 1] < 0) || (uv[uv_offset + 1] > 1)) {
        dist[i] = -1;
        return;
    }
    if (uv[uv_offset + 0] + uv[uv_offset + 1] > 1) {
        dist[i] = -1;
        return;
    }
    dist[i] = t;
}

int intersect_all_triangles_device(double S[3], double E[3],
    double uvt[3], double point[3], double normal[3], double obinv[4][4])
{
    //Allocate device memory for distance array
    size_t bytes = sizeof(double) * num_tris;
    double* d_dist;
    cudaMalloc(&d_dist, bytes);


    int i;
    int closest = -1;
    double tempUV[2];
    uvt[2] = 1e50;



    //Copy ray and UV array to device
    double* d_S, * d_E, * d_UV;
    cudaMalloc(&d_S, 3 * sizeof(double));
    cudaMalloc(&d_E, 3 * sizeof(double));
    cudaMalloc(&d_UV, num_tris * 2 * sizeof(double));

    cudaMemcpy(d_S, S, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E, E, 3 * sizeof(double), cudaMemcpyHostToDevice);

    //Copy triangle data to device
    Triangle* d_tris;
    cudaMalloc(&d_tris, num_tris * sizeof(Triangle));
    cudaMemcpy(d_tris, tris, num_tris * sizeof(Triangle), cudaMemcpyHostToDevice);

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

    //Copy number of triangles to device
    int* dev_num_tris;
    cudaMalloc(&dev_num_tris, sizeof(int));
    cudaMemcpy(dev_num_tris, &num_tris, sizeof(int), cudaMemcpyHostToDevice);


    //Threads per CTA (1024)
    int NUM_THREADS = 1 << 10;

    //CTAs per grid
    int NUM_BLOCKS = (num_tris + NUM_THREADS - 1) / NUM_THREADS;

    intersect_single_triangle << <NUM_BLOCKS, NUM_THREADS >> > (d_S, d_E, d_UV, d_dist, d_tris, dev_x, dev_y, dev_z, dev_num_tris);

    //Free device memory
    cudaFree(d_tris);
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_z);
    cudaFree(dev_num_tris);

    //Copy result from device to host
    double* dist;
    dist = (double*)malloc(bytes);
    cudaMemcpy(dist, d_dist, bytes, cudaMemcpyDeviceToHost);

    //Free device memory
    cudaFree(d_dist);

    for (i = 0; i < num_tris; i++) {
        //Find the closest intersection
        if ((dist[i] > 0) && (dist[i] < uvt[2])) {
            //Copy UV array from device to host
            cudaMemcpy(tempUV, &d_UV[i * num_tris], 2 * sizeof(double), cudaMemcpyDeviceToHost);
            uvt[2] = dist[i];
            uvt[0] = tempUV[0];
            uvt[1] = tempUV[1];
            closest = i;

        }
    }

    free(dist);
    cudaFree(d_S);
    cudaFree(d_E);
    cudaFree(d_UV);

    if (closest != -1) {
        //Load point with coordinates of intersection between ray and object
        point[0] = S[0] + uvt[2] * (E[0] - S[0]);
        point[1] = S[1] + uvt[2] * (E[1] - S[1]);
        point[2] = S[2] + uvt[2] * (E[2] - S[2]);


        //Find normal vector information at the closest triangle
        double An[3], Bn[3], Cn[3];
        int index_A = tris[closest].An;
        int index_B = tris[closest].Bn;
        int index_C = tris[closest].Cn;

        An[0] = xnormal[index_A];
        An[1] = ynormal[index_A];
        An[2] = znormal[index_A];

        Bn[0] = xnormal[index_B];
        Bn[1] = ynormal[index_B];
        Bn[2] = znormal[index_B];

        Cn[0] = xnormal[index_C];
        Cn[1] = ynormal[index_C];
        Cn[2] = znormal[index_C];

        //Interpolate between the normal at each vertex to find normal at intersection point
        interpolate_normal_vector(normal, An, Bn, Cn, uvt, obinv);
    }
    return closest;
}*/