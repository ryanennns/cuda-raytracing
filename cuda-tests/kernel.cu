#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include "src/Vector3D.cu"
#include "src/Rgb.cu"

const int N = 3;

__global__ void kernel(Vector3D* A, Vector3D* B) {
    int idx = threadIdx.x;


    B[idx].x = A[idx].x * 2.0;
    B[idx].y = A[idx].y * 2.0;
    B[idx].z = A[idx].z * 2.0;

    printf("%d:  %.2lf, %.2lf, %.2lf\n", idx, B[idx].x, B[idx].y, B[idx].z);
}

int main() {
    Vector3D A[N] = {Vector3D(1,1,1), Vector3D(2,2,2), Vector3D(3,3,3)};
    Vector3D B[N];

    Vector3D tester = Vector3D();

    Vector3D* d_A = nullptr;
    Vector3D* d_B = nullptr;

    cudaMalloc(&d_A, sizeof(A));
    cudaMalloc(&d_B, sizeof(B));

    cudaMemcpy(d_A, A, sizeof(A), cudaMemcpyHostToDevice);

    kernel<<<1, N>>> (d_A, d_B);

    cudaDeviceSynchronize();

    cudaMemcpy(B, d_B, sizeof(B), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("\n");
		printf("%d.x:  %.2lf\n", i, B[i].x);
		printf("%d.y:  %.2lf\n", i, B[i].y);
		printf("%d.z:  %.2lf\n", i, B[i].z);
	}

    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}