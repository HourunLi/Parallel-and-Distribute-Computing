#include<iostream>
#include<sys/time.h>
#include<stdlib.h>
#include<stdio.h>
#include<cuda.h>

#define N 32768
#define ITERATIONS 1
#define BLOCK_SIZE 32
#define VECTOR_SIZE 16
using namespace std;

__global__ void sgemm(float *A, float *B, float *C, int n, float a, float b) {
	/* Computation method optimization.
	 * Peform outer product instead of inner product to reduce  
	 * instructions from shared memory from two to one.
	 */
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	// Explicitly allocate As as column-major array 
	// to store TILE*TILE submatrix of A.
	__shared__ float As[BLOCK_SIZE * BLOCK_SIZE];

	// Allocate register files for sub-result of C at each thread.
	float cv[BLOCK_SIZE] = {0};

	// Basic iterations is similar with Tiling. But notice that 
	// the total number of threads is less than that of Tiling.
	int aBegin = N * BLOCK_SIZE * by;
	int aEnd = aBegin + N - 1;
	int aStep = BLOCK_SIZE;

	int bBegin = BLOCK_SIZE * VECTOR_SIZE * bx;
	int bStep = BLOCK_SIZE * N;

	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		// Load Asub with size of TILE*TILE in colomn-major style.
		// Each thread needs to load TILE_SIZE / VECTOR_SIZE values of A.
		int t = VECTOR_SIZE;
		for (int i = 0; i < BLOCK_SIZE / VECTOR_SIZE; ++i) {
			As[ (i*t+ty) + BLOCK_SIZE * tx] = A[a + N*(i*t+ty) + tx];
		}
		__syncthreads();

		float *ap = As;	// Point to the first address of As, increase later.
		float *bp = &B[b + BLOCK_SIZE * ty + tx];

#pragma unroll
		for (int i = 0; i < BLOCK_SIZE; ++i) {
			float bv = *bp;	
		// Each thread calculate a vector of C with size of TILE_SIZE.
			for (int j = 0; j < BLOCK_SIZE; ++j) {
				cv[j] += ap[j] * bv;
			}
			ap += BLOCK_SIZE;
			bp += N;
		}
		__syncthreads();
	}
	
	// Store each value of Csub back to C in global memory.
	int c = N * BLOCK_SIZE * by + BLOCK_SIZE * VECTOR_SIZE * bx;
	c += BLOCK_SIZE * ty + tx;
#pragma unroll
	for (int i = 0; i < BLOCK_SIZE; ++i) {
		C[c] = cv[i] * a + C[c] * b;
		c += N;
	}
}

void compare(float* res1, float* res2, int n){
  int fail=0;
  for(int i=0; i<n; i++){
    float a,b;
    if(res1[i]<0)
      a=res1[i]*(-1);
    else
      a=res1[i];
    if(res2[i]<0)
      b=res2[i]*(-1);
    else
      b=res2[i];
    if((a<0.01)&&(b<0.01)){
      continue;
    }
    if(i<10)
      printf("i=%d %lf %lf\n",i,a,b);
    float diff=(a-b)/(a+0.000001);
    if(diff<0)
      diff=diff*(-1);
    if(diff>0.0005)
      fail++;
  }
  printf("Number of errors: %d\n", fail);
}

double timestamp(){
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}
// float A[N*N], B[N*N]/*, C_cpu[N*N]*/, C_gpu_final[N*N];

int main(){
    float *A, *B, *C_gpu_final;
    A = (float*)malloc(sizeof(float)*N*N);
    B = (float*)malloc(sizeof(float)*N*N);
    C_gpu_final = (float*)malloc(sizeof(float)*N*N);
  //float A[N][N], B[N][N], C_cpu[N][N], C_gpu_final[N][N];
  float a=0.5, b=0.3;
  for(int i=0; i<N; i++){
    for(int j=0; j<N; j++){
      A[i*N+j]=(float)rand()/(float)(RAND_MAX/a);
      B[i*N+j]=(float)rand()/(float)(RAND_MAX/a);
    //   C_cpu[i*N+j]=0;
      C_gpu_final[i*N+j]=0;
    }
  }

//   for(int j=0; j<N; j++){
//     for(int i=0; i<N; i++){
//       C_cpu[i*N+j]+=b*C_cpu[i*N+j];
//       for(int k=0; k<N; k++){
//         C_cpu[i*N+j] += a*A[i*N+k]*B[k*N+j];
//       }
//     }
//   }

  float *A_gpu;
  float *B_gpu;
  float *C_gpu;
  cudaMalloc((void **)&A_gpu, sizeof(float)*N*N);
  cudaMalloc((void **)&B_gpu, sizeof(float)*N*N);
  cudaMalloc((void **)&C_gpu, sizeof(float)*N*N);
  cudaMemcpy(A_gpu, A, sizeof(float)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, sizeof(float)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(C_gpu, C_gpu_final, sizeof(float)*N*N, cudaMemcpyHostToDevice);

  dim3 block(BLOCK_SIZE, VECTOR_SIZE);
  dim3 grid((size_t)ceil((float)N / (BLOCK_SIZE*VECTOR_SIZE)), 
            (size_t)ceil((float)N / BLOCK_SIZE));

//   sgemm<<<grid,block>>>(A_gpu, B_gpu, C_gpu, N, a, b);
//   cudaThreadSynchronize();
//   cudaMemcpy(C_gpu_final, C_gpu, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
//   compare(C_cpu, C_gpu_final, N*N);

  double time1=timestamp();
  for(int numOfTimes=0; numOfTimes<ITERATIONS; numOfTimes++){
    sgemm<<<grid,block>>>(A_gpu, B_gpu, C_gpu, N, a, b);
  }
  cudaThreadSynchronize();
  double time2=timestamp();

  double time = (time2-time1)/ITERATIONS;
  double flops = 2.0*N*N*N;
  double gflopsPerSecond = flops/(1000000000)/time;
  double GB = (double)(N)*N*4/1000000000;
  double GBpS = (double)(N)*N*4/1000000000/time;
  printf("GFLOPS/s=%lf\n",gflopsPerSecond );
  printf("GB/s=%lf\n",GBpS);
  printf("GFLOPS=%lf\n",flops/(1000000000));
  printf("GB=%lf\n",GB);
  printf("time(s)=%lf\n",time);


  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);
  free(A);
  free(B);
  free(C_gpu_final);
  return 0;
}
