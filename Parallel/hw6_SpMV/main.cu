#include<iostream>
#include<sys/time.h>
#include<stdlib.h>
#include<stdio.h>
#include<cassert>
#include "spmv_util.h"
using namespace std;

#define ITERATIONS 10
#define DIM_THREAD_BLOCK_X 256
#define WARPSIZE 32
#define DIM_THREAD_BLOCK_Y 1
// ! change your method
#define METHOD3
using namespace std;

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
//The CSR-format matrix is dimXdim that has n non-zero elements.
/**
 * @param row: therowptr
 * @param col: the column index of the non-zero element
 * @param data: the source data
 * @param vec: the vector to be multiplied
 * @param res
 * @param dim: the height of the matrix
 * @param n: the number of the non zero elements
 */

// method 1 unoptimized
#ifdef METHOD1
__global__ void spmv(int* row, int* col, float* data, float* vec, float* res, int dim, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<dim){
    float tmp = 0;
    for(int j=row[i]; j<row[i+1]; j++){
      int colTmp = col[j];
      tmp +=  data[j] * vec[colTmp];
    }
    res[i] = tmp;
  }
}
#endif

/**
  * Every warp tackles a single row
  * Every single per block
  */
#ifdef METHOD2
__global__ void spmv(int* row, int* col, float* data, float* vec, float* res, int dim, int n){
  //thread id and block id
  int tid = threadIdx.x, bid = blockIdx.x;
  //begin and end position
  int begin = row[bid], end =row[bid+1];
  //the num of iterations
  int iterations = ceil((float)(end - begin) / WARPSIZE);

  float s = 0;
#pragma unroll
  for(int i = begin; i < end; i += WARPSIZE) {
    int pos = i + tid;
    if(pos < end) {
      int colTmp = col[pos];
      s += data[pos] * vec[colTmp];
    }
  }

#pragma unroll
  for (int offset = WARPSIZE / 2; offset > 0; offset /= 2){
    s += __shfl_down_sync(0xffffffff, s, offset);
  }
  if(!tid)
    res[bid] = s;
}
#endif

/**
  * Every warp tackles a single row
  * Several single per block
  * This version is the most optimized
  */
#ifdef METHOD3
__global__ void spmv(int* row, int* col, float* data, float* vec, float* res, int dim, int n){
  //thread id
  int tid = threadIdx.x;
  //thread id in global
  int gtid = blockIdx.x * blockDim.x + tid;
  //thread id in a warp
  int wtid = tid % WARPSIZE;
  //the index th row of matrix to be multiplied
  int index = gtid / WARPSIZE;
  //begin and end position
  int begin = row[index], end =row[index+1];
  //the num of iterations
  int iterations = ceil((float)(end - begin) / WARPSIZE);

  if(index < dim) {
    float s = 0;
    int colTmp;
#pragma unroll
    for(int i = begin; i < end; i += WARPSIZE) {
      int pos = i + wtid;
      if(pos < end) {
        colTmp = col[pos];
        s += data[pos] * vec[colTmp];
      }
    }
#pragma unroll
    for (int offset = WARPSIZE / 2; offset > 0; offset /= 2){
      s += __shfl_down_sync(0xffffffff, s, offset);
    }
    if(!wtid)
      res[index] = s;
  }
}
#endif 

int main(int argc, char **argv){
  if (argc < 2){
	printf("\nUsage: ./main *.mtx\n");
        return 1;
  }

  char* filename = argv[1];
  coo_matrix<int, float> mat;
  init_coo_matrix(mat);
  ReadMMF(filename, &mat);

  csr_matrix<int, float> csrmat;
  coo2csr<int, float>(&mat, &csrmat);
  free_coo_matrix(mat);
  printf("\nMatInfo: Width %d Height %d NNZ %d\n", csrmat.matinfo.width, csrmat.matinfo.height, csrmat.matinfo.nnz);
  // free_csr_matrix(csrmat_tmp);
  // int dim=20000;
  // int n=dim*dim/100;
  // int *row = (int*)malloc(sizeof(int)*(dim+1));
  // int *col = (int*)malloc(sizeof(int)*n);
  // float *data = (float*)malloc(sizeof(float)*n);
  // initMatrix(row, col, data, n, dim);
  int dim = csrmat.matinfo.height;
  int n = csrmat.matinfo.nnz;
  float *vec = (float*)malloc(sizeof(float)*dim);
  for(int i=0; i<dim; i++){
    vec[i]=1;
  }

  float *result = (float*)malloc(sizeof(float)*dim);
  float *result_gpu_res = (float*)malloc(sizeof(float)*dim);
  
  // for(int i=0; i<dim; i++){
  //   float t = 0;
  //   for(int j=row[i]; j<row[i+1]; j++){
  //     int colNum = col[j];
  //     t += data[j] * vec[colNum];
  //   }
  //   result[i] = t;
  // }
  for(int i = 0; i < dim; i++){
    float t = 0;
    for(int j = csrmat.csr_row_ptr[i]; j < csrmat.csr_row_ptr[i+1]; j++){
      int colNum = csrmat.csr_col_id[j];
      t += csrmat.csr_data[j] * vec[colNum];
    }
    result[i] = t;
  }

  int *row_gpu;
  int *col_gpu;
  float *data_gpu;
  float *vec_gpu;
  float *result_gpu;
  cudaMalloc( (void **)&row_gpu, sizeof(int)*(dim+1));
  cudaMalloc( (void **)&col_gpu, sizeof(int)*n);
  cudaMalloc( (void **)&data_gpu, sizeof(float)*n);
  cudaMalloc( (void **)&vec_gpu, sizeof(float)*dim);
  cudaMalloc( (void **)&result_gpu, sizeof(float)*dim);
  // cudaMemcpy(row_gpu, row, sizeof(int)*(dim+1), cudaMemcpyHostToDevice);
  // cudaMemcpy(col_gpu, col, sizeof(int)*n, cudaMemcpyHostToDevice);
  // cudaMemcpy(data_gpu, data, sizeof(float)*n, cudaMemcpyHostToDevice);
  // cudaMemcpy(vec_gpu, vec, sizeof(float)*dim, cudaMemcpyHostToDevice);
  cudaMemcpy(row_gpu, csrmat.csr_row_ptr, sizeof(int)*(dim+1), cudaMemcpyHostToDevice);
  cudaMemcpy(col_gpu, csrmat.csr_col_id, sizeof(int)*n, cudaMemcpyHostToDevice);
  cudaMemcpy(data_gpu, csrmat.csr_data, sizeof(float)*n, cudaMemcpyHostToDevice);
  cudaMemcpy(vec_gpu, vec, sizeof(float)*dim, cudaMemcpyHostToDevice);

  // ! using for method 1 unoptimized
  #ifdef METHOD1
  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((size_t)ceil( (float)dim / (DIM_THREAD_BLOCK_X) ), 1);
  #endif

  #ifdef METHOD2
  dim3 block(WARPSIZE, DIM_THREAD_BLOCK_Y);
  dim3 grid(dim, 1);
  #endif

  #ifdef METHOD3
  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((size_t)ceil( (float)dim / (DIM_THREAD_BLOCK_X/WARPSIZE) ), 1);
  #endif

  spmv<<<grid,block>>>(row_gpu, col_gpu, data_gpu, vec_gpu, result_gpu, dim, n);
  cudaThreadSynchronize();
  cudaMemcpy(result_gpu_res, result_gpu, sizeof(float)*dim, cudaMemcpyDeviceToHost);
  compare(result, result_gpu_res, dim);

  double time1=timestamp();
  for(int numOfTimes=0; numOfTimes<ITERATIONS; numOfTimes++){

    spmv<<<grid,block>>>(row_gpu, col_gpu, data_gpu, vec_gpu, result_gpu, dim, n);

  }
  cudaThreadSynchronize();
  double time2=timestamp();

  double time = (time2-time1)/ITERATIONS;
  double flops = 2 * (double)n;
  double gflopsPerSecond = flops/(1000000000)/time;
  double dataCopy = sizeof(int)*dim + sizeof(int)*n + sizeof(float)*n + sizeof(float)*dim*2;
  double bandwidth = dataCopy/time/1000000000;
  printf("GFLOPS/s=%lf\n",gflopsPerSecond );
  printf("GB/s=%lf\n",bandwidth );
  printf("GB=%lf\n",dataCopy/1000000000);
  printf("GFLOPS=%lf\n",flops/(1000000000));
  printf("time(s)=%lf\n",time);
  return 0;
}
