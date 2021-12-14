#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <immintrin.h>

#define N 32768
#define BLOCK_SIZE 256
#define BLOCKS (N / BLOCK_SIZE)
#define AVX_ALIGNED __attribute__((aligned(64)))

static_assert(N == BLOCK_SIZE * BLOCKS);

#define ITERATIONS 10
using namespace std;

double timestamp() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

void yourFunction(float a, float b, float A[N][N], float B[N][N], float C[N][N]) {
  typedef float block_array_t[BLOCKS][BLOCKS][BLOCK_SIZE][BLOCK_SIZE];
  static block_array_t bA AVX_ALIGNED;
  static block_array_t ibB AVX_ALIGNED;
  static block_array_t bC AVX_ALIGNED;

#pragma omp parallel for schedule(guided) collapse(2)
  for (int iBlock = 0; iBlock < BLOCKS; iBlock++) {
    for (int jBlock = 0; jBlock < BLOCKS; jBlock++) {
      int iBase = iBlock * BLOCK_SIZE;
      int jBase = jBlock * BLOCK_SIZE;
      for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
          bA[iBlock][jBlock][i][j] = A[iBase + i][jBase + j];
          ibB[iBlock][jBlock][i][j] = B[jBase + j][iBase + i];
          bC[iBlock][jBlock][i][j] = C[iBase + i][jBase + j] * b;
        }
      }
    }
  }

#pragma omp parallel for schedule(dynamic, 1) collapse(2)
  for (int iBlock = 0; iBlock < BLOCKS; iBlock++) {
    for (int jBlock = 0; jBlock < BLOCKS; jBlock++) {
      for (int kBlock = 0; kBlock < BLOCKS; kBlock++) {
        for (int i = 0; i < BLOCK_SIZE; i++) {
          for (int j = 0; j < BLOCK_SIZE; j++) {
            float s = 0;
            for (int k = 0; k < BLOCK_SIZE; k++) {
              s += bA[iBlock][kBlock][i][k] * ibB[jBlock][kBlock][j][k];
            }

            bC[iBlock][jBlock][i][j] += s * a;
          }
        }
      }
    }
  }

#pragma omp parallel for schedule(guided) collapse(2)
  for (int iBlock = 0; iBlock < BLOCKS; iBlock++) {
    for (int jBlock = 0; jBlock < BLOCKS; jBlock++) {
      int iBase = iBlock * BLOCK_SIZE;
      int jBase = jBlock * BLOCK_SIZE;
      for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
          C[iBase + i][jBase + j] = bC[iBlock][jBlock][i][j];
        }
      }
    }
  }
}

void validate(float C[N][N], float myC[N][N]) {
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      if (fabs(C[i][j] - myC[i][j]) > 1e-4)
        printf("%f %f\n", C[i][j], myC[i][j]);
}

int main() {
  static float A[N][N] AVX_ALIGNED;
  static float B[N][N] AVX_ALIGNED;
  static float C[N][N] AVX_ALIGNED;
  const float a = 0.5, b = 0.3;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = (float)rand() / (float)(RAND_MAX / a);
      B[i][j] = (float)rand() / (float)(RAND_MAX / a);
      C[i][j] = 0;
    }
  }

  // for (int j = 0; j < N; j++) {
  //   for (int i = 0; i < N; i++) {
  //     C[i][j] += b * C[i][j];
  //     float tmp = 0;
  //     for (int k = 0; k < N; k++) {
  //       // C[i][j] += a*A[i][k]*B[k][j];
  //       tmp += A[i][k] * B[k][j];
  //     }
  //     C[i][j] += tmp * a;
  //   }
  // }

  static float myC[N][N];
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      myC[i][j] = 0;
    }
  }

  double time1 = timestamp();
  for (int numOfTimes = 0; numOfTimes < ITERATIONS; numOfTimes++) {
    yourFunction(a, b, A, B, myC);
    // if (numOfTimes == 0) validate(C, myC);
  }
  double time2 = timestamp();

  double time = (time2 - time1) / ITERATIONS;
  double flops = 2.0 * N * N + 2.0 * N * N * N + 2.0 * N * N;
  // double flops = 2*N*N + 2*N*N*N + N*N*N;
  double gflopsPerSecond = flops / (1000000000) / time;
  printf("GFLOPS/s=%lf\n", gflopsPerSecond);
  printf("GFLOPS=%lf\n", flops / (1000000000));
  printf("time(s)=%lf\n", time);
  return 0;
}
