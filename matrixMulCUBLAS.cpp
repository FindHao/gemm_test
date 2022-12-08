/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
* Matrix multiplication: C = A * B.
* Host code.
*
* This sample implements matrix multiplication as described in Chapter 3
* of the programming guide and uses the CUBLAS library to demonstrate
* the best performance.

* SOME PRECAUTIONS:
* IF WE WANT TO CALCULATE ROW-MAJOR MATRIX MULTIPLY C = A * B,
* WE JUST NEED CALL CUBLAS API IN A REVERSE ORDER: cublasSegemm(B, A)!
* The reason is explained as follows:

* CUBLAS library uses column-major storage, but C/C++ use row-major storage.
* When passing the matrix pointer to CUBLAS, the memory layout alters from
* row-major to column-major, which is equivalent to an implicit transpose.

* In the case of row-major C/C++ matrix A, B, and a simple matrix multiplication
* C = A * B, we can't use the input order like cublasSgemm(A, B)  because of
* implicit transpose. The actual result of cublasSegemm(A, B) is A(T) * B(T).
* If col(A(T)) != row(B(T)), equal to row(A) != col(B), A(T) and B(T) are not
* multipliable. Moreover, even if A(T) and B(T) are multipliable, the result C
* is a column-based cublas matrix, which means C(T) in C/C++, we need extra
* transpose code to convert it to a row-based C/C++ matrix.

* To solve the problem, let's consider our desired result C, a row-major matrix.
* In cublas format, it is C(T) actually (because of the implicit transpose).
* C = A * B, so C(T) = (A * B) (T) = B(T) * A(T). Cublas matrice B(T) and A(T)
* happen to be C/C++ matrice B and A (still because of the implicit transpose)!
* We don't need extra transpose code, we only need alter the input order!
*
* CUBLAS provides high-performance matrix multiplication.
* See also:
* V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
* in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
* Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
*/

// Utilities and system includes
#include <assert.h>
#include <helper_string.h> // helper for shared functions common to CUDA Samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

#ifndef min
#define min(a, b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a, b) ((a > b) ? a : b)
#endif

// Optional Command-line multiplier for matrix sizes
typedef struct _matrixSize
{
  unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;


// Allocates a matrix with random float entries.
void randomInit(float *data, int size)
{
  for (int i = 0; i < size; ++i)
    data[i] = rand() / (float)RAND_MAX;
}

void initializeCUDA(int argc, char **argv, int &devID, int &iSizeMultiple,
                    sMatrixSize &matrix_size)
{
  // By default, we use device 0, otherwise we override the device ID based on
  // what is provided at the command line
  cudaError_t error;
  // devID = 0;
  // devID = findCudaDevice(argc, (const char **)argv);
  if (checkCmdLineFlag(argc, (const char **)argv, "sizemult"))
  {
    iSizeMultiple =
        getCmdLineArgumentInt(argc, (const char **)argv, "sizemult");
  }
  iSizeMultiple = min(iSizeMultiple, 10);
  iSizeMultiple = max(iSizeMultiple, 1);
  // cudaDeviceProp deviceProp;
  // error = cudaGetDeviceProperties(&deviceProp, devID);
  // if (error != cudaSuccess)
  // {
  //   printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error,
  //          __LINE__);
  //   exit(EXIT_FAILURE);
  // }
  // printf("GPU Device %d: \"%s\" with compute capability %d.%d\n", devID,
        //  deviceProp.name, deviceProp.major, deviceProp.minor);
  int block_size = 32;
  matrix_size.uiWA = 4 * block_size * iSizeMultiple;
  matrix_size.uiHA = 4 * block_size * iSizeMultiple;
  matrix_size.uiWB = 4 * block_size * iSizeMultiple;
  matrix_size.uiHB = 4 * block_size * iSizeMultiple;
  matrix_size.uiWC = 4 * block_size * iSizeMultiple;
  matrix_size.uiHC = 4 * block_size * iSizeMultiple;

  printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n", matrix_size.uiHA,
         matrix_size.uiWA, matrix_size.uiHB, matrix_size.uiWB, matrix_size.uiHC,
         matrix_size.uiWC);
  if (matrix_size.uiWA != matrix_size.uiHB ||
      matrix_size.uiHA != matrix_size.uiHC ||
      matrix_size.uiWB != matrix_size.uiWC)
  {
    printf("ERROR: Matrix sizes do not match!\n");
    exit(-1);
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test matrix multiply using CUBLAS
////////////////////////////////////////////////////////////////////////////////
int matrixMultiply(int argc, char **argv, int devID, sMatrixSize &matrix_size, sMatrixSize &matrix_size2)
{
  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
  int block_size = 32;
  // set seed for rand()
  srand(2006);
  // allocate host memory for matrices A and B
  unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float *h_A = (float *)malloc(mem_size_A);
  unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float *h_B = (float *)malloc(mem_size_B);
  // allocate another two host memory for matrices A2 and B2 
  unsigned int size_A2 = matrix_size2.uiWA * matrix_size2.uiHA;
  unsigned int mem_size_A2 = sizeof(float) * size_A2;
  float *h_A2 = (float *)malloc(mem_size_A2);
  unsigned int size_B2 = matrix_size2.uiWB * matrix_size2.uiHB;
  unsigned int mem_size_B2 = sizeof(float) * size_B2;
  float *h_B2 = (float *)malloc(mem_size_B2);


  // set seed for rand()
  srand(2006);

  // initialize host memory
  randomInit(h_A, size_A);
  randomInit(h_B, size_B);
  // initialize another two host memory
  randomInit(h_A2, size_A2);
  randomInit(h_B2, size_B2);

  // allocate device memory
  float *d_A, *d_B, *d_C;
  unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
  unsigned int mem_size_C = sizeof(float) * size_C;
  // allocate device memory for another test
  float *d_A2, *d_B2, *d_C2;
  unsigned int size_C2 = matrix_size2.uiWC * matrix_size2.uiHC;
  unsigned int mem_size_C2 = sizeof(float) * size_C2;

  // allocate host memory for the result
  float *h_C = (float *)malloc(mem_size_C);
  float *h_CUBLAS = (float *)malloc(mem_size_C);
  // allocate host memory for the result of another test
  float *h_C2 = (float *)malloc(mem_size_C2);
  float *h_CUBLAS2 = (float *)malloc(mem_size_C2);

  checkCudaErrors(cudaMalloc((void **)&d_A, mem_size_A));
  checkCudaErrors(cudaMalloc((void **)&d_B, mem_size_B));
  checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **)&d_C, mem_size_C));
  // allocate device memory for another test
  checkCudaErrors(cudaMalloc((void **)&d_A2, mem_size_A2));
  checkCudaErrors(cudaMalloc((void **)&d_B2, mem_size_B2));
  checkCudaErrors(cudaMemcpy(d_A2, h_A2, mem_size_A2, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_B2, h_B2, mem_size_B2, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **)&d_C2, mem_size_C2));


  // setup execution parameters
  dim3 threads(block_size, block_size);
  dim3 grid(matrix_size.uiWC / threads.x, matrix_size.uiHC / threads.y);
  // setup execution parameters for another test
  dim3 threads2(block_size, block_size);
  dim3 grid2(matrix_size2.uiWC / threads2.x, matrix_size2.uiHC / threads2.y);

  // create and start timer
  printf("Computing result using CUBLAS...");

  // execute the kernel
  int nIter = 10;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasHandle_t handle;
  cudaEvent_t start, stop;
  cudaEvent_t start2, stop2;
  float msecTotal = 0.0f;

  checkCudaErrors(cublasCreate(&handle));

  // Perform warmup operation with cublas
  for (int j = 0; j < 100; j++)
  {
    // note cublas is column primary!
    // need to transpose the order
    checkCudaErrors(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA,
        matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A,
        matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
    checkCudaErrors(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size2.uiWB, matrix_size2.uiHA,
        matrix_size2.uiWA, &alpha, d_B2, matrix_size2.uiWB, d_A2,
        matrix_size2.uiWA, &beta, d_C2, matrix_size2.uiWB));
  }
  checkCudaErrors(cudaDeviceSynchronize());

  // Test round 2:  1664*1664 matrix multiplications, then 10 128*128 matrix multiplications
  checkCudaErrors(cudaEventCreate(&start2));
  checkCudaErrors(cudaEventCreate(&stop2));
  printf("\n==============\nTest round 2 starts\n");
  printf("1664*1664 matrix multiplications, then 10 128*128 matrix multiplications\n");
  checkCudaErrors(cudaEventRecord(start2, NULL));
  checkCudaErrors(cublasSgemm(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size2.uiWB, matrix_size2.uiHA,
      matrix_size2.uiWA, &alpha, d_B2, matrix_size2.uiWB, d_A2,
      matrix_size2.uiWA, &beta, d_C2, matrix_size2.uiWB));
  for (int j = 0; j < nIter; j++)
  {
    // note cublas is column primary!
    // need to transpose the order
    checkCudaErrors(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA,
        matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A,
        matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
  }
  checkCudaErrors(cudaEventRecord(stop2, NULL));
  // Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stop2));
  msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start2, stop2));
  //  print total time
  printf("Time= %.3f msec", msecTotal); // Test round 2:  1664*1664 matrix multiplications, then 10 128*128 matrix multiplications



  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  // Test round 1: 10 128*128 matrix multiplications, then 1664*1664 matrix multiplications
  printf("\n==============\nTest round 1 starts\n");
  printf("10 128*128 matrix multiplications, then 1664*1664 matrix multiplications\n");
  // Record the start event
  checkCudaErrors(cudaEventRecord(start, NULL));
  for (int j = 0; j < nIter; j++)
  {
    // note cublas is column primary!
    // need to transpose the order
    checkCudaErrors(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA,
        matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A,
        matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
  }
  // run another test
  checkCudaErrors(cublasSgemm(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size2.uiWB, matrix_size2.uiHA,
      matrix_size2.uiWA, &alpha, d_B2, matrix_size2.uiWB, d_A2,
      matrix_size2.uiWA, &beta, d_C2, matrix_size2.uiWB));

  printf("done.\n");
  // Record the stop event
  checkCudaErrors(cudaEventRecord(stop, NULL));
  // Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaDeviceSynchronize());
                                       
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
  //  print total time
  printf("Time= %.3f msec", msecTotal);


  // checkCudaErrors(
  //     cudaMemcpy(h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost));
  // // Destroy the handle
  checkCudaErrors(cublasDestroy(handle));
  // clean up memory for another test
  free(h_A2);
  free(h_B2);
  free(h_C2);
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));
  // clean up memory for another test
  checkCudaErrors(cudaFree(d_A2));
  checkCudaErrors(cudaFree(d_B2));
  checkCudaErrors(cudaFree(d_C2));
  return EXIT_SUCCESS; // return value = 1
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
  printf("[Matrix Multiply CUBLAS] - Starting...\n");

  int devID = 0, sizeMult;
  sMatrixSize matrix_size;
  sMatrixSize matrix_size2;
  printf("small matrix\n");
  sizeMult = 1;
  // maxtri size is 128*128
  initializeCUDA(argc, argv, devID, sizeMult, matrix_size);
  printf("large matrix\n");
  sizeMult = 13;
  // maxtri size is 1664*1664
  initializeCUDA(argc, argv, devID, sizeMult, matrix_size2);
  int matrix_result = matrixMultiply(argc, argv, devID, matrix_size, matrix_size2);

  return matrix_result;
}