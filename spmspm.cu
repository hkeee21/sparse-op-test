// file: spmm.cu.cc
//
// Using cusparse API to test SpMM performance.
//  author: guyue huang
//  date  : 2021/10/13
// compile: nvcc version >=11.0

// #include "../../src/ge-spmm/gespmm.h" // gespmmCsrSpMM()
#include "util/sp_util.hpp"        // read_mtx
#include <cstdlib>                    // std::rand(), RAND_MAX
#include <cuda_runtime_api.h>         // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h> // cusparseSpMM (>= v11.0) or cusparseScsrmm
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

int main(int argc, const char **argv) {

  /// check command-line argument

  if (argc != 3) {
    printf("Require command-line argument: name of the sparse matrix file in "
           ".mtx format.\n");
    return EXIT_FAILURE;
  }

  //
  // Load sparse matrix
  //

  int A_num_rows;                              // number of A-rows
  int A_num_cols;                              // number of A-columns
  int A_nnz;                                   // number of non-zeros in A
  int B_num_rows;                              // number of A-rows
  int B_num_cols;                              // number of A-columns
  int B_nnz;
  std::vector<int> csr_indptr_buffer_A; // buffer for indptr array in CSR format
  std::vector<int> csr_indices_buffer_A; // buffer for indices (column-ids) array in CSR format
  std::vector<int> csr_indptr_buffer_B; // buffer for indptr array in CSR format
  std::vector<int> csr_indices_buffer_B; // buffer for indices (column-ids) array in CSR format
  // load sparse matrix from mtx file
  read_mtx_file(argv[1], A_num_rows, A_num_cols, A_nnz, csr_indptr_buffer_A, csr_indices_buffer_A);
  read_mtx_file(argv[2], B_num_rows, B_num_cols, B_nnz, csr_indptr_buffer_B, csr_indices_buffer_B);
//   printf("Finish reading matrix %d rows, %d columns, %d nnz. \nIgnore original "
//          "values and use randomly generated values.\n",
//          A_num_rows, A_num_cols, A_nnz);
  // int B_num_rows,B_num_cols,B_nnz;
  // if (A_num_rows != A_num_cols){
  //   //B=A^T
  //    B_num_rows = A_num_cols;
  //    B_num_cols = A_num_rows;
  //    B_nnz = A_nnz;
  // }
  // else{
  //   // B=A
  //    B_num_rows = A_num_cols;
  //    B_num_cols = A_num_rows;
  //    B_nnz = A_nnz;
  // }

  float               alpha       = 1.0f;
  float               beta        = 0.0f;
  cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cudaDataType        computeType = CUDA_R_32F;

  //--------------------------------------------------------------------------
  // Host memory management:
  int   *hA_csrOffsets = csr_indptr_buffer_A.data();
  int   *hA_columns    = csr_indices_buffer_A.data();
  int   *hB_csrOffsets = csr_indptr_buffer_B.data();
  int   *hB_columns    = csr_indices_buffer_B.data();
  float *hA_values     = (float *)malloc(sizeof(float) * A_nnz);
  float *hB_values     = (float *)malloc(sizeof(float) * B_nnz);
  fill_random(hA_values, A_nnz);
  fill_random(hB_values, B_nnz);

  // int   *hB_csrOffsets, *hB_columns;
  // float *hB_values;
  // hB_csrOffsets = hA_csrOffsets;
  // hB_columns = hA_columns;
  // hB_values = hA_values;


  // Device memory management: Allocate and copy A, B
  int   *dA_csrOffsets, *dA_columns, *dB_csrOffsets, *dB_columns,
        *dC_csrOffsets, *dC_columns;
  float *dA_values, *dB_values, *dC_values;
  // allocate A
  CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                          (A_num_rows + 1) * sizeof(int)) )
  CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))   )
  CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float)) )
    // copy A
    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
        (A_num_rows + 1) * sizeof(int),
        cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
        cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values,
        A_nnz * sizeof(float), cudaMemcpyHostToDevice) )


// if (A_num_rows == A_num_cols){
    // B=A
  // allocate B
  CHECK_CUDA( cudaMalloc((void**) &dB_csrOffsets,
                          (B_num_rows + 1) * sizeof(int)) )
  CHECK_CUDA( cudaMalloc((void**) &dB_columns, B_nnz * sizeof(int))   )
  CHECK_CUDA( cudaMalloc((void**) &dB_values,  B_nnz * sizeof(float)) )


  // copy B
  CHECK_CUDA( cudaMemcpy(dB_csrOffsets, hB_csrOffsets,
                          (B_num_rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dB_columns, hB_columns, B_nnz * sizeof(int),
                          cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dB_values, hB_values,
                          B_nnz * sizeof(float), cudaMemcpyHostToDevice) )
//   }
// else{
//     // B=A^T
//   // allocate B
//   CHECK_CUDA( cudaMalloc((void**) &dB_csrOffsets,
//                           (B_num_rows + 1) * sizeof(int)) )
//   CHECK_CUDA( cudaMalloc((void**) &dB_columns, B_nnz * sizeof(int))   )
//   CHECK_CUDA( cudaMalloc((void**) &dB_values,  B_nnz * sizeof(float)) )

//   cusparseHandle_t     handle = NULL;
//   void*  dBuffer_trans    = NULL;
//   size_t bufferSize_trans = 0;
//   CHECK_CUSPARSE( cusparseCreate(&handle) )
//   CHECK_CUSPARSE( cusparseCsr2cscEx2_bufferSize(handle,A_num_rows,A_num_cols,A_nnz,dA_values,dA_csrOffsets,dA_columns,dB_values,dB_csrOffsets,dB_columns, CUDA_R_32F,CUSPARSE_ACTION_NUMERIC,CUSPARSE_INDEX_BASE_ZERO,CUSPARSE_CSR2CSC_ALG1,&bufferSize_trans))

//     CHECK_CUDA( cudaMalloc((void**) &dBuffer_trans, bufferSize_trans) )
//     CHECK_CUSPARSE( cusparseCsr2cscEx2(handle,A_num_rows,A_num_cols,A_nnz,dA_values,dA_csrOffsets,dA_columns,dB_values,dB_csrOffsets,dB_columns, CUDA_R_32F,CUSPARSE_ACTION_NUMERIC,CUSPARSE_INDEX_BASE_ZERO,CUSPARSE_CSR2CSC_ALG1,dBuffer_trans))
// }

  // allocate C offsets
  CHECK_CUDA( cudaMalloc((void**) &dC_csrOffsets,
                          (A_num_rows + 1) * sizeof(int)) )
  //--------------------------------------------------------------------------
  // CUSPARSE APIs
  cusparseHandle_t     handle = NULL;
  cusparseSpMatDescr_t matA, matB, matC;
  void*  dBuffer1    = NULL, *dBuffer2   = NULL;
  size_t bufferSize1 = 0,    bufferSize2 = 0;
  CHECK_CUSPARSE( cusparseCreate(&handle) )
  // Create sparse matrix A in CSR format
  CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                    dA_csrOffsets, dA_columns, dA_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
  CHECK_CUSPARSE( cusparseCreateCsr(&matB, B_num_rows, B_num_cols, B_nnz,
                                    dB_csrOffsets, dB_columns, dB_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
  CHECK_CUSPARSE( cusparseCreateCsr(&matC, A_num_rows, B_num_cols, 0,
                                    NULL, NULL, NULL,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
  //--------------------------------------------------------------------------
  // SpGEMM Computation
  cusparseSpGEMMDescr_t spgemmDesc;
  CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) )

  // ask bufferSize1 bytes for external memory
  CHECK_CUSPARSE(
      cusparseSpGEMM_workEstimation(handle, opA, opB,
                                    &alpha, matA, matB, &beta, matC,
                                    computeType, CUSPARSE_SPGEMM_DEFAULT,
                                    spgemmDesc, &bufferSize1, NULL) )
  CHECK_CUDA( cudaMalloc((void**) &dBuffer1, bufferSize1) )
  // inspect the matrices A and B to understand the memory requirement for
  // the next step
  CHECK_CUSPARSE(
      cusparseSpGEMM_workEstimation(handle, opA, opB,
                                    &alpha, matA, matB, &beta, matC,
                                    computeType, CUSPARSE_SPGEMM_DEFAULT,
                                    spgemmDesc, &bufferSize1, dBuffer1) )

  // ask bufferSize2 bytes for external memory
  CHECK_CUSPARSE(
      cusparseSpGEMM_compute(handle, opA, opB,
                              &alpha, matA, matB, &beta, matC,
                              computeType, CUSPARSE_SPGEMM_DEFAULT,
                              spgemmDesc, &bufferSize2, NULL) )
  CHECK_CUDA( cudaMalloc((void**) &dBuffer2, bufferSize2) )

  // compute the intermediate product of A * B
  GpuTimer gpu_timer;
  int warmup_iter = 10;
  int repeat_iter = 100;
  for (int iter = 0; iter < warmup_iter + repeat_iter; iter++) {
    if (iter == warmup_iter) {
      gpu_timer.start();
    }

    CHECK_CUSPARSE( cusparseSpGEMM_compute(handle, opA, opB,
        &alpha, matA, matB, &beta, matC,
        computeType, CUSPARSE_SPGEMM_DEFAULT,
        spgemmDesc, &bufferSize2, dBuffer2) )
  }
  gpu_timer.stop();

  float kernel_dur_msecs = gpu_timer.elapsed_msecs() / repeat_iter;

  printf("%.4f ms\n",kernel_dur_msecs);

//   printf("[Cusparse] Report: spgemm A(%d x %d) * A sparsity %f "
//          "(nnz=%d) \n Time %f (ms)\n",
//          A_num_rows, A_num_cols, (float)A_nnz / A_num_rows / A_num_cols, A_nnz, kernel_dur_msecs);

//   float MFlop_count = (float)A_nnz / 1e6 * N * 2;

//   float gflops = MFlop_count / kernel_dur_msecs;

//   printf("[Cusparse] Report: spgemm A(%d x %d) * A sparsity %f "
//          "(nnz=%d) \n Time %f (ms), Throughput %f (gflops).\n",
//          A_num_rows, A_num_cols, (float)A_nnz / A_num_rows / A_num_cols, A_nnz, kernel_dur_msecs, gflops);

  // get matrix C non-zero entries C_nnz1
  int64_t C_num_rows1, C_num_cols1, C_nnz1;
  CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
                                        &C_nnz1) )
  // allocate matrix C
  CHECK_CUDA( cudaMalloc((void**) &dC_columns, C_nnz1 * sizeof(int))   )
  CHECK_CUDA( cudaMalloc((void**) &dC_values,  C_nnz1 * sizeof(float)) )

  // NOTE: if 'beta' != 0, the values of C must be update after the allocation
  //       of dC_values, and before the call of cusparseSpGEMM_copy

  // update matC with the new pointers
  CHECK_CUSPARSE(
      cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values) )

  // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

  // copy the final products to the matrix C
  CHECK_CUSPARSE(
      cusparseSpGEMM_copy(handle, opA, opB,
                          &alpha, matA, matB, &beta, matC,
                          computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc) )

  // destroy matrix/vector descriptors
  CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
  CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
  CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
  CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
  CHECK_CUSPARSE( cusparseDestroy(handle) )
//   printf("spgemm success\n");
  //--------------------------------------------------------------------------
  // // device result check
  // int   hC_csrOffsets_tmp[A_NUM_ROWS + 1];
  // int   hC_columns_tmp[C_NUM_NNZ];
  // float hC_values_tmp[C_NUM_NNZ];
  // CHECK_CUDA( cudaMemcpy(hC_csrOffsets_tmp, dC_csrOffsets,
  //                         (A_num_rows + 1) * sizeof(int),
  //                         cudaMemcpyDeviceToHost) )
  // CHECK_CUDA( cudaMemcpy(hC_columns_tmp, dC_columns, C_nnz * sizeof(int),
  //                         cudaMemcpyDeviceToHost) )
  // CHECK_CUDA( cudaMemcpy(hC_values_tmp, dC_values, C_nnz * sizeof(float),
  //                         cudaMemcpyDeviceToHost) )
  // int correct = 1;
  // for (int i = 0; i < A_num_rows + 1; i++) {
  //     if (hC_csrOffsets_tmp[i] != hC_csrOffsets[i]) {
  //         correct = 0;
  //         break;
  //     }
  // }
  // for (int i = 0; i < C_nnz; i++) {
  //     if (hC_columns_tmp[i] != hC_columns[i] ||
  //         hC_values_tmp[i]  != hC_values[i]) { // direct floating point
  //         correct = 0;                         // comparison is not reliable
  //         break;
  //     }
  // }
  // if (correct)
  //     printf("spgemm_example test PASSED\n");
  // else {
  //     printf("spgemm_example test FAILED: wrong result\n");
  //     return EXIT_FAILURE;
  // }
  //--------------------------------------------------------------------------
  // device memory deallocation
  CHECK_CUDA( cudaFree(dBuffer1) )
  CHECK_CUDA( cudaFree(dBuffer2) )
  CHECK_CUDA( cudaFree(dA_csrOffsets) )
  CHECK_CUDA( cudaFree(dA_columns) )
  CHECK_CUDA( cudaFree(dA_values) )
  CHECK_CUDA( cudaFree(dB_csrOffsets) )
  CHECK_CUDA( cudaFree(dB_columns) )
  CHECK_CUDA( cudaFree(dB_values) )
  CHECK_CUDA( cudaFree(dC_csrOffsets) )
  CHECK_CUDA( cudaFree(dC_columns) )
  CHECK_CUDA( cudaFree(dC_values) )
  return EXIT_SUCCESS;

  // if (correct) {

  //   GpuTimer gpu_timer;
  //   int warmup_iter = 10;
  //   int repeat_iter = 100;
  //   for (int iter = 0; iter < warmup_iter + repeat_iter; iter++) {
  //     if (iter == warmup_iter) {
  //       gpu_timer.start();
  //     }

  //     cusparseSpMM(handle,
  //                  CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
  //                  CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
  //                  &alpha, csrDescr, dnMatInputDescr, &beta, dnMatOutputDescr,
  //                  CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, workspace);
  //   }
  //   gpu_timer.stop();

  //   float kernel_dur_msecs = gpu_timer.elapsed_msecs() / repeat_iter;

  //   float MFlop_count = (float)A_nnz / 1e6 * B_num_cols * 2;

  //   float gflops = MFlop_count / kernel_dur_msecs;

  //   printf("[Cusparse] Report: spmm A(%d x %d) * B(%d x %d) sparsity %f "
  //          "(nnz=%d) \n Time %f (ms), Throughput %f (gflops).\n",
  //          A_num_rows, A_num_cols, A_num_cols, B_num_cols, (float)A_nnz / A_num_rows / A_num_cols, A_nnz, kernel_dur_msecs, gflops);
  // }


}
