/* Copyright 2015 The math21 Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "matrix_cuda.h"

void math21_matrix_multiply_k1AB_add_k2C_similar_cuda(int ta, int tb, int nr_C, int nc_C, int n_common, float k1,
                                                      const float *A, int stride_a,
                                                      const float *B, int stride_b,
                                                      float k2,
                                                      float *C, int stride_c) {
    cublasHandle_t handle = math21_cuda_blas_handle();
    cublasStatus_t status = cublasSgemm(handle, (tb ? CUBLAS_OP_T : CUBLAS_OP_N),
                                     (ta ? CUBLAS_OP_T : CUBLAS_OP_N), nc_C, nr_C, n_common, &k1, B, stride_b, A,
                                     stride_a, &k2, C, stride_c);
    math21_cuda_cublas_check_error(status);
}
