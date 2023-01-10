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

#include "../cpu/files_c.h"
#include "../cuda/files_c.h"
#include "../opencl/files_c.h"
#include "op_wrapper.h"

void math21_generic_vector_kx_wrapper(NumN n, NumR k, PtrVoidWrapper x, NumN stride_x, NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_vector_kx_cpu(n, k, x, stride_x, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_vector_kx_cuda(n, k, x, stride_x, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_vector_kx_opencl(n, k, x, stride_x, type);
#endif
}

void math21_generic_vector_kx_add_y_wrapper(
        NumN n, NumR k, PtrVoidInWrapper x, NumN stride_x, PtrVoidWrapper y,
        NumN stride_y, NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_vector_kx_add_y_cpu(n, k, x, stride_x, y, stride_y, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_vector_kx_add_y_cuda(n, k, x, stride_x, y, stride_y, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_vector_kx_add_y_opencl(n, k, x, stride_x, y, stride_y, type);
#endif
}

void math21_generic_vector_xy_wrapper(
        NumN n, PtrVoidInWrapper x, NumN stride_x, PtrVoidWrapper y,
        NumN stride_y, NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_vector_xy_cpu(n, x, stride_x, y, stride_y, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_vector_xy_cuda(n, x, stride_x, y, stride_y, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_vector_xy_opencl(n, x, stride_x, y, stride_y, type);
#endif
}

void math21_generic_vector_sin_wrapper(NumN n, PtrVoidInWrapper x, PtrVoidWrapper y, NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_vector_sin_cpu(n, x, y, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_vector_sin_cuda(n, x, y, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_vector_sin_opencl(n, x, y, type);
#endif
}

void math21_generic_vector_cos_wrapper(NumN n, PtrVoidInWrapper x, PtrVoidWrapper y, NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_vector_cos_cpu(n, x, y, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_vector_cos_cuda(n, x, y, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_vector_cos_opencl(n, x, y, type);
#endif
}

void math21_generic_tensor_3d_swap_row_in_d2_wrapper(
        NumN n, PtrVoidWrapper x, NumN i, NumN j, NumN d1, NumN d2, NumN d3, NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_tensor_3d_swap_row_in_d2_cpu(n, x, i, j, d1, d2, d3, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_tensor_3d_swap_row_in_d2_cuda(n, x, i, j, d1, d2, d3, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_tensor_3d_swap_row_in_d2_opencl(n, x, i, j, d1, d2, d3, type);
#endif
}

void math21_generic_vector_addToC_wrapper(NumN n, PtrVoidInWrapper A,
                                          PtrVoidInWrapper B, PtrVoidWrapper C, NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_vector_addToC_cpu(n, A, B, C, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_vector_addToC_cuda(n, A, B, C, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_vector_addToC_opencl(n, A, B, C, type);
#endif
}

void math21_generic_vector_mulToC_wrapper(NumN n, PtrVoidInWrapper A,
                                          PtrVoidInWrapper B, PtrVoidWrapper C, NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_vector_mulToC_cpu(n, A, B, C, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_vector_mulToC_cuda(n, A, B, C, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_vector_mulToC_opencl(n, A, B, C, type);
#endif
}

void math21_generic_broadcast_in_dn_wrapper(NumN n, PtrVoidInWrapper x, PtrVoidWrapper y,
                                            NumN dims_x, PtrNInWrapper dx,
                                            NumN dims_y, PtrNInWrapper dy,
                                            NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_broadcast_in_dn_cpu(n, x, y, dims_x, dx, dims_y, dy, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_broadcast_in_dn_cuda(n, x, y, dims_x, dx, dims_y, dy, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_broadcast_in_dn_opencl(n, x, y, dims_x, dx, dims_y, dy, type);
#endif
}

void math21_generic_optimization_adam_update_part_2_wrapper(
        NumN x_size, PtrVoidWrapper x, PtrVoidInWrapper m, PtrVoidInWrapper v,
        NumR beta1, NumR beta2, NumR alpha, NumR eps, NumN t, NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_optimization_adam_update_part_2_cpu(x_size, x, m, v, beta1, beta2, alpha, eps, t, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_optimization_adam_update_part_2_cuda(x_size, x, m, v, beta1, beta2, alpha, eps, t, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_optimization_adam_update_part_2_opencl(x_size, x, m, v, beta1, beta2, alpha, eps, t, type);
#endif
}

void math21_generic_tensor_f_shrink_wrapper(
        NumN fname, NumN n, PtrVoidInWrapper x, PtrVoidWrapper y,
        NumN dims_x, PtrNInWrapper dx, NumN dims_y, PtrNInWrapper dy,
        NumN n_b, PtrNInWrapper b,
        NumN n_v, NumN dims_v, PtrNInWrapper dv, NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_tensor_f_shrink_cpu(fname, n, x, y,
                                       dims_x, dx, dims_y, dy, n_b, b, n_v, dims_v, dv, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_tensor_f_shrink_cuda(fname, n, x, y,
                                        dims_x, dx, dims_y, dy, n_b, b, n_v, dims_v, dv, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_tensor_f_shrink_opencl(fname, n, x, y,
                                          dims_x, dx, dims_y, dy, n_b, b, n_v, dims_v, dv, type);
#endif
}

void math21_generic_tensor_f_inner_product_like_shrink_wrapper(
        NumN fname, NumN n,
        PtrVoidInWrapper x1, PtrVoidInWrapper x2, PtrVoidWrapper y,
        NumN dims_x, PtrNInWrapper dx, NumN dims_y, PtrNInWrapper dy,
        NumN n_b, PtrNInWrapper b,
        NumN n_v, NumN dims_v, PtrNInWrapper dv, NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_tensor_f_inner_product_like_shrink_cpu(
            fname, n, x1, x2, y,
            dims_x, dx, dims_y, dy, n_b, b, n_v, dims_v, dv, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_tensor_f_inner_product_like_shrink_cuda(
            fname, n, x1, x2, y,
            dims_x, dx, dims_y, dy, n_b, b, n_v, dims_v, dv, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_tensor_f_inner_product_like_shrink_opencl(
            fname, n, x1, x2, y,
            dims_x, dx, dims_y, dy, n_b, b, n_v, dims_v, dv, type);
#endif
}

void math21_generic_tensor_f_inner_product_like_bcshrink_wrapper(
        NumN fname, NumN n,
        PtrVoidInWrapper x1, PtrVoidInWrapper x2, PtrVoidWrapper y,
        NumN dims_x1, PtrNInWrapper dx1, NumN dims_x2, PtrNInWrapper dx2,
        NumN dims_x, PtrNInWrapper dx, NumN dims_y, PtrNInWrapper dy,
        NumN n_b, PtrNInWrapper b,
        NumN n_v, NumN dims_v, PtrNInWrapper dv, NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_tensor_f_inner_product_like_bcshrink_cpu(
            fname, n, x1, x2, y,
            dims_x1, dx1, dims_x2, dx2,
            dims_x, dx, dims_y, dy, n_b, b, n_v, dims_v, dv, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_tensor_f_inner_product_like_bcshrink_cuda(
            fname, n, x1, x2, y,
            dims_x1, dx1, dims_x2, dx2,
            dims_x, dx, dims_y, dy, n_b, b, n_v, dims_v, dv, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_tensor_f_inner_product_like_bcshrink_opencl(
            fname, n, x1, x2, y,
            dims_x1, dx1, dims_x2, dx2,
            dims_x, dx, dims_y, dy, n_b, b, n_v, dims_v, dv, type);
#endif
}

void math21_generic_tensor_f_with_broadcast_in_dn_wrapper(NumN fname, NumN n,
                                                          PtrVoidInWrapper x1,
                                                          PtrVoidInWrapper x2,
                                                          PtrVoidWrapper y,
                                                          NumN dims_x1, PtrNInWrapper dx1,
                                                          NumN dims_x2, PtrNInWrapper dx2,
                                                          NumN dims_y, PtrNInWrapper dy, NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_tensor_f_with_broadcast_in_dn_cpu(fname, n,
                                                     x1, x2, y, dims_x1, dx1, dims_x2, dx2,
                                                     dims_y, dy, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_tensor_f_with_broadcast_in_dn_cuda(fname, n,
                                                      x1, x2, y, dims_x1, dx1, dims_x2, dx2,
                                                      dims_y, dy, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_tensor_f_with_broadcast_in_dn_opencl(fname, n,
                                                        x1, x2, y, dims_x1, dx1, dims_x2, dx2,
                                                        dims_y, dy, type);
#endif
}

void math21_generic_vector_f_add_like_wrapper(NumN fname, NumN n,
                                              PtrVoidInWrapper x1,
                                              PtrVoidInWrapper x2,
                                              PtrVoidWrapper y, NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_vector_f_add_like_cpu(fname, n, x1, x2, y, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_vector_f_add_like_cuda(fname, n, x1, x2, y, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_vector_f_add_like_opencl(fname, n, x1, x2, y, type);
#endif
}

void math21_generic_vector_f_sin_like_wrapper(NumN fname, NumN n,
                                              PtrVoidInWrapper x1,
                                              PtrVoidWrapper y, NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_vector_f_sin_like_cpu(fname, n, x1, y, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_vector_f_sin_like_cuda(fname, n, x1, y, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_vector_f_sin_like_opencl(fname, n, x1, y, type);
#endif
}

void math21_generic_vector_f_kx_like_wrapper(NumN fname, NumN n, NumR k,
                                             PtrVoidInWrapper x1,
                                             PtrVoidWrapper y, NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_vector_f_kx_like_cpu(fname, n, k, x1, y, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_vector_f_kx_like_cuda(fname, n, k, x1, y, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_vector_f_kx_like_opencl(fname, n, k, x1, y, type);
#endif
}

void math21_generic_matrix_multiply_onto_k1AB_add_k2C_similar_wrapper(
        NumB ta, NumB tb, NumN nr_C, NumN nc_C, NumN n_common, NumR k1,
        PtrVoidInWrapper A, NumN stride_a,
        PtrVoidInWrapper B, NumN stride_b,
        NumR k2, PtrVoidWrapper C, NumN stride_c, NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_matrix_multiply_onto_k1AB_add_k2C_similar_cpu(
            ta, tb, nr_C, nc_C, n_common, k1, A, stride_a,
            B, stride_b, k2, C, stride_c, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_matrix_multiply_onto_k1AB_add_k2C_similar_cuda(
            ta, tb, nr_C, nc_C, n_common, k1, A, stride_a,
            B, stride_b, k2, C, stride_c, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_matrix_multiply_onto_k1AB_add_k2C_similar_opencl(
            ta, tb, nr_C, nc_C, n_common, k1, A, stride_a,
            B, stride_b, k2, C, stride_c, type);
#endif
}

void math21_generic_matrix_transpose_wrapper(NumN n,
                                             PtrVoidInWrapper x,
                                             PtrVoidWrapper y,
                                             NumN nr_x, NumN nc_x, NumN type1, NumN type2) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_matrix_transpose_cpu(n, x, y, nr_x, nc_x, type1, type2);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_matrix_transpose_cuda(n, x, y, nr_x, nc_x, type1, type2);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_matrix_transpose_opencl(n, x, y, nr_x, nc_x, type1, type2);
#endif
}

void math21_generic_matrix_trans_reverse_axis_wrapper(NumN n,
                                                      PtrVoidInWrapper x,
                                                      PtrVoidWrapper y,
                                                      NumN nr_x, NumN nc_x, NumB isXAxis, NumN type1, NumN type2) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_matrix_trans_reverse_axis_cpu(n, x, y, nr_x, nc_x, isXAxis, type1, type2);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_matrix_trans_reverse_axis_cuda(n, x, y, nr_x, nc_x, isXAxis, type1, type2);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_matrix_trans_reverse_axis_opencl(n, x, y, nr_x, nc_x, isXAxis, type1, type2);
#endif
}

void math21_generic_tensor_swap_axes_24_in_d5_wrapper(
        PtrVoidInWrapper x,
        PtrVoidWrapper y,
        NumN d1, NumN d2, NumN d3, NumN d4, NumN d5, NumN type1, NumN type2) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_tensor_swap_axes_24_in_d5_cpu(x, y, d1, d2, d3, d4, d5, type1, type2);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_tensor_swap_axes_24_in_d5_cuda(x, y, d1, d2, d3, d4, d5, type1, type2);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_tensor_swap_axes_24_in_d5_opencl(x, y, d1, d2, d3, d4, d5, type1, type2);
#endif
}

void math21_generic_cross_correlation_X_to_X_prime_wrapper(
        PtrVoidInWrapper X, PtrVoidWrapper X_prime, NumN nch_X, NumN nr_X, NumN nc_X,
        NumN nr_k, NumN nc_k, NumN nr_p, NumN nc_p,
        NumN nr_s, NumN nc_s, NumN nr_d, NumN nc_d, NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_cross_correlation_X_to_X_prime_cpu(X, X_prime, nch_X, nr_X, nc_X,
                                                      nr_k, nc_k, nr_p, nc_p, nr_s, nc_s, nr_d, nc_d, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_cross_correlation_X_to_X_prime_cuda(X, X_prime, nch_X, nr_X, nc_X,
                                                       nr_k, nc_k, nr_p, nc_p, nr_s, nc_s, nr_d, nc_d, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_cross_correlation_X_to_X_prime_opencl(X, X_prime, nch_X, nr_X, nc_X,
                                                         nr_k, nc_k, nr_p, nc_p, nr_s, nc_s, nr_d, nc_d, type);
#endif
}

void math21_generic_cross_correlation_dX_prime_to_dX_wrapper(
        PtrVoidInWrapper dX_prime, PtrVoidWrapper dX, NumN nch_X, NumN nr_X, NumN nc_X,
        NumN nr_k, NumN nc_k, NumN nr_p, NumN nc_p,
        NumN nr_s, NumN nc_s, NumN nr_d, NumN nc_d, NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_cross_correlation_dX_prime_to_dX_cpu(dX_prime, dX, nch_X, nr_X, nc_X,
                                                        nr_k, nc_k, nr_p, nc_p, nr_s, nc_s, nr_d, nc_d, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_cross_correlation_dX_prime_to_dX_cuda(dX_prime, dX, nch_X, nr_X, nc_X,
                                                         nr_k, nc_k, nr_p, nc_p, nr_s, nc_s, nr_d, nc_d, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_cross_correlation_dX_prime_to_dX_opencl(dX_prime, dX, nch_X, nr_X, nc_X,
                                                           nr_k, nc_k, nr_p, nc_p, nr_s, nc_s, nr_d, nc_d, type);
#endif
}