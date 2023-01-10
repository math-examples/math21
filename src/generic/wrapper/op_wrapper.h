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

#pragma once

#include "inner_c.h"
#include "op_wrapper_set.h"
#include "op_wrapper_f_set.h"

#ifdef __cplusplus
extern "C" {
#endif

void math21_generic_vector_kx_wrapper(NumN n, NumR k, PtrVoidWrapper x, NumN stride_x, NumN type);

void math21_generic_vector_kx_add_y_wrapper(
        NumN n, NumR k, PtrVoidInWrapper x, NumN stride_x, PtrVoidWrapper y,
        NumN stride_y, NumN type);

void math21_generic_vector_xy_wrapper(
        NumN n, PtrVoidInWrapper x, NumN stride_x, PtrVoidWrapper y,
        NumN stride_y, NumN type);

void math21_generic_vector_sin_wrapper(NumN n, PtrVoidInWrapper x, PtrVoidWrapper y, NumN type);

void math21_generic_vector_cos_wrapper(NumN n, PtrVoidInWrapper x, PtrVoidWrapper y, NumN type);

void math21_generic_tensor_3d_swap_row_in_d2_wrapper(
        NumN n, PtrVoidWrapper x, NumN i, NumN j, NumN d1, NumN d2, NumN d3, NumN type);

void math21_generic_vector_addToC_wrapper(NumN n, PtrVoidInWrapper A,
                                          PtrVoidInWrapper B, PtrVoidWrapper C, NumN type);

void math21_generic_vector_mulToC_wrapper(NumN n, PtrVoidInWrapper A,
                                          PtrVoidInWrapper B, PtrVoidWrapper C, NumN type);

void math21_generic_broadcast_in_dn_wrapper(NumN n, PtrVoidInWrapper x, PtrVoidWrapper y,
                                            NumN dims_x, PtrNInWrapper dx,
                                            NumN dims_y, PtrNInWrapper dy,
                                            NumN type);

void math21_generic_optimization_adam_update_part_2_wrapper(
        NumN x_size, PtrVoidWrapper x, PtrVoidInWrapper m, PtrVoidInWrapper v,
        NumR beta1, NumR beta2, NumR alpha, NumR eps, NumN t, NumN type);

void math21_generic_tensor_f_shrink_wrapper(
        NumN fname, NumN n, PtrVoidInWrapper x, PtrVoidWrapper y,
        NumN dims_x, PtrNInWrapper dx, NumN dims_y, PtrNInWrapper dy,
        NumN n_b, PtrNInWrapper b,
        NumN n_v, NumN dims_v, PtrNInWrapper dv, NumN type);

void math21_generic_tensor_f_inner_product_like_shrink_wrapper(
        NumN fname, NumN n,
        PtrVoidInWrapper x1, PtrVoidInWrapper x2, PtrVoidWrapper y,
        NumN dims_x, PtrNInWrapper dx, NumN dims_y, PtrNInWrapper dy,
        NumN n_b, PtrNInWrapper b,
        NumN n_v, NumN dims_v, PtrNInWrapper dv, NumN type);

void math21_generic_tensor_f_inner_product_like_bcshrink_wrapper(
        NumN fname, NumN n,
        PtrVoidInWrapper x1, PtrVoidInWrapper x2, PtrVoidWrapper y,
        NumN dims_x1, PtrNInWrapper dx1, NumN dims_x2, PtrNInWrapper dx2,
        NumN dims_x, PtrNInWrapper dx, NumN dims_y, PtrNInWrapper dy,
        NumN n_b, PtrNInWrapper b,
        NumN n_v, NumN dims_v, PtrNInWrapper dv, NumN type);

void math21_generic_tensor_f_with_broadcast_in_dn_wrapper(NumN fname, NumN n,
                                                          PtrVoidInWrapper x1,
                                                          PtrVoidInWrapper x2,
                                                          PtrVoidWrapper y,
                                                          NumN dims_x1, PtrNInWrapper dx1,
                                                          NumN dims_x2, PtrNInWrapper dx2,
                                                          NumN dims_y, PtrNInWrapper dy, NumN type);

void math21_generic_vector_f_add_like_wrapper(NumN fname, NumN n,
                                              PtrVoidInWrapper x1,
                                              PtrVoidInWrapper x2,
                                              PtrVoidWrapper y, NumN type);

void math21_generic_vector_f_sin_like_wrapper(NumN fname, NumN n,
                                              PtrVoidInWrapper x1,
                                              PtrVoidWrapper y, NumN type);

void math21_generic_vector_f_kx_like_wrapper(NumN fname, NumN n, NumR k,
                                             PtrVoidInWrapper x1,
                                             PtrVoidWrapper y, NumN type);

void math21_generic_matrix_multiply_onto_k1AB_add_k2C_similar_wrapper(
        NumB ta, NumB tb, NumN nr_C, NumN nc_C, NumN n_common, NumR k1,
        PtrVoidInWrapper A, NumN stride_a,
        PtrVoidInWrapper B, NumN stride_b,
        NumR k2, PtrVoidWrapper C, NumN stride_c, NumN type);

void math21_generic_matrix_transpose_wrapper(NumN n,
                                             PtrVoidInWrapper x,
                                             PtrVoidWrapper y,
                                             NumN nr_x, NumN nc_x, NumN type1, NumN type2);

void math21_generic_matrix_trans_reverse_axis_wrapper(NumN n,
                                                      PtrVoidInWrapper x,
                                                      PtrVoidWrapper y,
                                                      NumN nr_x, NumN nc_x, NumB isXAxis, NumN type1, NumN type2);

void math21_generic_tensor_swap_axes_24_in_d5_wrapper(
        PtrVoidInWrapper x,
        PtrVoidWrapper y,
        NumN d1, NumN d2, NumN d3, NumN d4, NumN d5, NumN type1, NumN type2);

void math21_generic_cross_correlation_X_to_X_prime_wrapper(
        PtrVoidInWrapper X, PtrVoidWrapper X_prime, NumN nch_X, NumN nr_X, NumN nc_X,
        NumN nr_k, NumN nc_k, NumN nr_p, NumN nc_p,
        NumN nr_s, NumN nc_s, NumN nr_d, NumN nc_d, NumN type);

void math21_generic_cross_correlation_dX_prime_to_dX_wrapper(
        PtrVoidInWrapper dX_prime, PtrVoidWrapper dX, NumN nch_X, NumN nr_X, NumN nc_X,
        NumN nr_k, NumN nc_k, NumN nr_p, NumN nc_p,
        NumN nr_s, NumN nc_s, NumN nr_d, NumN nc_d, NumN type);

#ifdef __cplusplus
}
#endif
