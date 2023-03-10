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

// generated by op_cpu.cc

#include "template_opencl_03.h"
#include "op_opencl.h"

#ifdef MATH21_FLAG_USE_OPENCL

using namespace math21;

// here stride_a is trailing dimension of a
// stride1_abs = stride1 * stride = 1 * stride = stride
void math21_generic_matrix_multiply_onto_k1AB_add_k2C_similar_opencl(
        NumB ta, NumB tb, NumN nr_C, NumN nc_C, NumN n_common, NumR k1,
        PtrVoidInWrapper A, NumN stride_a,
        PtrVoidInWrapper B, NumN stride_b,
        NumR k2, PtrVoidWrapper C, NumN stride_c, NumN type) {
    if (type == m21_type_NumR) {
        math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_opencl<NumR>(
                ta, tb, nr_C, nc_C, n_common, (NumR) k1, A, stride_a,
                B, stride_b, (NumR) k2, C, stride_c);
    } else if (type == m21_type_NumR32) {
        math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_opencl<NumR32>(
                ta, tb, nr_C, nc_C, n_common, (NumR32) k1, A, stride_a,
                B, stride_b, (NumR32) k2, C, stride_c);
    } else if (type == m21_type_NumR64) {
        math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_opencl<NumR64>(
                ta, tb, nr_C, nc_C, n_common, (NumR64) k1, A, stride_a,
                B, stride_b, (NumR64) k2, C, stride_c);
    } else {
        math21_tool_assert(0);
    }
}

void math21_generic_matrix_transpose_opencl(NumN n, PtrVoidInWrapper x, PtrVoidWrapper y,
                                            NumN nr_x, NumN nc_x, NumN type1, NumN type2) {
    if (type1 == m21_type_NumN8) {
        if (type2 == m21_type_NumN8) {
            math21_template_matrix_transpose_opencl<NumN8, NumN8>(n, x, y, nr_x, nc_x);
        } else if (type2 == m21_type_NumR) {
            math21_template_matrix_transpose_opencl<NumN8, NumR>(n, x, y, nr_x, nc_x);
        } else {
            math21_tool_assert(0);
        }
    } else if (type1 == m21_type_NumR) {
        if (type2 == m21_type_NumN8) {
            math21_template_matrix_transpose_opencl<NumR, NumN8>(n, x, y, nr_x, nc_x);
        } else if (type2 == m21_type_NumN) {
            math21_template_matrix_transpose_opencl<NumR, NumN>(n, x, y, nr_x, nc_x);
        } else if (type2 == m21_type_NumR) {
            math21_template_matrix_transpose_opencl<NumR, NumR>(n, x, y, nr_x, nc_x);
        } else {
            math21_tool_assert(0);
        }
    } else {
        math21_tool_assert(0);
    }
}

void math21_generic_matrix_trans_reverse_axis_opencl(NumN n, PtrVoidInWrapper x, PtrVoidWrapper y,
                                            NumN nr_x, NumN nc_x, NumB isXAxis, NumN type1, NumN type2) {
    if (type1 == m21_type_NumN8) {
        if (type2 == m21_type_NumN8) {
            math21_template_matrix_trans_reverse_axis_opencl<NumN8, NumN8>(n, x, y, nr_x, nc_x, isXAxis);
        } else if (type2 == m21_type_NumR) {
            math21_template_matrix_trans_reverse_axis_opencl<NumN8, NumR>(n, x, y, nr_x, nc_x, isXAxis);
        } else {
            math21_tool_assert(0);
        }
    } else if (type1 == m21_type_NumR) {
        if (type2 == m21_type_NumN8) {
            math21_template_matrix_trans_reverse_axis_opencl<NumR, NumN8>(n, x, y, nr_x, nc_x, isXAxis);
        } else if (type2 == m21_type_NumN) {
            math21_template_matrix_trans_reverse_axis_opencl<NumR, NumN>(n, x, y, nr_x, nc_x, isXAxis);
        } else if (type2 == m21_type_NumR) {
            math21_template_matrix_trans_reverse_axis_opencl<NumR, NumR>(n, x, y, nr_x, nc_x, isXAxis);
        } else {
            math21_tool_assert(0);
        }
    } else {
        math21_tool_assert(0);
    }
}

void math21_generic_tensor_swap_axes_24_in_d5_opencl(PtrVoidInWrapper x, PtrVoidWrapper y,
                                            NumN d1, NumN d2, NumN d3, NumN d4, NumN d5, NumN type1, NumN type2) {
    if (type1 == m21_type_NumN8) {
        if (type2 == m21_type_NumN8) {
            math21_template_tensor_swap_axes_24_in_d5_opencl<NumN8, NumN8>(x, y, d1, d2, d3, d4, d5);
        } else if (type2 == m21_type_NumR) {
            math21_template_tensor_swap_axes_24_in_d5_opencl<NumN8, NumR>(x, y, d1, d2, d3, d4, d5);
        } else {
            math21_tool_assert(0);
        }
    } else if (type1 == m21_type_NumR) {
        if (type2 == m21_type_NumN8) {
            math21_template_tensor_swap_axes_24_in_d5_opencl<NumR, NumN8>(x, y, d1, d2, d3, d4, d5);
        } else if (type2 == m21_type_NumN) {
            math21_template_tensor_swap_axes_24_in_d5_opencl<NumR, NumN>(x, y, d1, d2, d3, d4, d5);
        } else if (type2 == m21_type_NumR) {
            math21_template_tensor_swap_axes_24_in_d5_opencl<NumR, NumR>(x, y, d1, d2, d3, d4, d5);
        } else {
            math21_tool_assert(0);
        }
    } else {
        math21_tool_assert(0);
    }
}

#endif