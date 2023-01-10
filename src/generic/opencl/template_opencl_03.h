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

#include "inner_cc.h"
#include "../../algebra/files.h"
#include "common.h"

#ifdef MATH21_FLAG_USE_OPENCL

namespace math21 {
    const static std::string kernel_file = "generic_03.kl";
    const static std::string kernel_file_transpose = "generic_03_transpose.kl";
    static Map_<std::string, std::shared_ptr<m21clprogram>> thePrograms;
    Map_<std::string, std::shared_ptr<m21clprogram>> programs_trans;
// todo: reduce the number of programs.

// see math21_matrix_multiply_k1AB_add_k2C_similar
// C = k1*(A*B) + k2*C or similar
    template<typename T>
    void math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_opencl(
            NumB ta, NumB tb, NumN nr_C, NumN nc_C, NumN n_common, T k1,
            PtrVoidInWrapper A, NumN stride_a,
            PtrVoidInWrapper B, NumN stride_b,
            T k2, PtrVoidWrapper C, NumN stride_c) {
//    A -= 1;
//    B -= 1;
//    C -= 1;
        NumN n = nr_C * nc_C;
        std::string functionName;
        if (!ta && !tb) {
            functionName = "math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_nn_naive_opencl_kernel";
        } else if (ta && !tb) {
            functionName = "math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_tn_naive_opencl_kernel";
        } else if (!ta && tb) {
            functionName = "math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_nt_naive_opencl_kernel";
        } else {
            functionName = "math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_tt_naive_opencl_kernel";
        }
        cl_kernel kernel = math21_opencl_kernel_get<T>(functionName, kernel_file, thePrograms);
        math21_opencl_kernel_arg_set(kernel, n, nr_C, nc_C, n_common, k1, A, stride_a, B, stride_b, k2, C, stride_c);
        math21_opencl_kernel_run(kernel, n);
    }

    template<typename T1, typename T2>
    void math21_template_matrix_transpose_opencl(
            NumN n, PtrVoidInWrapper x, PtrVoidWrapper y, NumN d1, NumN d2) {
//    x -= 1;
//    y -= 1;
        cl_kernel kernel = math21_opencl_kernel_get<T1, T2>(
                "math21_template_matrix_transpose_opencl_kernel", kernel_file_transpose, programs_trans);
        math21_opencl_kernel_arg_set(kernel, n, x, y, d1, d2);
        math21_opencl_kernel_run(kernel, n);
    }

    template<typename T1, typename T2>
    void math21_template_matrix_trans_reverse_axis_opencl(
            NumN n, PtrVoidInWrapper x, PtrVoidWrapper y, NumN d1, NumN d2, NumB isXAxis) {
//    x -= 1;
//    y -= 1;
        cl_kernel kernel;
        if (isXAxis) {
            kernel = math21_opencl_kernel_get<T1, T2>(
                    "math21_template_matrix_trans_reverse_axis_opencl_kernel", kernel_file_transpose, programs_trans);
        } else {
            kernel = math21_opencl_kernel_get<T1, T2>(
                    "math21_template_matrix_trans_reverse_y_axis_opencl_kernel", kernel_file_transpose, programs_trans);
        }
        math21_opencl_kernel_arg_set(kernel, n, x, y, d1, d2);
        math21_opencl_kernel_run(kernel, n);
    }

    template<typename T1, typename T2>
    void math21_template_tensor_swap_axes_24_in_d5_opencl(
            PtrVoidInWrapper x, PtrVoidWrapper y, NumN d1, NumN d2, NumN d3, NumN d4, NumN d5) {
//    x -= 1;
//    y -= 1;
        NumN n = d1 * d2 * d3 * d4 * d5;

        cl_kernel kernel = math21_opencl_kernel_get<T1, T2>(
                "math21_template_tensor_swap_axes_24_in_d5_opencl_kernel", kernel_file_transpose, programs_trans);
        math21_opencl_kernel_arg_set(kernel, n, x, y, d1, d2, d3, d4, d5);
        math21_opencl_kernel_run(kernel, n);
    }
}

#endif