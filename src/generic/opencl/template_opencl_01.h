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
    const static std::string kernel_file = "generic_01.kl";
    static Map_<std::string, std::shared_ptr<m21clprogram>> thePrograms;

    template<typename T>
    void math21_template_vector_kx_opencl(NumN n, T k, PtrVoidWrapper x, NumN stride_x) {
        cl_kernel kernel = math21_opencl_kernel_get<T>(
                "math21_template_vector_kx_opencl_kernel", kernel_file, thePrograms);
        math21_opencl_kernel_arg_set(kernel, n, k, x, stride_x);
        math21_opencl_kernel_run(kernel, n);
    }

    template<typename T>
    void math21_template_vector_kx_add_y_opencl(
            NumN n, T k, PtrVoidInWrapper x, NumN stride_x, PtrVoidWrapper y, NumN stride_y) {
//    x -= 1;
//    y -= 1;
        cl_kernel kernel = math21_opencl_kernel_get<T>(
                "math21_template_vector_kx_add_y_opencl_kernel", kernel_file, thePrograms);
        math21_opencl_kernel_arg_set(kernel, n, k, x, stride_x, y, stride_y);
        math21_opencl_kernel_run(kernel, n);
    }

    template<typename T>
    void math21_template_vector_xy_opencl(NumN n, PtrVoidInWrapper x, NumN stride_x, PtrVoidWrapper y,
                                          NumN stride_y) {
//    x -= 1;
//    y -= 1;
        cl_kernel kernel = math21_opencl_kernel_get<T>(
                "math21_template_vector_xy_opencl_kernel", kernel_file, thePrograms);
        math21_opencl_kernel_arg_set(kernel, n, x, stride_x, y, stride_y);
        math21_opencl_kernel_run(kernel, n);
    }

    template<typename T>
    void math21_template_vector_sin_opencl(NumN n, PtrVoidInWrapper x, PtrVoidWrapper y) {
        cl_kernel kernel = math21_opencl_kernel_get<T>(
                "math21_template_vector_sin_opencl_kernel", kernel_file, thePrograms);
        math21_opencl_kernel_arg_set(kernel, n, x, y);
        math21_opencl_kernel_run(kernel, n);
    }

    template<typename T>
    void math21_template_vector_cos_opencl(NumN n, PtrVoidInWrapper x, PtrVoidWrapper y) {
        cl_kernel kernel = math21_opencl_kernel_get<T>(
                "math21_template_vector_cos_opencl_kernel", kernel_file, thePrograms);
        math21_opencl_kernel_arg_set(kernel, n, x, y);
        math21_opencl_kernel_run(kernel, n);
    }

    template<typename T>
    void math21_template_tensor_3d_swap_row_in_d2_opencl(
            NumN n, PtrVoidWrapper x, NumN i, NumN j, NumN d1, NumN d2, NumN d3) {
        //    x -= 1;
        cl_kernel kernel = math21_opencl_kernel_get<T>(
                "math21_template_tensor_3d_swap_row_in_d2_opencl_kernel", kernel_file, thePrograms);
        math21_opencl_kernel_arg_set(kernel, n, x, i, j, d1, d2, d3);
        math21_opencl_kernel_run(kernel, n);
    }

    template<typename T>
    void math21_template_vector_addToC_opencl(NumN n, PtrVoidInWrapper A,
                                              PtrVoidInWrapper B, PtrVoidWrapper C) {
        cl_kernel kernel = math21_opencl_kernel_get<T>(
                "math21_template_vector_addToC_opencl_kernel", kernel_file, thePrograms);
        math21_opencl_kernel_arg_set(kernel, n, A, B, C);
        math21_opencl_kernel_run(kernel, n);
    }

    template<typename T>
    void math21_template_vector_mulToC_opencl(NumN n, PtrVoidInWrapper A,
                                              PtrVoidInWrapper B, PtrVoidWrapper C) {
        cl_kernel kernel = math21_opencl_kernel_get<T>(
                "math21_template_vector_mulToC_opencl_kernel", kernel_file, thePrograms);
        math21_opencl_kernel_arg_set(kernel, n, A, B, C);
        math21_opencl_kernel_run(kernel, n);
    }

    template<typename T>
    void math21_template_vector_broadcast_in_dn_opencl(NumN n, PtrVoidInWrapper x, PtrVoidWrapper y,
                                                       NumN dims_x, PtrNInWrapper dx,
                                                       NumN dims_y, PtrNInWrapper dy) {
        // todo: uncomment out using opencl svm when opencl>=2
//    x -= 1;
//    y -= 1;
//    dx -= 1;
//    dy -= 1;

        cl_kernel kernel = math21_opencl_kernel_get<T>(
                "math21_template_vector_broadcast_in_dn_opencl_kernel", kernel_file, thePrograms);
        math21_opencl_kernel_arg_set(kernel, n, x, y, dims_x, dx, dims_y, dy);
        math21_opencl_kernel_run(kernel, n);
    }

    template<typename T>
    void math21_template_optimization_adam_update_part_2_opencl(
            NumN n, PtrVoidWrapper x, PtrVoidInWrapper m,
            PtrVoidInWrapper v,
            T beta1, T beta2, T alpha, T eps, NumN t) {
//    x -= 1;
//    m -= 1;
//    v -= 1;

        cl_kernel kernel = math21_opencl_kernel_get<T>(
                "math21_template_optimization_adam_update_part_2_opencl_kernel", kernel_file, thePrograms);
        math21_opencl_kernel_arg_set(kernel, n, x, m, v, beta1, beta2, alpha, eps, t);
        math21_opencl_kernel_run(kernel, n);
    }
}

#endif