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

#include "inner.h"
#include "../../gpu/files.h"
#include "../kernels/generic_01.kl"
#include "common.h"

namespace math21 {

    // x = k*x
    template<typename T>
    void math21_template_vector_kx_cuda(NumN n, T k, T *x, NumN stride_x) {
        math21_cuda_kernel_arg_set_and_run(
                n,
                math21_template_vector_kx_cuda_kernel<T>,
                n, k, x, stride_x);
    }

    template<typename T>
    void math21_template_vector_kx_add_y_cuda(NumN n, T k, const T *x, NumN stride_x, T *y, NumN stride_y) {
        x -= 1;
        y -= 1;
        math21_cuda_kernel_arg_set_and_run(
                n,
                math21_template_vector_kx_add_y_cuda_kernel<T>,
                n, k, x, stride_x, y, stride_y);
    }

    template<typename T>
    void math21_template_vector_xy_cuda(NumN n, const T *x, NumN stride_x, T *y, NumN stride_y) {
        x -= 1;
        y -= 1;
        math21_cuda_kernel_arg_set_and_run(
                n,
                math21_template_vector_xy_cuda_kernel<T>,
                n, x, stride_x, y, stride_y);
    }

    template<typename T>
    void math21_template_vector_sin_cuda(NumN n, const T *x, T *y) {
        math21_cuda_kernel_arg_set_and_run(
                n,
                math21_template_vector_sin_cuda_kernel<T>,
                n, x, y);
    }

    template<typename T>
    void math21_template_vector_cos_cuda(NumN n, const T *x, T *y) {
        math21_cuda_kernel_arg_set_and_run(
                n,
                math21_template_vector_cos_cuda_kernel<T>,
                n, x, y);
    }

    template<typename T>
    void math21_template_tensor_3d_swap_row_in_d2_cuda(NumN n, T *x, NumN i, NumN j, NumN d1, NumN d2, NumN d3) {
        x -= 1;
        math21_cuda_kernel_arg_set_and_run(
                n,
                math21_template_tensor_3d_swap_row_in_d2_cuda_kernel<T>,
                n, x, i, j, d1, d2, d3);
    }

    template<typename T>
    void math21_template_vector_addToC_cuda(NumN n, const T *A, const T *B, T *C) {
        math21_cuda_kernel_arg_set_and_run(
                n,
                math21_template_vector_addToC_cuda_kernel<T>,
                n, A, B, C);
    }

    template<typename T>
    void math21_template_vector_mulToC_cuda(NumN n, const T *A, const T *B, T *C) {
        math21_cuda_kernel_arg_set_and_run(
                n,
                math21_template_vector_mulToC_cuda_kernel<T>,
                n, A, B, C);
    }

    // todo: use index 1 for x, y
    // a special kind of sub
    // x is sub-tensor of y
    template<typename T>
    void math21_template_vector_broadcast_in_dn_cuda(NumN n, const T *x, T *y,
                                                     NumN dims_x, const NumN *dx, NumN dims_y, const NumN *dy) {
        x -= 1;
        y -= 1;
        dx -= 1;
        dy -= 1;
        math21_cuda_kernel_arg_set_and_run(
                n,
                math21_template_vector_broadcast_in_dn_cuda_kernel<T>,
                n, x, y, dims_x, dx, dims_y, dy);
    }

    template<typename T>
    void math21_template_optimization_adam_update_part_2_cuda(NumN n, T *x, const T *m, const T *v,
                                                              T beta1, T beta2, T alpha, T eps, NumN t) {
        x -= 1;
        m -= 1;
        v -= 1;
        math21_cuda_kernel_arg_set_and_run(
                n,
                math21_template_optimization_adam_update_part_2_cuda_kernel<T>,
                n, x, m, v, beta1, beta2, alpha, eps, t);
    }
}