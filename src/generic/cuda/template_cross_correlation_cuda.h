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
#include "common.h"
#include "../kernels/generic_cross_correlation.kl"

namespace math21 {
    // d: dilation, k: kernel, p: pad, s: stride
    template<typename T>
    void math21_template_cross_correlation_X_to_X_prime_cuda(
            const T *X, T *X_prime,
            int nch_X, int nr_X, int nc_X,
            int nr_k, int nc_k,
            int nr_p, int nc_p,
            int nr_s, int nc_s,
            int nr_d, int nc_d) {
        int nr_k_ext = (nr_k - 1) * nr_d + 1;
        int nc_k_ext = (nc_k - 1) * nc_d + 1;
        int nc_X_prime_1 = (nr_X + 2 * nr_p - nr_k_ext) / nr_s + 1;
        int nc_X_prime_2 = (nc_X + 2 * nc_p - nc_k_ext) / nc_s + 1;
        int n = nch_X * nc_X_prime_1 * nc_X_prime_2;

        math21_cuda_kernel_arg_set_and_run(
                n,
                math21_template_cross_correlation_X_to_X_prime_cuda_kernel<T>,
                n, X, X_prime, nr_X, nc_X, nr_k, nc_k,
                nr_p, nc_p, nr_s, nc_s, nr_d, nc_d,
                nc_X_prime_1, nc_X_prime_2);
    }

    // d: dilation, k: kernel, p: pad, s: stride
    template<typename T>
    void math21_template_cross_correlation_dX_prime_to_dX_cuda(
            const T *dX_prime, T *dX,
            int nch_X, int nr_X, int nc_X,
            int nr_k, int nc_k,
            int nr_p, int nc_p,
            int nr_s, int nc_s,
            int nr_d, int nc_d) {
        int nr_k_ext = (nr_k - 1) * nr_d + 1;
        int nc_k_ext = (nc_k - 1) * nc_d + 1;
        int nc_X_prime_1 = (nr_X + 2 * nr_p - nr_k_ext) / nr_s + 1;
        int nc_X_prime_2 = (nc_X + 2 * nc_p - nc_k_ext) / nc_s + 1;
        int n = nch_X * nr_X * nc_X;

        if (nr_d == 1 && nc_d == 1) {
            math21_cuda_kernel_arg_set_and_run(
                    n,
                    math21_template_cross_correlation_dX_prime_to_dX_without_dilation_addto_cuda_kernel<T>,
                    n, dX_prime, dX, nr_X, nc_X, nr_k, nc_k,
                    nr_p, nc_p, nr_s, nc_s, nc_X_prime_1, nc_X_prime_2);
        } else {
            math21_cuda_kernel_arg_set_and_run(
                    n,
                    math21_template_cross_correlation_dX_prime_to_dX_addto_cuda_kernel<T>,
                    n, dX_prime, dX, nr_X, nc_X, nr_k, nc_k,
                    nr_p, nc_p, nr_s, nc_s, nr_d, nc_d,
                    nc_X_prime_1, nc_X_prime_2);
        }
    }
}