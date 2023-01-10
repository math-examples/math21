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

#define MATH21_IS_FROM_CPU

#include "../kernels/generic_cross_correlation.kl"

#undef MATH21_IS_FROM_CPU

namespace math21 {

    // X -> X_prime
// n_common = l.size*l.size*l.c/l.groups
// X_prime shape: (nch_X * nr_K * nc_K ) * nc_Y_m
// X_prime shape: (nch_K * nr_K * nc_K ) * nc_Y_m
// X_prime shape: n_common * nc_Y_m
// X_prime shape: n_common * (nc_X_prime_1 * nc_X_prime_2)
// X_prime shape: nr_X_prime * (nc_X_prime_1 * nc_X_prime_2)
// X_prime size: (nch_X * nr_K * nc_K ) * (nc_X_prime_1 * nc_X_prime_2)
    // d: dilation, k: kernel, p: pad, s: stride
    template<typename T>
    void math21_template_cross_correlation_X_to_X_prime_cpu(
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

        NumN id;
#pragma omp parallel for
        for (id = 1; id <= n; ++id) {
            math21_template_cross_correlation_X_to_X_prime_cpu_kernel(
                    n, X, X_prime, nr_X, nc_X, nr_k, nc_k,
                    nr_p, nc_p, nr_s, nc_s, nr_d, nc_d,
                    nc_X_prime_1, nc_X_prime_2, id);
        }
    }

    // dX_prime -> dX, dX_prime: n_common*nc_Y_m
    // n_common = l.size*l.size*l.c/l.groups
    // d: dilation, k: kernel, p: pad, s: stride
    template<typename T>
    void math21_template_cross_correlation_dX_prime_to_dX_cpu(
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

        NumN id;
        if (nr_d == 1 && nc_d == 1) {
#pragma omp parallel for
            for (id = 1; id <= n; ++id) {
                math21_template_cross_correlation_dX_prime_to_dX_without_dilation_addto_cpu_kernel(
                        n, dX_prime, dX, nr_X, nc_X, nr_k, nc_k,
                        nr_p, nc_p, nr_s, nc_s, nc_X_prime_1, nc_X_prime_2, id);
            }
        } else {
#pragma omp parallel for
            for (id = 1; id <= n; ++id) {
                math21_template_cross_correlation_dX_prime_to_dX_addto_cpu_kernel(
                        n, dX_prime, dX, nr_X, nc_X, nr_k, nc_k,
                        nr_p, nc_p, nr_s, nc_s, nr_d, nc_d,
                        nc_X_prime_1, nc_X_prime_2, id);
            }
        }
    }
}