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

#include "template_cross_correlation_cpu.h"
#include "op_cpu.h"

using namespace math21;

void math21_generic_cross_correlation_X_to_X_prime_cpu(
        const void *X, void *X_prime,
        NumN nch_X, NumN nr_X, NumN nc_X,
        NumN nr_k, NumN nc_k,
        NumN nr_p, NumN nc_p,
        NumN nr_s, NumN nc_s,
        NumN nr_d, NumN nc_d, NumN type) {
    if (type == m21_type_NumR) {
        math21_template_cross_correlation_X_to_X_prime_cpu(
                (const NumR *) X, (NumR *) X_prime, nch_X, nr_X, nc_X,
                nr_k, nc_k, nr_p, nc_p, nr_s, nc_s, nr_d, nc_d);
    } else if (type == m21_type_NumR32) {
        math21_template_cross_correlation_X_to_X_prime_cpu(
                (const NumR32 *) X, (NumR32 *) X_prime, nch_X, nr_X, nc_X,
                nr_k, nc_k, nr_p, nc_p, nr_s, nc_s, nr_d, nc_d);
    } else if (type == m21_type_NumR64) {
        math21_template_cross_correlation_X_to_X_prime_cpu(
                (const NumR64 *) X, (NumR64 *) X_prime, nch_X, nr_X, nc_X,
                nr_k, nc_k, nr_p, nc_p, nr_s, nc_s, nr_d, nc_d);
    } else {
        math21_tool_assert(0);
    }
}

void math21_generic_cross_correlation_dX_prime_to_dX_cpu(
        const void *dX_prime, void *dX,
        NumN nch_X, NumN nr_X, NumN nc_X,
        NumN nr_k, NumN nc_k,
        NumN nr_p, NumN nc_p,
        NumN nr_s, NumN nc_s,
        NumN nr_d, NumN nc_d, NumN type) {
    if (type == m21_type_NumR) {
        math21_template_cross_correlation_dX_prime_to_dX_cpu(
                (const NumR *) dX_prime, (NumR *) dX, nch_X, nr_X, nc_X,
                nr_k, nc_k, nr_p, nc_p, nr_s, nc_s, nr_d, nc_d);
    } else if (type == m21_type_NumR32) {
        math21_template_cross_correlation_dX_prime_to_dX_cpu(
                (const NumR32 *) dX_prime, (NumR32 *) dX, nch_X, nr_X, nc_X,
                nr_k, nc_k, nr_p, nc_p, nr_s, nc_s, nr_d, nc_d);
    } else if (type == m21_type_NumR64) {
        math21_template_cross_correlation_dX_prime_to_dX_cpu(
                (const NumR64 *) dX_prime, (NumR64 *) dX, nch_X, nr_X, nc_X,
                nr_k, nc_k, nr_p, nc_p, nr_s, nc_s, nr_d, nc_d);
    } else {
        math21_tool_assert(0);
    }
}