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

namespace math21 {

    enum {
        m21_mat_decomp_none = 0,
        m21_mat_decomp_eigenvalue_sym,
        m21_mat_decomp_svd,
    };

    // solve A*X=B
    NumB math21_equation_solve_linear_equation_simple(const MatR &A, const MatR &B, MatR &X);

    NumB math21_equation_solve_linear_least_squares_pseudoinverse(const MatR &A, const MatR &B, MatR &X);

    NumB math21_equation_solve_weighted_least_squares_pseudoinverse(const MatR &A, const MatR &B, const VecR &w, MatR &X);

    NumB math21_equation_solve_linear_equation_svd(const MatR &A, const MatR &B, MatR &X);

    NumB math21_equation_solve_homogeneous_linear_equation_svd(const MatR &A, MatR &x);

    NumB math21_equation_solve_linear_equation_with_option(const MatR &A, const MatR &B, MatR &X, NumN method = m21_mat_decomp_none);
}