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

#include "inner_cc.h"
#include "svd.h"
#include "eigen_sym.h"
#include "gje.h"
#include "ops.h"
#include "equations.h"

namespace math21 {
    // solve A*X=B;
    NumB math21_equation_solve_linear_equation_simple(const MatR &A, const MatR &B, MatR &X) {
        MatR A_inv;
        A_inv = A;
        if (!math21_operator_container_isEqual(X.shape(), B.shape())) {
            X.setSize(B.shape());
        }
        X.assign(B);
        numerical_recipes::GaussJordanElimination gje;
        if (!gje.solve(A_inv, X)) {
            return 0;
        }
        return 1;
    }

    /**
    Ax = b, A: m*n, m>=n, rank(A) = n
    <=> min||Ax-b||
    <=> min E= (Ax-b).t * (Ax-b)
    <=> dE/dx = 0
    <=> x = (A.t * A).inv * A.t * b
    see math21_equation_solve_linear_equation_svd, math21_operator_matrix_nx3_pseudoinverse
     */
    NumB math21_equation_solve_linear_least_squares_pseudoinverse(const MatR &A, const MatR &B, MatR &X) {
        MatR AtA;
        math21_op_mat_mul(A, A, AtA, 1, 0);
        MatR inv;
        math21_operator_inverse(AtA, inv);
        MatR AtB;
        math21_op_mat_mul(A, B, AtB, 1, 0);
        math21_op_mat_mul(inv, AtB, X, 0, 0);
        return 1;
    }

    /**
    min||<w, Ax-b>||
    <=> min ||(WA)x - (Wb)||, where W = diag(w)
     */
    NumB math21_equation_solve_weighted_least_squares_pseudoinverse(
            const MatR &A, const MatR &B, const VecR &w, MatR &X) {
        MatR W;
        math21_operator_matrix_diag_do(w, W);
        MatR WA, WB;
        math21_op_mat_mul(W, A, WA);
        math21_op_mat_mul(W, B, WB);
        return math21_equation_solve_linear_least_squares_pseudoinverse(WA, WB, X);
    }

    // see https://personalpages.manchester.ac.uk/staff/timothy.f.cootes/MathsMethodsNotes/L3_linear_algebra3.pdf
    NumB math21_equation_solve_linear_equation_svd(const MatR &A, const MatR &B, MatR &X) {
        NumN m, n, p;
        m = A.nrows();
        n = A.ncols();
        p = B.ncols();
        MATH21_ASSERT(m >= n, "Solving under-determined linear systems not supported!");
        ShiftedMatR As, Bs, Xs;
        As.setTensor(A);
        Bs.setTensor(B);
        if (B.dims() == 1) {
            Xs.setSize(n);
        } else {
            Xs.setSize(n, p);
        }
        numerical_recipes::SVD svd(As);
        if (B.dims() == 1) {
            svd.solve_vec(Bs, Xs);
        } else {
            svd.solve_mat(Bs, Xs);
        }
        X = Xs.getTensor();
        return 1;
    }

    /**
    solve Ax = 0, A in M(F, m, n), m>=n, rank(A) <n
    <=> min ||Ax||, s.t. ||x||=1
    <=> x is the last column of V, where A = U*W*V' (svd)
     */
    NumB math21_equation_solve_homogeneous_linear_equation_svd(const MatR &A, MatR &x) {
        MatR u, w, v;
        math21_operator_svd_real(A, u, w, v);
        math21_op_matrix_get_col(x, v, v.ncols());
        return 1;
    }

    // A*V = V*D => A = V*D*V.t => A.inv = V*D.inv*V.t, where V right eigenvectors matrix, D eigenvalues matrix.
    // Ax=b => x = A.inv * b => x = V*D.inv*V.t * b
    // A is real symmetric matrix required by method Symmeig.
    NumB math21_equation_solve_linear_equation_eigen_real_sym(const MatR &A, const MatR &B, MatR &X) {
        if (math21_global_is_debug()) {
            MATH21_ASSERT(math21_operator_mat_is_symmetric(A));
        }
        ShiftedMatR As;
        As.setTensor(A);
//        numerical_recipes::Symmeig module(As); // not good
        numerical_recipes::Jacobi module(As);
        const VecR &D = module.get_eigenvalues().getTensor();
        const MatR &V = module.get_eigenvectors().getTensor();

        MatR tmp;
        math21_op_mat_mul(V, B, tmp, 1); // B can be matrix
        math21_op_divide_onto_1(tmp, D);
        math21_op_mat_mul(V, tmp, X);
        return 1;
    }

    NumB math21_equation_solve_linear_equation_with_option(const MatR &A, const MatR &B, MatR &X, NumN method) {
        NumB flag;
        if (method == m21_mat_decomp_eigenvalue_sym) {
            flag = math21_equation_solve_linear_equation_eigen_real_sym(A, B, X);
        } else if (method == m21_mat_decomp_svd) {
            flag = math21_equation_solve_linear_equation_svd(A, B, X);
        } else {
            flag = math21_equation_solve_linear_equation_simple(A, B, X);
        }
//        NumB test = 1;
        NumB test = 0;
        if (test) {
            MatR B_est;
            math21_op_mat_mul(A, X, B_est);
            if (test == 1) {
                NumR dist = math21_op_vector_distance(B, B_est, 2);
                NumR length = math21_op_vector_norm(B, 2);
                NumR err = dist / length;
                m21log("err", err);
            } else {
                MatR dist_r;
                math21_op_subtract(B, B_est, dist_r);
                math21_op_abs_onto(dist_r);
                math21_op_divide_onto_1(dist_r, B);
                dist_r.log("dist_r");
            }
        }
        return flag;
    }
}