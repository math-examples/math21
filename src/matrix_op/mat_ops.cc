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

#include "../op/files.h"
#include "operations.h"
#include "mat_ops.h"
#include "ten_ops.h"

namespace math21 {

    void math21_operator_mat_eye(MatR &A) {
        if (A.isEmpty()) {
            return;
        }
        A = 0;
        NumN r = xjmin(A.nrows(), A.ncols());
        NumN i;
        for (i = 1; i <= r; i++) A(i, i) = 1;
    }


#ifndef MATH21_FLAG_USE_CUDA

    void math21_cuda_test_02() {}

#endif

    NumB math21_operator_tensor_f_elementwise_is_compatible(const TenR &x1, const TenR &x2) {
        NumN n = xjmin(x1.dims(), x2.dims());
        for (NumN i = 1; i <= n; ++i) {
            if (x1.dim(i) != x2.dim(i) && x1.dim(i) > 1 && x2.dim(i) > 1) {
                return 0;
            }
        }
        return 1;
    }

    void math21_operator_tensor_f_elementwise_compatible_get_shape(const TenR &x1, const TenR &x2, VecN &d) {
        NumN n = xjmax(x1.dims(), x2.dims());
        d.setSize(n);
        NumN d1, d2;
        for (NumN i = 1; i <= n; ++i) {
            d1 = i > x1.dims() ? 1 : x1.dim(i);
            d2 = i > x2.dims() ? 1 : x2.dim(i);
            d(i) = xjmax(d1, d2);
        }
    }

    // see matlab compatible-array-sizes-for-basic-operations
    void math21_operator_tensor_f_elementwise_compatible_binary(const TenR &x1, const TenR &x2, TenR &y,
                                                                NumR (*f)(const NumR &x1, const NumR &x2),
                                                                const TenB &mask) {
        MATH21_ASSERT(math21_operator_tensor_f_elementwise_is_compatible(x1, x2))
        VecN d;
        math21_operator_tensor_f_elementwise_compatible_get_shape(x1, x2, d);
        if (!y.isSameSize(d)) {
            y.setSize(d);
        }
        VecN index;
        index.setSize(d.shape());
        index = 1;
        VecN index1;
        index1.setSize(x1.dims());
        index1 = 1;
        VecN index2;
        index2.setSize(x2.dims());
        index2 = 1;
        NumN n = d.size();
        NumB isUseMask = 0;
        if (mask.isSameSize(y.shape())) {
            isUseMask = 1;
        }
        while (1) {
            for (NumN i = 1; i <= n; ++i) {
                if (i <= x1.dims()) {
                    index1(i) = x1.dim(i) == 1 ? 1 : index(i);
                }
                if (i <= x2.dims()) {
                    index2(i) = x2.dim(i) == 1 ? 1 : index(i);
                }
            }
            if (isUseMask) {
                if (mask(index)) {
                    y(index) = f(x1(index1), x2(index2));
                }
            } else {
                y(index) = f(x1(index1), x2(index2));
            }
            if (math21_operator_container_increaseNumFromRight(d, index) == 0) {
                break;
            }
        }
    }

    void math21_operator_tensor_f_elementwise_unary(const TenR &x, TenR &y, NumR (*f)(const NumR &x)) {
        if (!y.isSameSize(x.shape())) {
            y.setSize(x.shape());
        }
        math21_operator_container_f_elementwise_unary(x, y, f);
    }

    // That x + y is legal when like x has shape (2,2) and y has shape (1,4)
    void math21_operator_tensor_as_container_f_elementwise_binary(const TenR &x1, const TenR &x2, TenR &y,
                                                                  NumR (*f)(const NumR &x1, const NumR &x2),
                                                                  const TenB &mask) {
        if (!y.isSameSize(x1.shape())) {
            y.setSize(x1.shape());
        }
        math21_operator_container_f_elementwise_binary(x1, x2, y, f, mask);
    }

    // That x + y is illegal when like x has shape (2,2) and y has shape (1,4)
    void math21_operator_tensor_f_elementwise_binary(const TenR &x1, const TenR &x2, TenR &y,
                                                     NumR (*f)(const NumR &x1, const NumR &x2), const TenB &mask) {
        if (x1.isSameSize(x2.shape())) {
            math21_operator_tensor_as_container_f_elementwise_binary(x1, x2, y, f, mask);
        } else {
            math21_operator_tensor_f_elementwise_compatible_binary(x1, x2, y, f, mask);
        }
    }

    void math21_operator_tensor_f_elementwise_binary(const NumR &x1, const TenR &x2, TenR &y,
                                                     NumR (*f)(const NumR &x1, const NumR &x2), const TenB &mask) {
        TenR v(1);
        v = x1;
        math21_operator_tensor_f_elementwise_binary(v, x2, y, f, mask);
    }

    void math21_operator_tensor_f_elementwise_binary(const TenR &x1, const NumR &x2, TenR &y,
                                                     NumR (*f)(const NumR &x1, const NumR &x2), const TenB &mask) {
        TenR v(1);
        v = x2;
        math21_operator_tensor_f_elementwise_binary(x1, v, y, f, mask);
    }

    void math21_operator_tensor_f_elementwise_ternary(const TenR &x1, const TenR &x2, const TenR &x3, TenR &y,
                                                      NumR (*f)(const NumR &x1, const NumR &x2, const NumR &x3)) {
        if (!y.isSameSize(x1.shape())) {
            y.setSize(x1.shape());
        }
        math21_operator_container_f_elementwise_ternary(x1, x2, x3, y, f);
    }

    void math21_operator_tensor_f_shrink_axes_to_index(NumN dims, const VecN &axes, VecN &index) {
        index.setSize(dims);
        if (axes.isEmpty()) {
            index = 1;
        } else {
            index = 0;
            math21_tool_assert(axes.size() <= dims);
            for (NumN i = 1; i <= axes.size(); ++i) {
                math21_tool_assert(axes(i) <= index.size());
                index(axes(i)) = 1;
            }
        }
    }

    // B = inverse of A
    void math21_operator_matrix_2_2_inverse(const MatR &A, MatR &B) {
        MATH21_ASSERT(A.isSameSize(2, 2))
        if (!B.isSameSize(2, 2)) {
            B.setSize(2, 2);
        }
        NumR a, b, c, d, det;
        det = A(1, 1) * A(2, 2) - A(2, 1) * A(1, 2);
        MATH21_ASSERT(det != 0)
        a = A(2, 2) / det;
        b = -A(2, 1) / det;
        c = -A(1, 2) / det;
        d = A(1, 1) / det;
        B =
                a, c,
                b, d;
    }

    // B = inverse of A
    void math21_operator_matrix_2_2_inverse(MatR &A) {
        math21_operator_matrix_2_2_inverse(A, A);
    }

    // Mij = minor of Aij
    NumR math21_operator_matrix_compute_minor(const MatR &A, NumN i, NumN j) {
        math21_tool_assert(0 && "not implement");
        return 0;
    }

// Cij = cofactor of Aij, Cij = (-1)^(i+j) * Mij, Mij is minor of Aij
    NumR math21_operator_matrix_compute_cofactor(const MatR &A, NumN i, NumN j) {
        math21_tool_assert(0 && "not implement");
        return 0;
    }

// co-factor matrix of A
    void math21_operator_matrix_compute_cofactor(const MatR &A, MatR &cofactor) {
        math21_tool_assert(0 && "not implement");
    }

    // A*X = det(A)*I = X*A
    // X is classical adjoint (also called adjugate) of A.
    // X = cofactor transpose of A
    void math21_operator_matrix_compute_classical_adjoint(const MatR &A, MatR &X) {
        math21_tool_assert(0 && "not implement");
    }

    // determinant, |A|
    NumR math21_operator_matrix_compute_det(const MatR &A) {
        math21_tool_assert(0 && "not implement");
        return 0;
    }

    // adjoint is the conjugate transpose of A
    void math21_operator_matrix_compute_adjoint(const MatC &A, MatC &X) {
        math21_tool_assert(0 && "not implement");
    }

    // see math21_operator_matrix_compute_minor
    // minor matrix of A
    void math21_operator_matrix_3x3_compute_minor(const MatR &A, MatR &minorM) {
        MATH21_ASSERT(A.isSameSize(3, 3));
        if (!minorM.isSameSize(3, 3)) {
            minorM.setSize(3, 3);
        }
        minorM(1, 1) = A(2, 2) * A(3, 3) - A(2, 3) * A(3, 2);
        minorM(1, 2) = A(2, 1) * A(3, 3) - A(2, 3) * A(3, 1);
        minorM(1, 3) = A(2, 1) * A(3, 2) - A(2, 2) * A(3, 1);
        minorM(2, 1) = A(1, 2) * A(3, 3) - A(1, 3) * A(3, 2);
        minorM(2, 2) = A(1, 1) * A(3, 3) - A(1, 3) * A(3, 1);
        minorM(2, 3) = A(1, 1) * A(3, 2) - A(1, 2) * A(3, 1);
        minorM(3, 1) = A(1, 2) * A(2, 3) - A(1, 3) * A(2, 2);
        minorM(3, 2) = A(1, 1) * A(2, 3) - A(1, 3) * A(2, 1);
        minorM(3, 3) = A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1);
    }

    // see math21_operator_matrix_compute_cofactor
    // co-factor matrix of A,
    // minor is minor of A
    void math21_operator_matrix_compute_cofactor_using_minor(const MatR &minorM, MatR &cofactor) {
        MATH21_ASSERT(minorM.dims() == 2)
        NumN nr, nc, ir, ic;
        nr = minorM.nrows();
        nc = minorM.ncols();
        if (!cofactor.isSameSize(nr, nc)) {
            cofactor.setSize(nr, nc);
        }
        for (ir = 1; ir <= nr; ++ir) {
            for (ic = 1; ic <= nc; ++ic) {
                cofactor(ir, ic) = minorM(ir, ic) * ((ir + ic) % 2 == 0 ? 1 : -1);
            }
        }
    }

    void math21_operator_matrix_compute_cofactor_using_minor(MatR &minor) {
        math21_operator_matrix_compute_cofactor_using_minor(minor, minor);
    }

    void math21_operator_matrix_3x3_compute_cofactor(const MatR &A, MatR &cofactor) {
        math21_operator_matrix_3x3_compute_minor(A, cofactor);
        math21_operator_matrix_compute_cofactor_using_minor(cofactor);
    }

    NumR math21_operator_matrix_compute_det_using_cofactor(const MatR &A, const MatR &cofactor) {
        NumN nc, ic;
        nc = A.ncols();
        NumR det = 0;
        for (ic = 1; ic <= nc; ++ic) {
            det += A(1, ic) * cofactor(1, ic);
        }
        return det;
    }

    // output: inverse of A
    void math21_operator_matrix_3x3_symmetric_inverse(const MatR &A, MatR &B) {
        math21_operator_matrix_3x3_compute_cofactor(A, B);
        NumR det = math21_operator_matrix_compute_det_using_cofactor(A, B);
        MATH21_ASSERT(det != 0, "singular matrix");
        // cofactor is symmetric
        math21_op_vector_kx_onto(1 / det, B);
    }

// output: inverse of A
    void math21_operator_matrix_3x3_inverse(const MatR &A, MatR &B) {
        math21_operator_matrix_3x3_compute_cofactor(A, B);
        NumR det = math21_operator_matrix_compute_det_using_cofactor(A, B);
        MatR adjugate;
        math21_op_matrix_transpose(B, adjugate);
        math21_op_mul(1 / det, adjugate, B);
    }

// see math21_operator_matrix_2_2_inverse
// https://mathworld.wolfram.com/Matrix1-Inverse.html
// https://mathworld.wolfram.com/Moore-PenroseMatrixInverse.html
// we use Moore-PenroseMatrixInverse, A: n*3, B: 3*n
// B = (A.t * A).inv * A.t
    void math21_operator_matrix_nx3_pseudoinverse(const MatR &A, MatR &B) {
        MATH21_ASSERT(A.dim(2) == 3);
        MatR AtA;
        math21_op_mat_mul(A, A, AtA, 1, 0);
        MatR inv;
        math21_operator_matrix_3x3_symmetric_inverse(AtA, inv);
        math21_op_mat_mul(inv, A, B, 0, 1);
    }

    // y = f(C), C = A+B, y in R => dy/dA = dy/dC, dy/dB = dy/dC
    void math21_operator_matrix_ad_reverse_add(const MatR &dC, MatR &dA_or_dB) {
        dA_or_dB.copyFrom(dC);
    }

    // y = f(C), C = A*B, y in R => dy/dA = dy/dC * B.t, dy/dB = A.t * dy/dC
    void math21_operator_matrix_ad_reverse_mul(
            const MatR &A, const MatR &B, const MatR &dC, MatR &dA_or_dB, NumN pos) {
        if (pos == 1) {
            math21_operator_matrix_mul_with_trans_option(1, dC, B, dA_or_dB, 0, 1);
        } else {
            math21_operator_matrix_mul_with_trans_option(1, A, dC, dA_or_dB, 1, 0);
        }
    }

    // References:
    // [An extended collection of matrix derivative results for forward and reverse mode algorithmic differentiation] by Mike Giles.
    // (https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf)
    // y = f(C), C = A.inv, y in R => dy/dA = -C.t * dy/dC * C.t
    void math21_operator_matrix_ad_reverse_inv(const MatR &C, const MatR &dC, MatR &dA) {
        MatR tmp;
        math21_operator_matrix_mul_with_trans_option(1, C, dC, tmp, 1, 0);
        math21_operator_matrix_mul_with_trans_option(-1, tmp, C, dA, 0, 1);
    }

    // y = f(C), C = |A|, y in R => dy/dA = dy/dC * C * A^(-T)
    void math21_operator_matrix_ad_reverse_det(const MatR &A, const MatR &C, const MatR &dC, MatR &dA) {
        MatR tmp;
        math21_operator_matrix_mul_with_trans_option(1, dC, C, tmp, 0, 0);
        MatR A_inv;
        math21_operator_inverse(A, A_inv);
        math21_operator_matrix_mul_with_trans_option(1, tmp, A_inv, dA, 0, 1);
    }

/**
 * This method is faster than other methods when b is large, like b>=20.
Y = X*H.t => dV/dH, where X = (x1.t; x2.t; ...), Y = (y1.t; y2.t; ...)
y = Hx, x.t = k1*(u.t, 1), y.t = k2*(v.t, 1), H(m, n) = 1 => dv/dH, here y in R(m), x in R(n)

calculate Jacobian dV/dH
dv/dH = (1/ym^2)*[ym*(y1'.t; ...; y(m-1)'.t) - (y1, ..., y(m-1)).t * (ym'.t; ...; ym'.t)] which has shape (m-1, m*n)
yi' = d(yi)/dH = (0; ...; 0; x.t; 0; ...) which has shape (m, n) and then reshaped to m*n, i = 1, ..., m.
=> dv/dH = (dv1/dH; ...; dvi/dH; ...; dv(m-1)/dH),
where dvi/dH = (0, ..., 0, w, 0, ..., 0, -w^2*yi).t * (x.t, ..., x.t) having shape (m, n), i = 1, ..., m-1.
and w = 1/ym
=> reshape(dV/dH) = (dV1/dH; ...; dVi/dH; ...; dV(m-1)/dH) having shape transformed from (m-1, m, b, n), where V = (V1, ..., V(m-1)) with shape b*(m-1)
where reshape(dVi/dH) = (0; ...; 0; w; 0; ...; 0; -w^2*Yi).t * (X; ...; X) having shape from (m, b, n), i = 1, ..., m-1.
and w = 1/Ym, Y = (Y1, ..., Ym) = k2 * (V, 1) with shape (b, m)
so dV/dH has shape (b*(m-1), m*n) from (b, m-1, m, n)
 *
 * */
    void math21_operator_derivative_project(const MatR &X, const MatR &Y, MatR &dH, NumB rmLast, NumN type) {
        MATH21_ASSERT(type == m21_flag_projection_affine || type == m21_flag_projection_projective);
        NumN b = X.nrows();
        NumN n = X.ncols();
        NumN m = Y.ncols();
        MATH21_ASSERT(b > 0 && n > 0 && m > 1 && b == Y.nrows());
        VecR w;
        math21_op_matrix_get_col(w, Y, m);
        for (NumN i = 1; i <= w.size(); ++i)w(i) = xjabs(w(i)) > MATH21_EPSILON ? 1 / w(i) : 0;
        MatR wX; // b*n
        math21_op_mul(w, X, wX);

        TenR wVX; // (m-1)*b*n
        if (type == m21_flag_projection_projective) {
            math21_op_square_onto(w);
            math21_op_mul_onto(-1, w); // -w^2
            MatR Yt;
            math21_op_matrix_transpose(Y, Yt);
            MatR Vt; // (m-1)*b
            math21_operator_share_tensor_rows(Yt, 0, m - 1, Vt);
            MatR w2t; // 1*b
            math21_operator_share_to_row_vector(w, w2t);
            math21_op_mul_onto_2(w2t, Vt);
            TenR X_share; // 1*b*n
            math21_operator_tensor_share_add_axis(X, X_share, 1);
            math21_op_mul(Vt, X_share, wVX);
        }

        MatR J;
        J.setSize(m - 1, m, b, n);
        J = 0;
        for (NumN i = 1; i <= m - 1; ++i) {
            MatR Ji;
            math21_operator_share_tensor_row_i(J, i, Ji);
            MatR Jii;
            math21_operator_share_tensor_row_i(Ji, i, Jii);
            Jii.assign(wX);
            if (type == m21_flag_projection_projective) {
                MatR wVXi;
                math21_operator_share_tensor_row_i(wVX, i, wVXi);
                MatR Jim;
                math21_operator_share_tensor_row_i(Ji, m, Jim);
                Jim.assign(wVXi);
            }
        }
        MatR dH_tmp, dH_tmp2;
        math21_op_tensor_move_axis(J, dH_tmp, 3, 1); // (m-1, m, b, n) -> (b, m-1, m, n)
        math21_operator_share_to_matrix(dH_tmp, dH_tmp2, 0, m * n);
        if (rmLast) { // (b*(m-1), m*n-1)
            NumN n_paras;
            if (type == m21_flag_projection_affine) {
                n_paras = (m - 1) * n;
            } else {
                n_paras = m * n - 1;
            }
            dH.setSize(b * (m - 1), n_paras);
            math21_op_matrix_sub_region_tl_set(dH_tmp2, dH);
        } else {
            dH = dH_tmp2;
        }
    }
}