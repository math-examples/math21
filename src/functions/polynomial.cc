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
#include "polynomial.h"

namespace math21 {

    polynomial::polynomial() : Functional() {
        A.setSize(1);
        x0.setSize(1);
//        x0 = 0.8;
        x0 = 300;
    }

    //y = (x-20)^2
    NumR polynomial::valueAt(const VecR &x) {
        return xjsquare(x(1) - 40);
    }

    NumN polynomial::getXDim() {
        return 1;
    }

    const VecR &polynomial::derivativeValueAt(const VecR &x) {
        A(1) = 2 * (x(1) - 40);
        return A;
    }

    const VecR &polynomial::getX0() {
        return x0;
    }


// cubic curve (cubic polynomial curve)
// x = p(t) = (a, b, c, d) * (t^3, t^2, t, 1).t = A*T
// so x.t = T.t * A.t = (t^3, t^2, t, 1) * A.t

// cubic spline (piecewise cubic polynomial curve)
// x = f(t) = p(t), if x in [pi, p(i+1))
// x.t = p(t).t = T.t * A(i).t, if x in [pi, p(i+1)), where A(i) is matrix

// cubic Hermite curve
// given (p1, p2, dp1, dp2)
// (p1, p2, dp1, dp2).t = Binv *A.t,
// where Binv =
//           (0, 0, 0, 1,
//            1, 1, 1, 1,
//            0, 0, 1, 0,
//            3, 2, 1, 0)
// so x.t = T.t * A.t = T.t * B * C,
// where B = B(Hermite) :=(Binv).inv =
//                 (2, -2,  1,  1,
//                 -3,  3, -2, -1,
//                  0,  0,  1,  0,
//                  1,  0,  0,  0)
// B is basis matrix, C = (p1, p2, dp1, dp2).t is control matrix

// cubic Hermite spline (spline type is Hermite)

// cubic Bezier curve
// given (P1, P2, P3, P4), P1, P4 on curve
// => p1 = P1, p2 = P4, dp1 = 3*(P2-P1), dp2 = 3*(P4-P3)
// => (p1, p2, dp1, dp2).t = D * (P1, P2, P3, P4).t,
// where D = (1, 0,  0, 0,
//            0, 0,  0, 1,
//           -3, 3,  0, 0,
//            0, 0, -3, 3)
// so x.t = T.t * A.t = T.t * B * (P1, P2, P3, P4).t,,
// where B = B(Bezier) := B(Hermite) * D
// B = (-1,  3, -3, 1,
//       3, -6,  3, 0,
//      -3,  3,  0, 0,
//       1,  0,  0, 0)

// cubic Bezier spline (spline type is Bezier)

// cubic Catmull-Rom curve
// given (p0, p1, p2, p3)
// => p1 = p1, p2 = p2, dp1 = s*(p2-p0), dp2 = s*(p3-p1), where s = 0.5
// => (p1, p2, dp1, dp2).t = D * (p0, p1, p2, p3).t,
// where D = (0,  1,  0, 0,
//            0,  0,  1, 0,
//           -s,  0,  s, 0,
//            0, -s,  0, s)
// so x.t = T.t * A.t = T.t * B * (p0, p1, p2, p3).t,,
// where B = B(Catmull-Rom) := B(Hermite) * D
// B = (  -s,  2-s,    s-2,  s,
//       2*s,  s-3,  3-2*s, -s,
//        -s,    0,      s,  0,
//         0,    1,      0,  0)

// cubic Catmull-Rom spline
// given n+2 points: p0, p1, ..., pn, p(n+1), n>=2
// x = f(t) = p(t), if x in [pi, p(i+1)), i = 1, ..., n-1
// x.t = p(t).t = T.t * A(i).t, if x in [pi, p(i+1)), where A(i) is matrix
// A(i).t = B(Catmull-Rom) * (p(i-1), pi, p(i+1), p(i+2)).t

// natural cubic spline

// cubic B-spline

    NumR math21_function_polynomial(const VecR &k, NumR x) {
        VecR v(k.size());
        for (NumN i = 1; i <= v.size(); ++i) {
            v(i) = xjpow(x, i - 1);
        }
        return math21_op_vector_inner_product(k, v);
    }

// y = f(x) = sum(Kij * x^j), j=0, ..., 3. if x in [pi, p(i+1))
    void math21_function_cubic_spline_num(const MatR &K, const VecR &p, NumR x, NumR &y) {
        MATH21_ASSERT(K.dim(1) + 1 == p.size());
        NumN i = math21_operator_container_cdf_get(p, x, 0);
        VecR k;
        math21_op_matrix_get_row(k, K, i);
        y = math21_function_polynomial(k, x);
    }

    /**
    MatR data;
    data.setSize(4, 2);
    data =
            0, 100,
            2, -20,
            5, -100,
            8, 0;
    NumR k1, k2;
    k1 = 21;
    k2 = -100;
    MatR K;
    math21_fit_2d_cubic_spline_natural(data, k1, k2, K);

    NumN n = 100;
    MatR X(n);
    math21_operator_container_set_value(X, -1.0, 0.1);
    VecR p;
    math21_op_matrix_get_col(p, data, 1);
    VecR Y;
    math21_function_cubic_spline(K, p, X, Y);
    */
    void math21_function_cubic_spline(const MatR &K, const VecR &p, const VecR &x, VecR &y) {
        y.setSize(x.size());
        for (NumN i = 1; i <= x.size(); ++i) {
            math21_function_cubic_spline_num(K, p, x(i), y.at(i));
        }
    }

    // x.t = f(t) = T.t * A.t = (t^n, ..., t^j, ...,  t, 1) * A.t, where A.t = (an.t; ...; aj.t; ...; a1.t; a0.t)
    // f^(i)(t) = (t^(n-i), t^(n-i-1), ...,  t, 1)*B.t,
    // where B.t = v*(an.t; ...; aj.t; ...; ai.t), v.t = (P(n, i), ..., P(j, i), ..., P(i, i)),  P(j, i) = j!/(j-i)!
    void math21_function_derivative_ith_order_parametric_polynomial(const MatR &At, MatR &Bt, NumN i) {
        NumN n = At.nrows() - 1;
        if (i > n) {
            Bt.setSize(1, At.ncols());
            Bt = 0;
            return;
        }
        MatR At_part;
        math21_operator_share_tensor_rows(At, 0, n + 1 - i, At_part);
        VecR v(n + 1 - i);
        for (NumN j = n; j >= i; --j) {
            v.at(n + 1 - j) = xjfactorial_similar(j, i);
        }
        math21_op_mul(v, At_part, Bt);
    }

    // so x.t = T.t * A.t = (t^3, t^2, t, 1) * A.t
    void math21_function_parametric_polynomial_subroutine(const MatR &At, const MatR &Tt, MatR &xt) {
        math21_op_mat_mul(Tt, At, xt);
    }

    void math21_function_parametric_spline_subroutine(const TenR &At0, const MatR &Tt, MatR &xt) {
        MATH21_ASSERT(At0.dims() == 2 || At0.dims() == 3);
        TenR At1;
        if (At0.dims() == 2) {
            math21_operator_tensor_share_add_axis(At0, At1, 1);
        }
        const TenR &At = math21_tool_choose_non_empty(At1, At0);

        NumN n = At.dim(1);
        xt.setSize(n * Tt.nrows(), At.dim(3));

        MatR Ait, xit;
        for (NumN i = 1; i <= n; ++i) {
            math21_operator_share_tensor_row_i(At, i, Ait);
            math21_operator_share_tensor_rows(xt, (i - 1) * Tt.nrows(), Tt.nrows(), xit);
            math21_function_parametric_polynomial_subroutine(Ait, Tt, xit);
        }
    }

    NumN math21_function_parametric_spline_get_degree_n(const TenR &At){
        NumN n;
        if (At.dims() == 2) {
            n = At.dim(1) - 1;
        }else{
            MATH21_ASSERT(At.dims()==3);
            n = At.dim(2) - 1;
        }
        return n;
    }

    void math21_function_parametric_spline_evaluate(const TenR &At, const VecR &t, MatR &xt) {
        NumN n = math21_function_parametric_spline_get_degree_n(At);
        MatR Tt(t.size(), n + 1);
        VecR v;
        for (NumN i = 1; i <= Tt.dim(2); ++i) {
            math21_op_pow(t, i - 1, v);
            math21_op_matrix_set_col(v, Tt, Tt.dim(2) + 1 - i);
        }
        math21_function_parametric_spline_subroutine(At, Tt, xt);
    }

    // (t^3, t^2, t, 1)
    void math21_function_parametric_cubic_spline_evaluate(const TenR &At, const VecR &t, MatR &xt) {
        NumN n = math21_function_parametric_spline_get_degree_n(At);
        MATH21_ASSERT(n==3);
        math21_function_parametric_spline_evaluate(At, t, xt);
    }

    // find a, s.t. y = x.t * a, where a in R^4, x.t = (1, x, x^2, x^3)
    /**
     * MatR data;
        data.setSize(4, 2);
        data =
                -1, 2,
                0, 0,
                1, -2,
                2, 0;
        VecR a;
        math21_fit_2d_cubic_curve(data, a);
        // a = [0, -2.67, 0, 0.667]
     * */
    void math21_fit_2d_cubic_curve(const MatR &data, VecR &a) {
        MATH21_ASSERT(data.nrows() == 4)
        MatR M, b;
        M.setSize(4, 4);
        b.setSize(4);
        for (NumN i = 1; i <= data.dim(1); ++i) {
            NumR x, y;
            x = data(i, 1);
            y = data(i, 2);
            M(i, 1) = 1;
            M(i, 2) = x;
            M(i, 3) = x * x;
            M(i, 4) = x * x * x;
            b(i) = y;
        }
        math21_equation_solve_linear_equation_simple(M, b, a);
    }

    // (x1, y1), ..., (xn, yn), (x(n+1), y(n+1)) => f1, ..., fn
    // where fi = ai + bi*xi + ci*xi^2 + di*xi^3
    //
    // fi(xi) = yi, fi(x(i+1)) = y(i+1),
    // fi'(x(i+1)) = f(i+1)'(x(i+1)), fi''(x(i+1)) = f(i+1)''(x(i+1))
    // i = 1, ..., n-1
    // => 4(n-1) conditions
    // fn(xn) = yn, fn(x(n+1)) = y(n+1), fn'(x(n+1)) = k2, f1'(x1) = k1 => 4 conditions
    // So we have 4n conditions and 4n parameters.
    // See https://people.cs.clemson.edu/~dhouse/courses/405/notes/splines.pdf
    void math21_fit_2d_cubic_spline_natural(const MatR &data, NumR k1, NumR k2, MatR &x) {
        MATH21_ASSERT(data.nrows() >= 4)
        NumN n = data.nrows() - 1;
        MatR A, b;
        A.setSize(4 * n, 4 * n);
        b.setSize(4 * n);
        A = 0;
        b = 0;
        MatR M(4, 8), t(4);
        for (NumN i = 1; i <= n; ++i) {
            NumR x1, y1, x2, y2;
            x1 = data(i, 1);
            y1 = data(i, 2);
            x2 = data(i + 1, 1);
            y2 = data(i + 1, 2);
            M =
                    1, x1, x1 * x1, x1 * x1 * x1, 0, 0, 0, 0,
                    1, x2, x2 * x2, x2 * x2 * x2, 0, 0, 0, 0,
                    0, 1, 2 * x2, 3 * x2 * x2, 0, -1, -2 * x2, -3 * x2 * x2,
                    0, 0, 2, 6 * x2, 0, 0, -2, -6 * x2;
            t = y1, y2, 0, 0;
            if (i < n) {
                math21_op_matrix_sub_region_set(M, A, 0, 0, (i - 1) * 4, (i - 1) * 4, 0, 0);
                math21_op_vector_sub_region_set(t, b, 0, (i - 1) * 4, 0);
            } else {
                NumR x3, x4;
                x3 = data(1, 1);
                x4 = data(i + 1, 1);

                M.setSize(4, 4);
                M =
                        1, x1, x1 * x1, x1 * x1 * x1,
                        1, x2, x2 * x2, x2 * x2 * x2,
                        0, 1, 2 * x4, 3 * x4 * x4,
                        0, 1, 2 * x3, 3 * x3 * x3;
                math21_op_matrix_sub_region_set(M, A, 0, 0, (i - 1) * 4, (i - 1) * 4, 3, 0);
                math21_op_matrix_sub_region_set(M, A, 3, 0, i * 4 - 1, 0, 0, 0);
                t = y1, y2, k2, k1;
                math21_op_vector_sub_region_set(t, b, 0, (i - 1) * 4, 0);
            }
        }
        math21_equation_solve_linear_equation_simple(A, b, x);
        VecN d(2);
        d = x.size() / 4, 4;
        x.reshape(d);
    }

    void math21_fit_parametric_cubic_spline_Catmull_Rom(const MatR &data, TenR &At) {
        NumN n = data.nrows();
        MATH21_ASSERT(n >= 4);
        n -= 2;
        At.setSize(n - 1, 4, data.ncols());
        MatR B(4, 4);
        NumR s = 0.5;
        B =
                -s, 2 - s, s - 2, s,
                2 * s, s - 3, 3 - 2 * s, -s,
                -s, 0, s, 0,
                0, 1, 0, 0;
        MatR C;
        MatR Ait;
        for (NumN i = 1; i <= n - 1; ++i) {
            math21_operator_share_tensor_row_i(At, i, Ait);
            math21_operator_share_tensor_rows(data, i - 1, 4, C);
            math21_op_mat_mul(B, C, Ait);
        }
    }

    void math21_fit_parametric_cubic_spline(const MatR &data, TenR &At, NumN type) {
        math21_fit_parametric_cubic_spline_Catmull_Rom(data, At);
    }

}