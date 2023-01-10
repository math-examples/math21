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

#include "../functions_01/files.h"
#include "../functions_02/files.h"
#include "../functions_03/files.h"
#include "../differential.h"
#include "01.h"
#include "differential_operators.h"

namespace math21 {
    namespace ad {
        // compute derivative in define-by-run mode
        PointAd ad_grad_test(const PointAd &x, const PointAd &y) {
            return ad_grad(x, y);

//            derivative.setDebugLevel(1);

//            VarAd dydx = derivative.cd(x.id, y.id);
//            derivative.compute(dydx);

//            VarAd dydx = derivative.backward(x.id, y.id);

            math21_tool_assert(0);
        }

        void ad_grad_clear() {

        }

        PointAd ad_grad_jvp(const PointAd &x, const PointAd &y) {
            auto &derivative = ad_global_get_derivative();
            VarAd dydx = derivative.grad_jvp(x.getVarAd(), y.getVarAd());
            return PointAd(dydx);
        }

        // return a vector with same shape as x
        PointAd ad_grad_vjp(const PointAd &x, const PointAd &y) {
            auto &derivative = ad_global_get_derivative();
            VarAd dydx = derivative.grad_vjp(x.getVarAd(), y.getVarAd());
            return PointAd(dydx);
        }

        // f' = gradient of f, here f: R^n -> R
        PointAd ad_grad(const PointAd &x, const PointAd &y) {
            return ad_grad_vjp(x, y);
        }

        PointAd grad(const PointAd &x, const PointAd &y) {
            return ad_grad(x, y);
        }

        // elementwise grad
        // Returns the sum of each column of the Jacobian of
        // `y = f(x)`, in one pass. If the Jacobian is diagonal, then this is the diagonal
        // of the Jacobian.
        PointAd egrad(const PointAd &x, const PointAd &y) {
            auto y1 = ad_sum(y);
            return ad_grad(x, y1);
        }

        // graph is fully kept, so fv can be used.
        // dx isn't constant, so can compute higher order derivatives?
        PointAd ad_jacobian_one_graph(const PointAd &x, const PointAd &y) {
            auto &data = ad_global_get_data();
            if (x.isEmpty() || y.isEmpty()) {
                return PointAd();
            }
            const auto &x_vec = ad_get_variable(x).getValue();
            const auto &y_vec = ad_get_variable(y).getValue();
            NumN y_size = y_vec.size();
            op_row_pack pack0;
            Function &pack = pack0;
            SetVar X;
            for (NumN i = 1; i <= y_size; ++i) {
                auto dyidx = ad_grad(x, at(y, i));
                if (dyidx.isEmpty()) {
                    return dyidx;
                }
                X.add(dyidx.getVarAd());
            }
            VarAd dx = pack.evaluate(X, data);

            op_get_shape _get_shape;
            Function &f_get_shape = _get_shape;
            VarAd d_y = f_get_shape.evaluate(y.getVarAd(), data);
            VarAd d_x = f_get_shape.evaluate(x.getVarAd(), data);
            op_merge_vectors _op_merge_vectors;
            Function &f_op_merge_vectors = _op_merge_vectors;
            VarAd d = f_op_merge_vectors.evaluate(d_y, d_x, data);
            op_share_reshape _op_share_reshape;
            Function &f_op_share_reshape = _op_share_reshape;
            dx = f_op_share_reshape.evaluate(dx, d, data);
            return PointAd(dx);
        }

        // graph is not fully kept, so fv can't be used.
        // dx is constant, so can't compute higher order derivatives.
        PointAd ad_jacobian_record_again(const PointAd &x, const PointAd &y) {
            auto &derivative = ad_global_get_derivative();
            if (x.isEmpty() || y.isEmpty()) {
                return PointAd();
            }
            const auto &x_vec = ad_get_variable(x).getValue();
            const auto &y_vec = ad_get_variable(y).getValue();
            NumN x_size = x_vec.size();
            NumN y_size = y_vec.size();
            MatR dx_mat;
            dx_mat.setSize(y_size, x_size);
            for (NumN i = 1; i <= y_size; ++i) {
                auto dyidx = ad_grad(x, at(y, i));
                if (dyidx.isEmpty()) {
                    return dyidx;
                }
                const auto &dyidx_vec = ad_get_variable(dyidx).getValue();
                math21_tool_assert(dyidx_vec.size() == x_size);
                math21_operator_matrix_row_set_by_vec(dx_mat, i, dyidx_vec);
                derivative.removeLastRecord();
            }
            auto dx = ad_create_point_const("dx");
            auto &dx_vec = ad_get_variable(dx).getValue();
            dx_vec.swap(dx_mat);
            VecN d;
            math21_operator_merge(y_vec.shape(), x_vec.shape(), d);
            Function::variable_reshape_to_same_vspace_using_shape(d, dx.getVarAd(), ad_get_data());
            return dx;
        }

        // ad_jacobian_one_graph vs ad_jacobian_record_again
        PointAd ad_jacobian(const PointAd &x, const PointAd &y) {
//            return ad_jacobian_one_graph(x, y);
            return ad_jacobian_record_again(x, y);
        }

        // f'' = Jacobian of gradient of f, here f: R^n -> R
        PointAd ad_hessian(const PointAd &x, const PointAd &y) {
            return ad_jacobian(x, ad_grad(x, y));
        }

        // return a vector H*v with same shape as x
        // d(g*v)/dx = v.t * H = (H * v).t, g = f', H is Hessian matrix.
        // Hessian vector product
        // f'' = Jacobian of gradient of f, here f: R^n -> R
        PointAd ad_hessian_vector_product(const PointAd &x, const PointAd &y, const PointAd &vector) {
            auto g = ad_grad(x, y);
            auto vector_dot_grad = ad_inner_product(g, vector);
            return ad_grad(x, vector_dot_grad);
        }

        // don't call this if tensor shape changed in the graph
        void ad_fv(const PointAd &y) {
            if (ad_is_const_variable(y)) {
                m21warn("compute value of constant variable!");
                return;
            }
            auto &derivative = ad_global_get_derivative();
            SetVar X, Y;
            Y.add(y.getVarAd());
            derivative.fvs(X, Y);
        }

        void ad_fv(const PointAd &x, const PointAd &y) {
            auto &derivative = ad_global_get_derivative();
            SetVar X, Y;
            X.add(x.getVarAd());
            Y.add(y.getVarAd());
            derivative.fvs(X, Y);
        }
    }
}