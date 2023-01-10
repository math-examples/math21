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
#include "../functions_03/op_push.h"
#include "op_sum.h"
#include "op_multiply.h"

namespace math21 {
    namespace ad {
        VarAd op_multiply::cr_vjp_inner(const SetVar &X, VarAd x, VarAd y, VarAd dy, VariableMap &data) const {
            math21_tool_assert(X.size() >= 2);
            math21_tool_assert(X.contains(x));
            SetVar X_no_x;
            X.copyToNox(X_no_x, x);

            SetVar X_mul;
            X_mul.add(dy);
            X_mul.add(X_no_x);

            op_multiply multiply0;
            Function &multiply = multiply0;
            return multiply.evaluate(X_mul, data);
        }

        // todo: remove broadcast and use math21_op_mul, unbroadcast in vjp.
        // todo: add ad_is_containing_constant_num_0
        VarAd op_multiply::evaluate(const SetVar &X0, VariableMap &data) {
            math21_tool_assert(X0.size() >= 2);
            SetVar X;
            broadcast_num_to_vec(X0, X, data);
            VarAd y = data.createV(math21_string_concatenate(getName(), "(x)").c_str());
            variable_set_device_type_using_variable(X(1), y, data);
            data.at(y).setf(this);
            data.at(y).setX(X);
            for (NumN i = 1; i <= X.size(); ++i) {
                data.at(X(i)).addy(y);
            }
            SetVar Y;
            Y.add(y);
            fv(X, Y, data);
            return y;
        }

        void op_multiply::fv(const SetVar &X, const SetVar &Y, VariableMap &data) const {
            VarAd y = Y(1);
            variable_setSize_to_same_vspace_using_variable(X(1), y, data);
            auto &y_vec = data.at(y).getValue();
            y_vec = 1;
            for (NumN i = 1; i <= X.size(); ++i) {
                math21_op_container_mulToB(data(X(i)).getValue(), y_vec);
            }
        }

        VarAd op_kx::cr_vjp_inner(const SetVar &X, VarAd x, VarAd y, VarAd dy, VariableMap &data) const {
            math21_tool_assert(X.size() == 2);
            math21_tool_assert(X.contains(x));
            if (X(1) == x) {
                op_multiply _op_multiply;
                Function &f_op_multiply = _op_multiply;
                VarAd t = f_op_multiply.evaluate(dy, X(2), data);
                op_sum _sum;
                Function &f_sum = _sum;
                VarAd dk = f_sum.evaluate(t, data);
                if (variable_get_device_type(X(1), data) != variable_get_device_type(X(2), data)) {
                    if (variable_is_cpu(X(1), data)) {
                        op_pull _op_pull;
                        Function &f_op_pull = _op_pull;
                        return f_op_pull.evaluate(dk, data);
                    } else {
                        op_push _op_push;
                        Function &f_op_push = _op_push;
                        return f_op_push.evaluate(dk, data);
                    }
                } else {
                    return dk;
                }
            } else {
                op_kx _op_kx;
                Function &f_op_kx = _op_kx;
                return f_op_kx.evaluate(X(1), dy, data);
            }
        }

        // Device type of y is based on that of x.
        // X = {k, x}, k can be on cpu or not.
        VarAd op_kx::evaluate(const SetVar &X, VariableMap &data) {
            math21_tool_assert(X.size() == 2);
            VarAd y = data.createV(math21_string_concatenate(getName(), "(x)").c_str());
            variable_set_device_type_using_variable(X(2), y, data);
            data.at(y).setf(this);
            data.at(y).setX(X);
            for (NumN i = 1; i <= X.size(); ++i) {
                data.at(X(i)).addy(y);
            }
            SetVar Y;
            Y.add(y);
            fv(X, Y, data);
            return y;
        }

        void op_kx::fv(const SetVar &X, const SetVar &Y, VariableMap &data) const {
            auto &_k = data(X(1)).getValue();
            MATH21_ASSERT(_k.isScalarInMath());
            VecR k;
            k = _k;
            auto &x = data.at(X(2)).getValue();
            auto &y = data.at(Y(1)).getValue();
            math21_op_mul(k(1), x, y);
        }
    }
}