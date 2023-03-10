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
#include "op_inner_product.h"

namespace math21 {
    namespace ad {
        op_inner_product::op_inner_product() {
        }

        op_inner_product::~op_inner_product() {
        }

        VarAd op_inner_product::cr_vjp_inner(const SetVar &X, VarAd x, VarAd y, VarAd dy, VariableMap &data) const {
            math21_tool_assert(X.size() == 2);
            math21_tool_assert(X.contains(x));
            VarAd x1 = X(1);
            VarAd x2 = X(2);
            op_multiply multiply0;
            Function &multiply = multiply0;
            if (x == x1) {
                return multiply.evaluate(dy, x2, data);
            } else {
                return multiply.evaluate(dy, x1, data);
            }
        }

        VarAd op_inner_product::evaluate(const SetVar &X0, VariableMap &data) {
            math21_tool_assert(X0.size() == 2);
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

        void op_inner_product::fv(const SetVar &X, const SetVar &Y, VariableMap &data) const {
            VarAd y = Y(1);
            const auto &x1_vec = data.at(X(1)).getValue();
            const auto &x2_vec = data.at(X(2)).getValue();
            auto &y_vec = data.at(y).getValue();
            y_vec.setSize(1);
//            y_vec = math21_operator_container_InnerProduct(1, x1_vec, x2_vec);
            math21_op_inner_product(x1_vec, x2_vec, y_vec);
        }

        void op_inner_product::df(const SetVar &X, VarAd x, VarAd y, SetVar &output, VariableMap &data) const {
            math21_tool_assert(0);
        }

        void op_inner_product::cr(const SetVar &X, VarAd x, VarAd y, VarAd dy, SetVar &output, VariableMap &data) const {
            math21_tool_assert(0);
        }

        void op_inner_product::f(const SetVar &X, SetVar &output, VariableMap &data) {
            math21_tool_assert(0);
        }

        void op_inner_product::compute(const SetVar &X, const SetVar &Y, VariableMap &data, Derivative &derivative) {
            math21_tool_assert(0);
        }

        void op_inner_product::setSize(const SetVar &X, const SetVar &Y, VariableMap &data) const {
            math21_tool_assert(0);
        }

        Function *op_inner_product::clone() const {
            Function *f = new op_inner_product();
            return f;
        }

        const char *op_inner_product::getName() const {
            return "op_inner_product";
        }
    }
}