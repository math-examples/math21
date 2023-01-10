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
#include "op_row_unpack_i.h"

namespace math21 {
    namespace ad {
        op_row_unpack_i::op_row_unpack_i(NumN pos) : _pos(pos) {
        }

        op_row_unpack_i::~op_row_unpack_i() {
        }

        // can use unpack
        VarAd op_row_unpack_i::cr_vjp_inner(const SetVar &X, VarAd x, VarAd y, VarAd dy, VariableMap &data) const {
            math21_tool_assert(0);
            return VarAd(0);
        }

        VarAd op_row_unpack_i::evaluate(const SetVar &X, VariableMap &data) {
            math21_tool_assert(X.size() == 1);
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

        void op_row_unpack_i::fv(const SetVar &X, const SetVar &Y, VariableMap &data) const {
            VarAd y = Y(1);
            VarAd x = X(1);
            const auto &x_value = data(x).getValue();
            auto &y_value = data.at(y).getValue();
            VecN d_x, d_y;
            x_value.shape(d_x);
            d_y.setSize(d_x.size() - 1);
            math21_operator_container_set_partially(d_x, d_y, 1);
            y_value.setSize(d_y);
            TenR x_copy;
            math21_operator_share_copy(x_value, x_copy);
            VecN d_x_mat(2);
            d_x_mat = x_value.dim(1), x_value.size() / x_value.dim(1);
            x_copy.reshape(d_x_mat);
//            math21_operator_matrix_row_get(x_copy, _pos, y_value);
            math21_op_matrix_get_row(y_value, x_copy, _pos);
        }

        void op_row_unpack_i::df(const SetVar &X, VarAd x, VarAd y, SetVar &output, VariableMap &data) const {
            math21_tool_assert(0);
        }

        void op_row_unpack_i::cr(const SetVar &X, VarAd x, VarAd y, VarAd dy, SetVar &output, VariableMap &data) const {
            math21_tool_assert(0);
        }

        void op_row_unpack_i::f(const SetVar &X, SetVar &output, VariableMap &data) {
            math21_tool_assert(0);
        }

        void op_row_unpack_i::compute(const SetVar &X, const SetVar &Y, VariableMap &data, Derivative &derivative) {
            math21_tool_assert(0);
        }

        void op_row_unpack_i::setSize(const SetVar &X, const SetVar &Y, VariableMap &data) const {
            math21_tool_assert(0);
        }

        Function *op_row_unpack_i::clone() const {
            Function *f = new op_row_unpack_i(_pos);
            return f;
        }

        const char *op_row_unpack_i::getName() const {
            return "op_row_unpack_i";
        }
    }
}