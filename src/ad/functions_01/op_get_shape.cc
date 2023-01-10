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
#include "op_get_shape.h"

namespace math21 {
    namespace ad {

        VarAd op_get_shape::cr_vjp_inner(const SetVar &X, VarAd x, VarAd y, VarAd dy, VariableMap &data) const {
            math21_tool_assert(0);
            return VarAd(0);
        }

        VarAd op_get_shape::evaluate(const SetVar &X, VariableMap &data) {
            math21_tool_assert(X.size() == 1);
            VarAd y = data.createV(math21_string_concatenate(getName(), "(x)").c_str());
            data.at(y).setf(this);
            data.at(y).setX(X);
            data.at(y).setType(variable_type_zero_derivative);
            for (NumN i = 1; i <= X.size(); ++i) {
                data.at(X(i)).addy(y);
            }
            SetVar Y;
            Y.add(y);
            fv(X, Y, data);
            return y;
        }

        void op_get_shape::fv(const SetVar &X, const SetVar &Y, VariableMap &data) const {
            VarAd y = Y(1);
            const auto &x_value = data.at(X(1)).getValue();
            VecN d;
            x_value.shape(d);
            auto &y_value = data.at(y).getValue();
            y_value = d;
        }

        VarAd op_get_size::cr_vjp_inner(const SetVar &X, VarAd x, VarAd y, VarAd dy, VariableMap &data) const {
            math21_tool_assert(0);
            return VarAd(0);
        }

        VarAd op_get_size::evaluate(const SetVar &X, VariableMap &data) {
            math21_tool_assert(X.size() == 1);
            VarAd y = data.createV(math21_string_concatenate(getName(), "(x)").c_str());
            data.at(y).setf(this);
            data.at(y).setX(X);
            data.at(y).setType(variable_type_zero_derivative);
            for (NumN i = 1; i <= X.size(); ++i) {
                data.at(X(i)).addy(y);
            }
            SetVar Y;
            Y.add(y);
            fv(X, Y, data);
            return y;
        }

        void op_get_size::fv(const SetVar &X, const SetVar &Y, VariableMap &data) const {
            VarAd y = Y(1);
            const auto &x_value = data.at(X(1)).getValue();
            auto &y_value = data.at(y).getValue();
            y_value.setSize(1);
            y_value = x_value.size();
        }

        VarAd
        op_get_shrink_shape_keeping_dim::cr_vjp_inner(const SetVar &X, VarAd x, VarAd y, VarAd dy, VariableMap &data) const {
            math21_tool_assert(0);
            return VarAd(0);
        }

        VarAd op_get_shrink_shape_keeping_dim::evaluate(const SetVar &X, VariableMap &data) {
            math21_tool_assert(X.size() == 2);
            VarAd y = data.createV(math21_string_concatenate(getName(), "(x)").c_str());
            data.at(y).setf(this);
            data.at(y).setX(X);
            data.at(y).setType(variable_type_zero_derivative);
            for (NumN i = 1; i <= X.size(); ++i) {
                data.at(X(i)).addy(y);
            }
            SetVar Y;
            Y.add(y);
            fv(X, Y, data);
            return y;
        }

        // X = {x, axes}, y = shape
        void op_get_shrink_shape_keeping_dim::fv(const SetVar &X, const SetVar &Y, VariableMap &data) const {
            auto &y_value = data.at(Y(1)).getValue();
            VecN axes;
            axes = data.at(X(2)).getValue();
            VecN d_y;
            math21_operator_tensor_shrink_shape_using_axes_with_dim_kept(
                    data(X(1)).getValue(), axes, d_y);
            y_value = d_y;
        }
    }
}