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
#include "op_list_to_i_var.h"

namespace math21 {
    namespace ad {
        op_list_to_i_var::op_list_to_i_var(NumN pos) : _pos(pos) {
        }

        op_list_to_i_var::~op_list_to_i_var() {
        }

        VarAd op_list_to_i_var::cr_vjp_inner(const SetVar &X, VarAd x, VarAd y, VarAd dy, VariableMap &data) const {
            math21_tool_assert(0 && "not implement");
            return VarAd(0);
        }

        // share data
        VarAd op_list_to_i_var::evaluate(const SetVar &X, VariableMap &data) {
            math21_tool_assert(X.size() == 1);
            VarAd y = data.createV(math21_string_concatenate(getName(), "(x)").c_str());
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

        void op_list_to_i_var::fv(const SetVar &X, const SetVar &Y, VariableMap &data) const {
            VarAd y = Y(1);
            const auto &list_value = data(X(1)).getValue();
            math21_tool_assert(0 && "debug, NumR -> VarAd");
            VarAd x = (VarAd)list_value(_pos);
            const auto &x_value = data(x).getValue();
            auto &y_value = data.at(y).getValue();
            math21_operator_share_copy(x_value, y_value);
        }

        Function *op_list_to_i_var::clone() const {
            Function *f = new op_list_to_i_var(_pos);
            return f;
        }

        const char *op_list_to_i_var::getName() const {
            return "op_list_to_i_var";
        }
    }
}