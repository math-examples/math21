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

#include "files.h"

namespace math21 {
    namespace ad {
        op_num_constant::op_num_constant() {
        }

        op_num_constant::~op_num_constant() {
        }

        void op_num_constant::df(const SetVar &X, VarAd x, VarAd y, SetVar &output, VariableMap &data) const {
            SetVar input;
            op_num_constant numConstant;
            numConstant.f(input, output, data);
        }

        void op_num_constant::cr(const SetVar &X, VarAd x, VarAd y, VarAd dy, SetVar &output, VariableMap &data) const {
            SetVar input;
            op_num_constant numConstant;
            numConstant.f(input, output, data);
        }

        void op_num_constant::f(VarAd &y, VariableMap &data) {
            SetVar X;
            SetVar Y;
            f(X, Y, data);
            y = Y(1);
        }

        // todo: if error, maybe use createV
        void op_num_constant::f(const SetVar &X, SetVar &output, VariableMap &data) {
            VarAd y = data.createC("op_num_constant(x)");
            data.setValue(y, 0);
            data.at(y).setf(this);
            output.clear();
            output.add(y);
        }

        void op_num_constant::fv(const SetVar &X, const SetVar &Y, VariableMap &data) const {
        }

        void op_num_constant::setSize(const SetVar &X, const SetVar &Y, VariableMap &data) const {
        }

        Function *op_num_constant::clone() const {
            Function *f = new op_num_constant();
            return f;
        }

        const char *op_num_constant::getName() const {
            return "op_num_constant";
        }
    }
}