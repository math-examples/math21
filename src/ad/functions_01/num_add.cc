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

#include "num_multiply.h"
#include "num_constant.h"
#include "num_add.h"
#include "inner_cc.h"

namespace math21 {
    namespace ad {
        op_num_add::op_num_add() {
        }

        op_num_add::~op_num_add() {
        }

        void op_num_add::df(const SetVar &X, VarAd x, VarAd y, SetVar &output, VariableMap &data) const {
            if (X.contains(x)) {
                op_num_constant numConstant;
                numConstant.f(X, output, data);
                data.at(output(1)).getValue() = 1;
            } else {
                op_num_constant numConstant;
                numConstant.f(X, output, data);
                return;
            }
        }

        void op_num_add::cr(const SetVar &X, VarAd x, VarAd y, VarAd dy, SetVar &output, VariableMap &data) const {
            if (X.contains(x)) {
                op_num_constant numConstant;
                numConstant.f(X, output, data);
                data.at(output(1)).getValue() = 1;
            } else {
                op_num_constant numConstant;
                numConstant.f(X, output, data);
                return;
            }

            SetVar input;
            input.add(dy);
            input.add(output);
            op_num_multiply multiply;
            multiply.f(input, output, data);
            data.at(output(1)).setName("dx = dy * d(op_num_add(x))");
        }

        void op_num_add::backward(const SetVar &X, VarAd x, VarAd y, VarAd dy, SetVar &output, VariableMap &data) const {
            if (X.contains(x)) {
                op_num_constant numConstant;
                numConstant.forward(X, output, data);
                data.at(output(1)).getValue() = 1;
            } else {
                op_num_constant numConstant;
                numConstant.forward(X, output, data);
                return;
            }

            SetVar input;
            input.add(dy);
            input.add(output);
            op_num_multiply multiply;
            multiply.forward(input, output, data);
            data.at(output(1)).setName("dx = dy * d(op_num_add(x))");
        }

        void op_num_add::f(const SetVar &X, SetVar &output, VariableMap &data) {
            VarAd y = data.createV("op_num_add(x)");
            data.at(y).setf(this);
            data.at(y).setX(X);
            for (NumN i = 1; i <= X.size(); ++i) {
                data.at(X(i)).addy(y);
            }
            output.clear();
            output.add(y);
        }

        void op_num_add::fv(const SetVar &X, const SetVar &Y, VariableMap &data) const {
            if (X.isEmpty()) {
                return;
            }
            VarAd y = Y(1);
            if(data.at(y).isComputed()){
                return;
            }
            data.at(y).getValue().setSize(1);
            NumR sum = 0;
            for (NumN i = 1; i <= X.size(); ++i) {
                sum = sum + data(X(i)).getValue()(1);
            }
            data.at(y).getValue() = sum;
            data.at(y).setComputed(1);
        }

        void op_num_add::setSize(const SetVar &X, const SetVar &Y, VariableMap &data) const {
        }

        Function *op_num_add::clone() const {
            Function *f = new op_num_add();
            return f;
        }

        const char *op_num_add::getName() const {
            return "op_num_add";
        }
    }
}