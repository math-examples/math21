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

#include "num_input.h"
#include "inner_cc.h"

namespace math21 {
    namespace ad {
        op_num_input::op_num_input() {}

        op_num_input::~op_num_input() {}

        void
        op_num_input::df(const SetVar &X, VarAd x, VarAd y, SetVar &output, VariableMap &data) const { MATH21_ASSERT(0)}

        void op_num_input::cr(const SetVar &X, VarAd x, VarAd y, VarAd dy, SetVar &output,
                               VariableMap &data) const { MATH21_ASSERT(0) }

        void op_num_input::f(const SetVar &X, SetVar &output, VariableMap &data) {MATH21_ASSERT(0)}

        void op_num_input::fv(const SetVar &X, const SetVar &Y, VariableMap &data) const {}

        void op_num_input::compute(const SetVar &X, const SetVar &Y, VariableMap &data, Derivative &derivative) {}

        void op_num_input::setSize(const SetVar &X, const SetVar &Y, VariableMap &data) const {}

        // no clone, just use the global one.
        Function *op_num_input::clone() const {
            Function *f = new op_num_input();
            return f;
        }

        const char *op_num_input::getName() const {
            return "op_num_input";
        }
    }
}