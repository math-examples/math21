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

#pragma once

#include "inner.h"

namespace math21 {
    namespace ad {
        // todo: merge with op_num_binary
        // changed from op_num_unary
        struct op_ele_unary : public Function {
        private:
        public:
            op_ele_unary();

            virtual ~op_ele_unary();

            virtual VarAd df_vjp(const SetVar &X, VarAd x, VarAd y, VariableMap &data) const;

            virtual VarAd cr_vjp_inner(const SetVar &X, VarAd x, VarAd y, VarAd dy, VariableMap &data) const override;

            virtual VarAd evaluate(const SetVar &X, VariableMap &data) override;

            void cr(const SetVar &X, VarAd x, VarAd y, VarAd dy, SetVar &output, VariableMap &data) const override;

            void backward(const SetVar &X, VarAd x, VarAd y, VarAd dy, SetVar &output, VariableMap &data) const override;

            void f(const SetVar &X, SetVar &output, VariableMap &data) override;

            void fv(const SetVar &X, const SetVar &Y, VariableMap &data) const override;

//            virtual NumR evaluate_at_num(NumR x) const = 0;

            virtual void evaluate_at_tensor(const VecR &x, VecR &y) const = 0;
        };
    }
}