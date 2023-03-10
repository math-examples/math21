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
        struct op_sum : public Function {
        private:
            NumB isKeepingDims;
        public:
            explicit op_sum(NumB isKeepingDims = 0) : isKeepingDims(isKeepingDims) {}

            ~op_sum() override = default;

            VarAd cr_vjp_inner(const SetVar &X, VarAd x, VarAd y, VarAd dy, VariableMap &data) const override;

            void fv(const SetVar &X, const SetVar &Y, VariableMap &data) const override;

            void compute(const SetVar &X, const SetVar &Y, VariableMap &data, Derivative &derivative) override {
                math21_tool_assert(0);
            }

            Function *clone() const override {
                auto *f = new op_sum(isKeepingDims);
                return f;
            }

            const char *getName() const override {
                return "op_sum";
            }
        };
    }
}