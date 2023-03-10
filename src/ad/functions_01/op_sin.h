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
        struct op_sin : public Function {
        public:
            op_sin() {}

            ~op_sin() override {}

            VarAd cr_vjp_inner(const SetVar &X, VarAd x, VarAd y, VarAd dy, VariableMap &data) const override;

            VarAd evaluate(const SetVar &X, VariableMap &data) override;

            void fv(const SetVar &X, const SetVar &Y, VariableMap &data) const override;

            Function *clone() const override {
                auto *f = new op_sin();
                return f;
            }

            const char *getName() const override {
                return "op_sin";
            }
        };

        struct op_cos : public Function {
        public:
            op_cos() {}

            ~op_cos() override {}

            VarAd cr_vjp_inner(const SetVar &X, VarAd x, VarAd y, VarAd dy, VariableMap &data) const override;

            VarAd evaluate(const SetVar &X, VariableMap &data) override;

            void fv(const SetVar &X, const SetVar &Y, VariableMap &data) const override;

            Function *clone() const override {
                auto *f = new op_cos();
                return f;
            }

            const char *getName() const override {
                return "op_cos";
            }
        };

    }
}