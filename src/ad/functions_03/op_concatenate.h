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
        struct op_concatenate : public Function {
        public:
            op_concatenate() = default;

            ~op_concatenate() override = default;

            VarAd cr_vjp_inner(const SetVar &X, VarAd x, VarAd y, VarAd dy, VariableMap &data) const override;

            VarAd evaluate(const SetVar &X, VariableMap &data) override;

            void fv(const SetVar &X, const SetVar &Y, VariableMap &data) const override;

            Function *clone() const override {
                auto *f = new op_concatenate();
                return f;
            }

            const char *getName() const override {
                return "op_concatenate";
            }
        };

        struct op_axis_i_sub_get : public Function {
        public:
            op_axis_i_sub_get() = default;

            ~op_axis_i_sub_get() override = default;

            VarAd cr_vjp_inner(const SetVar &X, VarAd x, VarAd y, VarAd dy, VariableMap &data) const override;

            VarAd evaluate(const SetVar &X, VariableMap &data) override;

            void fv(const SetVar &X, const SetVar &Y, VariableMap &data) const override;

            Function *clone() const override {
                auto *f = new op_axis_i_sub_get();
                return f;
            }

            const char *getName() const override {
                return "op_axis_i_sub_get";
            }
        };

        struct op_axis_i_sub_set : public Function {
        public:
            op_axis_i_sub_set() = default;

            ~op_axis_i_sub_set() override = default;

            VarAd cr_vjp_inner(const SetVar &X, VarAd x, VarAd y, VarAd dy, VariableMap &data) const override;

            VarAd evaluate(const SetVar &X, VariableMap &data) override;

            void fv(const SetVar &X, const SetVar &Y, VariableMap &data) const override;

            Function *clone() const override {
                auto *f = new op_axis_i_sub_set();
                return f;
            }

            const char *getName() const override {
                return "op_axis_i_sub_set";
            }
        };
    }
}