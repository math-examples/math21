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
        struct op_vec_share_sub : public Function {
        private:
            NumZ from;
            NumZ to;
        public:
            op_vec_share_sub(NumZ from, NumZ to);

            virtual ~op_vec_share_sub();

            VarAd cr_vjp_inner(const SetVar &X, VarAd x, VarAd y, VarAd dy, VariableMap &data) const override;

            VarAd evaluate(const SetVar &X, VariableMap &data) override;

            void df(const SetVar &X, VarAd x, VarAd y, SetVar &output, VariableMap &data) const;

            void cr(const SetVar &X, VarAd x, VarAd y, VarAd dy, SetVar &output, VariableMap &data) const override;

            void f(const SetVar &X, SetVar &output, VariableMap &data) override;

            void fv(const SetVar &X, const SetVar &Y, VariableMap &data) const override;

            void compute(const SetVar &X, const SetVar &Y, VariableMap &data, Derivative &derivative) override;

            void setSize(const SetVar &X, const SetVar &Y, VariableMap &data) const override;

            Function *clone() const override;

            const char *getName() const override;

        };
    }

    namespace ad {
        struct op_vec_d_sub : public Function {
        private:
            NumZ from;
            NumZ to;
            NumN n; // not compatible with resize, so bugs may arise from here.
        public:
            op_vec_d_sub(NumN n, NumZ from, NumZ to);

            virtual ~op_vec_d_sub();

            VarAd cr_vjp_inner(const SetVar &X, VarAd x, VarAd y, VarAd dy, VariableMap &data) const override;

            VarAd evaluate(const SetVar &X, VariableMap &data) override;

            void df(const SetVar &X, VarAd x, VarAd y, SetVar &output, VariableMap &data) const;

            void cr(const SetVar &X, VarAd x, VarAd y, VarAd dy, SetVar &output, VariableMap &data) const override;

            void f(const SetVar &X, SetVar &output, VariableMap &data) override;

            void fv(const SetVar &X, const SetVar &Y, VariableMap &data) const override;

            void compute(const SetVar &X, const SetVar &Y, VariableMap &data, Derivative &derivative) override;

            void setSize(const SetVar &X, const SetVar &Y, VariableMap &data) const override;

            Function *clone() const override;

            const char *getName() const override;

        };
    }
}