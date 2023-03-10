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

#include "inner_h.h"

namespace math21 {
    class polynomial : public Functional {
    private:
        VecR A;
        VecR x0;
    public:
        polynomial();

        virtual ~polynomial() {}

        NumR valueAt(const VecR &x) override;

        NumN getXDim() override;

        const VecR &getX0();

        const VecR &derivativeValueAt(const VecR &x) override;
    };

    void math21_function_cubic_spline(const MatR &K, const VecR &p, const VecR &x, VecR &y);

    void math21_function_derivative_ith_order_parametric_polynomial(const MatR &At, MatR &Bt, NumN i);

    void math21_function_parametric_spline_evaluate(const TenR &At, const VecR &t, MatR &xt);

    void math21_function_parametric_cubic_spline_evaluate(const TenR &At, const VecR &t, MatR &xt);

    void math21_fit_2d_cubic_curve(const MatR &data, VecR &a);

    void math21_fit_2d_cubic_spline_natural(const MatR &data, NumR k1, NumR k2, MatR &x);

    void math21_fit_parametric_cubic_spline(const MatR &data, TenR &At, NumN type=0);
}