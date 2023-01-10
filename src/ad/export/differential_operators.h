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
        void ad_grad_clear();

        PointAd ad_grad(const PointAd &x, const PointAd &y);

        PointAd grad(const PointAd &x, const PointAd &y);

        PointAd egrad(const PointAd &x, const PointAd &y);

        PointAd ad_jacobian_one_graph(const PointAd &x, const PointAd &y);

        PointAd ad_jacobian(const PointAd &x, const PointAd &y);

        PointAd ad_hessian(const PointAd &x, const PointAd &y);

        PointAd ad_hessian_vector_product(const PointAd &x, const PointAd &y, const PointAd &vector);

        void ad_fv(const PointAd &y);

        void ad_fv(const PointAd &x, const PointAd &y);
    }
}