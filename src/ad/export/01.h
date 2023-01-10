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

        // natural logarithm function
        PointAd ad_log(const PointAd &x);

        PointAd ad_log(const PointAd &x, NumR base);

        PointAd ad_sin(const PointAd &p);

        PointAd ad_cos(const PointAd &p);

        PointAd ad_sum(const PointAd &p);

        PointAd ad_sum(const PointAd &p, const PointAd &axes, NumB isKeepingDims = 0);

        PointAd ad_tensor_broadcast(const PointAd &p, const PointAd &d);

        PointAd ad_add(const PointAd &p1, const PointAd &p2);

        PointAd ad_add(const PointAd &p1, const PointAd &p2, const PointAd &p3);

        PointAd ad_negate(const PointAd &p);

        PointAd ad_subtract(const PointAd &p1, const PointAd &p2);

        PointAd ad_mul(const PointAd &p1, const PointAd &p2);

        PointAd ad_mul(const PointAd &p1, const PointAd &p2, const PointAd &p3);

        PointAd ad_divide(const PointAd &p1, const PointAd &p2);

        PointAd ad_kx(const PointAd &k, const PointAd &x);

        PointAd operator+(const PointAd &p1, const PointAd &p2);

        PointAd operator-(const PointAd &p1);

        PointAd operator-(const PointAd &p1, const PointAd &p2);

        PointAd operator*(const PointAd &p1, const PointAd &p2);

        PointAd operator/(const PointAd &p1, const PointAd &p2);

        PointAd ad_power(const PointAd &x, NumR k, NumR p);

        // natural exponential function
        PointAd ad_exp(const PointAd &x);

        PointAd exp(const PointAd &x);

        PointAd ad_exp(const PointAd &x, NumR base);

        PointAd ad_mat_trans(const PointAd &x);

        PointAd ad_mat_mul(const PointAd &p1, const PointAd &p2);

        PointAd ad_inner_product(const PointAd &p1, const PointAd &p2);

        PointAd dot(const PointAd &p1, const PointAd &p2);

        PointAd ad_at(const PointAd &p, NumN index);

        PointAd at(const PointAd &p, NumN index);

        PointAd ad_push(const PointAd &p);

        PointAd ad_pull(const PointAd &p);

        PointAd ad_create_using_shape(const PointAd &value, const PointAd &d);

        PointAd ad_mean(const PointAd &p);

        PointAd ad_mean(const PointAd &p, const PointAd &axes, const PointAd &isKeepingDims);
    }
}