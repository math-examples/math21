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
        PointAd ad_tanh(const PointAd &x);

        PointAd tanh(const PointAd &x);

        PointAd ad_sigmoid_from_tanh(const PointAd &x);

        PointAd ad_sigmoid_from_logistic(const PointAd &x);

        PointAd ad_logsumexp(const PointAd &x, const VecN &axes = VecN(), NumB isKeepingDims = 0);

        PointAd ad_mvn_logpdf(const PointAd &x, const PointAd &mean, const PointAd &covariance);

        PointAd ad_vec_share_sub_from_to(const PointAd &x, NumZ from, NumZ to);

        PointAd ad_vec_share_sub_offset(const PointAd &x, NumN offset, NumN n);

        PointAd ad_get_shape(const PointAd &x);

        PointAd ad_get_size(const PointAd &x);

        PointAd ad_get_shrink_shape_keeping_dim(const PointAd &x, const PointAd &axes);

        PointAd ad_share_reshape(const PointAd &x, const PointAd &d);

        PointAd ad_row_pack(const PointAd &p1, const PointAd &p2, const PointAd &p3);

        PointAd ad_row_pack(const Set_ <PointAd > &x);

        void ad_row_unpack(const PointAd &x, Set_ <PointAd > &y);

        PointAd ad_log_pr(const PointAd &x);

        PointAd ad_gmm_log_likelihood(const PointAd &params, const PointAd &data,
                                       NumN n_component, NumN n_feature, NumB isECorder = 1);

        PointAd ad_repeat(const PointAd &x, const PointAd &repeats, PointAd axis);

        PointAd ad_undo_repeat_sum(const PointAd &x, const PointAd &repeats, PointAd axis);


        PointAd ad_concatenate(const PointAd &x1, const PointAd &x2, PointAd axis);

        PointAd ad_concatenate(const PointAd &x1, const PointAd &x2, const PointAd &x3, PointAd axis);

        PointAd ad_concatenate(const Seqce <PointAd > &xs, PointAd axis);

        // x, y can be empty tensor
        // can get sub-tensor y from tensor x
        PointAd ad_axis_i_sub_get(const PointAd &x, const PointAd &offset, const PointAd &di, const PointAd &axis);

        PointAd ad_axis_i_sub_set(const PointAd &x, const PointAd &offset, const PointAd &d_y, const PointAd &axis);

        PointAd ad_axis_swap(const PointAd &x, const PointAd &pos, const PointAd &pos2);
    }
}