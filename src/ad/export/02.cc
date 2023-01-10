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

#include "../functions_01/files.h"
#include "../functions_02/files.h"
#include "../functions_03/files.h"
#include "../differential.h"
#include "point.h"
#include "01.h"
#include "02.h"

namespace math21 {
    namespace ad {
        // f = tanh(x) = sinh(x)/cosh(x) = (exp(2x)-1)/(exp(2x)+1)
        // y = exp(-2.0 * x), f = (1.0 - y) / (1.0 + y)
        PointAd ad_tanh(const PointAd &x) {
            auto y = ad_exp(-2.0 * x);
            return (1.0 - y) / (1.0 + y);
        }

        PointAd tanh(const PointAd &x) {
            return ad_tanh(x);
        }

        // y = exp(-2.0 * x), f = 1.0 / (1.0 + y)
        // <=> y = f(x) = 0.5 * (ad_tanh(x) + 1.0)
        PointAd ad_sigmoid_from_tanh(const PointAd &x) {
            auto y = ad_exp(-2.0 * x);
            return 1.0 / (1.0 + y);
        }

        // y = exp(-x), f = 1.0 / (1.0 + y)
        PointAd ad_sigmoid_from_logistic(const PointAd &x) {
            auto y = ad_exp(-x);
            return 1.0 / (1.0 + y);
        }

        // axes: same as numpy, but from index 1
        PointAd ad_logsumexp(const PointAd &x, const VecN &axes, NumB isKeepingDims) {
            op_logsumexp logsumexp(axes, isKeepingDims);
            Function &function = logsumexp;
            VarAd y = function.evaluate(x.getVarAd(), ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_mvn_logpdf(const PointAd &x, const PointAd &mean, const PointAd &covariance) {
            op_mvn_logpdf logpdf;
            Function &function = logpdf;
            VarAd y = function.evaluate(x.getVarAd(), mean.getVarAd(), covariance.getVarAd(), ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_vec_share_sub_from_to(const PointAd &x, NumZ from, NumZ to) {
            op_vec_share_sub _op_vec_share_sub(from, to);
            Function &function = _op_vec_share_sub;
            VarAd y = function.evaluate(x.getVarAd(), ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_vec_share_sub_offset(const PointAd &x, NumN offset, NumN n) {
            return ad_vec_share_sub_from_to(x, offset + 1, offset + n);
        }

        PointAd ad_get_shape(const PointAd &x) {
            op_get_shape _op_get_shape;
            Function &function = _op_get_shape;
            VarAd y = function.evaluate(x.getVarAd(), ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_get_size(const PointAd &x) {
            op_get_size _;
            Function &f = _;
            VarAd y = f.evaluate(x.getVarAd(), ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_get_shrink_shape_keeping_dim(const PointAd &x, const PointAd &axes) {
            op_get_shrink_shape_keeping_dim _;
            Function &f = _;
            VarAd y = f.evaluate(x.getVarAd(), axes.getVarAd(), ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_share_reshape(const PointAd &x, const PointAd &d) {
            op_share_reshape _op_share_reshape;
            Function &function = _op_share_reshape;
            VarAd y = function.evaluate(x.getVarAd(), d.getVarAd(), ad_global_get_data());
            return PointAd(y);
        }

        // use SetVar to pack different vectors.
        PointAd ad_row_pack(const PointAd &p1, const PointAd &p2, const PointAd &p3) {
            Set_<PointAd > x;
            x.add(p1);
            x.add(p2);
            x.add(p3);
            return ad_row_pack(x);
        }

        // row pack tensors
        PointAd ad_row_pack(const Set_<PointAd > &x) {
            op_row_pack _op_row_pack;
            Function &function = _op_row_pack;
            SetVar s;
            for (NumN i = 1; i <= x.size(); ++i) {
                s.add(x(i).getVarAd());
            }
            VarAd y = function.evaluate(s, ad_global_get_data());
            return PointAd(y);
        }

        // row pack tensors
        PointAd ad_row_unpack_i(const PointAd &x, NumN pos) {
            op_row_unpack_i _op_row_unpack_i(pos);
            Function &function = _op_row_unpack_i;
            VarAd y = function.evaluate(x.getVarAd(), ad_global_get_data());
            return PointAd(y);
        }

        // row unpack tensor
        void ad_row_unpack(const PointAd &x, Set_<PointAd > &y) {
            y.clear();
            NumN n = ad_get_dim_i(x, 1);
            for (NumN i = 1; i <= n; ++i) {
                auto _ = ad_row_unpack_i(x, i);
                y.add(_);
            }
        }

        // x -> log P(x), e.x, preference of action -> probability of taking action (in gradient bandit algorithm)
        // P(xi) = e^xi/sum(e^xi)
        // log P(xi) = xi - logsum(e^xi)
        // log P(x) = x - logsumexp(x)
        PointAd ad_log_pr(const PointAd &x) {
            return x - ad_logsumexp(x);
        }

        // see math21_operator_gmm_unpack_params
        // params: {proportions, means, covs_sqrt} with shape [n_component, n_component*n_feature, n_component*n_feature*n_feature]
        // params: {proportions, covs_sqrt, means} with shape [n_component, n_component*n_feature*n_feature, n_component*n_feature]
        // data: n_data * n_feature, with n_feature = 2
        PointAd ad_gmm_log_likelihood(const PointAd &params, const PointAd &data,
                                       NumN n_component, NumN n_feature, NumB isECorder) {
            MATH21_ASSERT(ad_get_value(params).size() ==
                          n_component + n_component * n_feature + n_component * n_feature * n_feature,
                          ""
                                  << "ad_get_value(params).size() = " << ad_get_value(params).size()
                                  << "\nn_component = " << n_component
                                  << "\nn_feature = " << n_feature
            )
            auto proportions = ad_vec_share_sub_from_to(params, 1, n_component);
            auto log_prs = ad_log_pr(proportions);
            PointAd means, covs_sqrt;
            if (isECorder) {
                means = ad_vec_share_sub_from_to(params, n_component + 1, n_component + n_component * n_feature);
                covs_sqrt = ad_vec_share_sub_from_to(params, n_component + n_component * n_feature + 1, -1);
            } else { // CE order
                covs_sqrt = ad_vec_share_sub_from_to(params, n_component + 1,
                                                     n_component + n_component * n_feature * n_feature);
                means = ad_vec_share_sub_from_to(params, n_component + n_component * n_feature * n_feature + 1, -1);
            }
            VecN d_mean(1);
            d_mean = n_feature;
            auto p_d_mean = PointAd (d_mean);
            VecN d_cov_sqrt(2);
            d_cov_sqrt = n_feature, n_feature;
            auto p_d_cov_sqrt = PointAd (d_cov_sqrt);
            Set_<PointAd > cluster_lls;

            for (NumN i = 1; i <= n_component; ++i) {
                auto log_proportion = ad_vec_share_sub_from_to(log_prs, i, i);
                auto mean = ad_vec_share_sub_from_to(means, (i - 1) * n_feature + 1, i * n_feature);
                // maybe use std::shared_ptr to reduce memory
                mean = ad_share_reshape(mean, p_d_mean);
                auto cov_sqrt = ad_vec_share_sub_from_to(covs_sqrt,
                                                         (i - 1) * n_feature * n_feature + 1,
                                                         i * n_feature * n_feature);
//                ad_get_value(cov_sqrt).reshape(d_cov_sqrt);
                cov_sqrt = ad_share_reshape(cov_sqrt, p_d_cov_sqrt);
                // todo: fuse
                auto cov_sqrt_t = ad_mat_trans(cov_sqrt);
                auto cov = ad_mat_mul(cov_sqrt_t, cov_sqrt);
                auto _ = log_proportion + ad_mvn_logpdf(data, mean, cov);
                cluster_lls.add(_);
            }
            VecN axes(1);
            axes = 1;
            // ad_logsumexp: get log likelihood of one data
            return ad_sum(ad_logsumexp(ad_row_pack(cluster_lls), axes));
        }

        // repeats: The number of repetitions for each element.  `repeats` is broadcasted
        //        to fit the shape of the given axis.
        // axis: The axis along which to repeat values.  By default, use the
        //        flattened input array, and return a flat output array.
        PointAd ad_repeat(const PointAd &x, const PointAd &repeats, PointAd axis) {
            op_repeat _op_repeat;
            Function &function = _op_repeat;
            VarAd y = function.evaluate(x.getVarAd(), repeats.getVarAd(), axis.getVarAd(), ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_undo_repeat_sum(const PointAd &x, const PointAd &repeats, PointAd axis) {
            op_undo_repeat_sum _op_undo_repeat_sum;
            Function &function = _op_undo_repeat_sum;
            VarAd y = function.evaluate(x.getVarAd(), repeats.getVarAd(), axis.getVarAd(), ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_concatenate(const PointAd &x1, const PointAd &x2, PointAd axis) {
            op_concatenate _op_concatenate;
            Function &function = _op_concatenate;
            VarAd y = function.evaluate(x1.getVarAd(), x2.getVarAd(), axis.getVarAd(), ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_concatenate(const PointAd &x1, const PointAd &x2, const PointAd &x3, PointAd axis) {
            op_concatenate _op_concatenate;
            Function &function = _op_concatenate;
            SetVar X;
            X.add(x1.getVarAd());
            X.add(x2.getVarAd());
            X.add(x3.getVarAd());
            X.add(axis.getVarAd());
            VarAd y = function.evaluate(X, ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_concatenate(const Seqce<PointAd > &xs, PointAd axis) {
            op_concatenate _op_concatenate;
            Function &function = _op_concatenate;
            SetVar X;
            for (NumN i = 1; i <= xs.size(); ++i) {
                X.add(xs(i).getVarAd());
            }
            X.add(axis.getVarAd());
            VarAd y = function.evaluate(X, ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_axis_i_sub_get(
                const PointAd &x, const PointAd &offset, const PointAd &di, const PointAd &axis) {
            op_axis_i_sub_get _;
            Function &function = _;
            SetVar X;
            X.add(x.getVarAd());
            X.add(offset.getVarAd());
            X.add(di.getVarAd());
            X.add(axis.getVarAd());
            VarAd y = function.evaluate(X, ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_axis_i_sub_set(
                const PointAd &x, const PointAd &offset, const PointAd &d_y, const PointAd &axis) {
            op_axis_i_sub_set _;
            Function &function = _;
            SetVar X;
            X.add(x.getVarAd());
            X.add(offset.getVarAd());
            X.add(d_y.getVarAd());
            X.add(axis.getVarAd());
            VarAd y = function.evaluate(X, ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_axis_swap(const PointAd &x, const PointAd &pos, const PointAd &pos2) {
            op_axis_swap _;
            Function &function = _;
            VarAd y = function.evaluate(x.getVarAd(), pos.getVarAd(), pos2.getVarAd(), ad_global_get_data());
            return PointAd(y);
        }
    }
}