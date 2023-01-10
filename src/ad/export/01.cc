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
#include "../functions_04/files.h"
#include "../differential.h"
#include "point.h"
#include "01.h"

namespace math21 {
    namespace ad {

        NumN ad_get_max_device_type(const PointAd &x1, const PointAd &x2) {
            return xjmax(ad_get_device_type(x1), ad_get_device_type(x2));
        }

        NumN ad_get_max_device_type(const PointAd &x1, const PointAd &x2, const PointAd &x3) {
            return xjmax(xjmax(ad_get_device_type(x1), ad_get_device_type(x2)), ad_get_device_type(x3));
        }

        PointAd ad_to_device_type(const PointAd &x, NumN deviceType) {
            if (ad_get_device_type(x) != deviceType) {
                if (ad_point_is_cpu(x)) {
                    return ad_push(x);
                } else {
                    return ad_pull(x);
                }
            } else {
                return x;
            }
        }

        void ad_to_same_device(const PointAd &x1, const PointAd &x2, PointAd &y1, PointAd &y2) {
            NumN deviceType = ad_get_max_device_type(x1, x2);
            y1 = ad_to_device_type(x1, deviceType);
            y2 = ad_to_device_type(x2, deviceType);
        }

        void ad_to_same_device(const PointAd &x1, const PointAd &x2, const PointAd &x3,
                               PointAd &y1, PointAd &y2, PointAd &y3) {
            NumN deviceType = ad_get_max_device_type(x1, x2);
            y1 = ad_to_device_type(x1, deviceType);
            y2 = ad_to_device_type(x2, deviceType);
            y3 = ad_to_device_type(x3, deviceType);
        }

        PointAd ad_sin(const PointAd &p) {
            op_sin sin0;
            Function &function = sin0;
            VarAd x = p.getVarAd();
            VarAd y = function.evaluate(x, ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_cos(const PointAd &p) {
            op_cos cos0;
            Function &function = cos0;
            VarAd x = p.getVarAd();
            VarAd y = function.evaluate(x, ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_sum(const PointAd &p) {
            op_sum sum;
            Function &function = sum;
            VarAd x = p.getVarAd();
            VarAd y = function.evaluate(x, ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_sum(const PointAd &p, const PointAd &axes, NumB isKeepingDims) {
            op_sum sum(isKeepingDims);
            Function &function = sum;
            VarAd y = function.evaluate(p.getVarAd(), axes.getVarAd(), ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_tensor_broadcast(const PointAd &p, const PointAd &d) {
            op_broadcast_tensor sum;
            Function &function = sum;
            VarAd y = function.evaluate(p.getVarAd(), d.getVarAd(), ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_add(const PointAd &_p1, const PointAd &_p2) {
            PointAd p1, p2;
            ad_to_same_device(_p1, _p2, p1, p2);
            op_add add;
            Function &function = add;
            VarAd x1 = p1.getVarAd();
            VarAd x2 = p2.getVarAd();
            VarAd y = function.evaluate(x1, x2, ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_add(const PointAd &_p1, const PointAd &_p2, const PointAd &_p3) {
            PointAd p1, p2, p3;
            ad_to_same_device(_p1, _p2, _p3, p1, p2, p3);
            op_add add;
            Function &function = add;
            VarAd x1 = p1.getVarAd();
            VarAd x2 = p2.getVarAd();
            VarAd x3 = p3.getVarAd();
            VarAd y = function.evaluate(x1, x2, x3, ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_negate(const PointAd &p) {
            TenR m1;
            m1.setSize(1);
            m1 = -1;
            PointAd pm1(m1, 0, ad_get_device_type(p));
            return ad_mul(pm1, p);
        }

        PointAd ad_subtract(const PointAd &_p1, const PointAd &_p2) {
            PointAd p1, p2;
            ad_to_same_device(_p1, _p2, p1, p2);
            return ad_add(p1, ad_negate(p2));
        }

        PointAd ad_mul(const PointAd &_p1, const PointAd &_p2) {
            PointAd p1, p2;
            ad_to_same_device(_p1, _p2, p1, p2);
            op_multiply mul;
            Function &function = mul;
            VarAd x1 = p1.getVarAd();
            VarAd x2 = p2.getVarAd();
            VarAd y = function.evaluate(x1, x2, ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_mul(const PointAd &_p1, const PointAd &_p2, const PointAd &_p3) {
            PointAd p1, p2, p3;
            ad_to_same_device(_p1, _p2, _p3, p1, p2, p3);
            op_multiply mul;
            Function &function = mul;
            VarAd x1 = p1.getVarAd();
            VarAd x2 = p2.getVarAd();
            VarAd x3 = p3.getVarAd();
            VarAd y = function.evaluate(x1, x2, x3, ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_divide(const PointAd &_p1, const PointAd &_p2) {
            PointAd p1, p2;
            ad_to_same_device(_p1, _p2, p1, p2);
            return ad_mul(p1, ad_power(p2, 1, -1));
        }

        // Device type of y is based on that of x.
        // k is a number on cpu or not.
        // y = kx
        PointAd ad_kx(const PointAd &k, const PointAd &x) {
            op_kx _;
            Function &f = _;
            VarAd y = f.evaluate(k.getVarAd(), x.getVarAd(), ad_global_get_data());
            return PointAd(y);
        }

        PointAd operator+(const PointAd &p1, const PointAd &p2) {
            return ad_add(p1, p2);
        }

        PointAd operator-(const PointAd &p1) {
            return ad_negate(p1);
        }

        PointAd operator-(const PointAd &p1, const PointAd &p2) {
            return ad_subtract(p1, p2);
        }

        PointAd operator*(const PointAd &p1, const PointAd &p2) {
            return ad_mul(p1, p2);
        }

        PointAd operator/(const PointAd &p1, const PointAd &p2) {
            return ad_divide(p1, p2);
        }

        PointAd ad_power(const PointAd &x, NumR k, NumR p) {
            op_power power(k, p);
            Function &function = power;
            VarAd y = function.evaluate(x.getVarAd(), ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_exp(const PointAd &x) {
            op_exp exp;
            Function &function = exp;
            VarAd y = function.evaluate(x.getVarAd(), ad_global_get_data());
            return PointAd(y);
        }

        PointAd exp(const PointAd &x) {
            return ad_exp(x);
        }

        PointAd ad_exp(const PointAd &x, NumR base) {
            MATH21_ASSERT(0)
            return PointAd(VarAd(0));
        }

        PointAd ad_log(const PointAd &x) {
            op_log log;
            Function &function = log;
            VarAd y = function.evaluate(x.getVarAd(), ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_log(const PointAd &x, NumR base) {
            op_log log(base);
            Function &function = log;
            VarAd y = function.evaluate(x.getVarAd(), ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_mat_trans(const PointAd &x) {
            op_mat_trans _ad_mat_trans;
            Function &function = _ad_mat_trans;
            VarAd y = function.evaluate(x.getVarAd(), ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_mat_mul(const PointAd &p1, const PointAd &p2) {
            op_mat_mul mul0;
            Function &function = mul0;
            VarAd x1 = p1.getVarAd();
            VarAd x2 = p2.getVarAd();
            VarAd y = function.evaluate(x1, x2, ad_global_get_data());
//            function.forward(x1, x2, y, ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_inner_product(const PointAd &p1, const PointAd &p2) {
            op_inner_product inner_product0;
            Function &function = inner_product0;
            VarAd x1 = p1.getVarAd();
            VarAd x2 = p2.getVarAd();
            VarAd y = function.evaluate(x1, x2, ad_global_get_data());
            return PointAd(y);
        }

        PointAd dot(const PointAd &p1, const PointAd &p2) {
            return ad_inner_product(p1, p2);
        }

        PointAd ad_at(const PointAd &p, NumN index) {
            VecR b;
            auto &d = ad_get_variable(p).getValue().shape();
            b.setSize(d);
            b.zeros();
            b(index) = 1;
            return ad_inner_product(p, b);
        }

        PointAd at(const PointAd &p, NumN index) {
            return ad_at(p, index);
        }

        PointAd ad_push(const PointAd &p) {
            op_push _op_push;
            Function &f_op_push = _op_push;
            VarAd x = p.getVarAd();
            VarAd y = f_op_push.evaluate(x, ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_pull(const PointAd &p) {
            op_pull _op_pull;
            Function &f_op_pull = _op_pull;
            VarAd x = p.getVarAd();
            VarAd y = f_op_pull.evaluate(x, ad_global_get_data());
            return PointAd(y);
        }

        // value is a number and d is shape.
        PointAd ad_create_using_shape(const PointAd &value, const PointAd &d) {
            op_create _;
            Function &f = _;
            VarAd y = f.evaluate(value.getVarAd(), d.getVarAd(), ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_mean(const PointAd &p) {
            op_mean _;
            Function &f = _;
            VarAd y = f.evaluate(p.getVarAd(), ad_global_get_data());
            return PointAd(y);
        }

        PointAd ad_mean(const PointAd &p, const PointAd &axes, const PointAd &isKeepingDims) {
            op_mean _;
            Function &f = _;
            VarAd y = f.evaluate(p.getVarAd(), axes.getVarAd(), isKeepingDims.getVarAd(), ad_global_get_data());
            return PointAd(y);
        }

    }
}