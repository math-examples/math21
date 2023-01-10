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
    namespace detail {
        // tensor-nd is regarded as tensor-3d
        template<typename T>
        void imath21_op_tensor_3d_f_set_by_tensor_3d(
                NumN fname,
                const Tensor <T> &x, Tensor <T> &y,
                NumN d1_x, NumN d2_x, NumN d3_x, NumN d1_y, NumN d2_y, NumN d3_y,
                NumN stride1_x, NumN stride2_x, NumN stride3_x,
                NumN stride1_y, NumN stride2_y, NumN stride3_y,
                NumN offset1_x, NumN offset2_x, NumN offset3_x,
                NumN offset1_y, NumN offset2_y, NumN offset3_y,
                NumN d1 = 0, NumN d2 = 0, NumN d3 = 0) {
            MATH21_ASSERT(xjIsIn(offset1_x + 1, 1, d1_x));
            MATH21_ASSERT(xjIsIn(offset2_x + 1, 1, d2_x));
            MATH21_ASSERT(xjIsIn(offset3_x + 1, 1, d3_x));
            MATH21_ASSERT(xjIsIn(offset1_y + 1, 1, d1_y));
            MATH21_ASSERT(xjIsIn(offset2_y + 1, 1, d2_y));
            MATH21_ASSERT(xjIsIn(offset3_y + 1, 1, d3_y));
            NumN offset_x, offset_y;
            offset_x = (offset1_x * d2_x + offset2_x) * d3_x + offset3_x;
            offset_y = (offset1_y * d2_y + offset2_y) * d3_y + offset3_y;
            d1 = math21_number_container_assign_get_n(d1, d1_x, stride1_x, offset1_x, d1_y, stride1_y, offset1_y);
            d2 = math21_number_container_assign_get_n(d2, d2_x, stride2_x, offset2_x, d2_y, stride2_y, offset2_y);
            d3 = math21_number_container_assign_get_n(d3, d3_x, stride3_x, offset3_x, d3_y, stride3_y, offset3_y);
            if (d1 * d2 * d3 == 0)return;
            if (fname == m21_fname_none) {
                if (x.is_cpu()) {
                    math21_generic_tensor_3d_set_by_tensor_3d_cpu(
                            d1, d2, d3,
                            x.getDataAddress(), d1_x, d2_x, d3_x, stride1_x, stride2_x, stride3_x,
                            y.getDataAddress(), d1_y, d2_y, d3_y, stride1_y, stride2_y, stride3_y,
                            offset_x, offset_y, x.getSpace().type);
                } else {
                    math21_generic_tensor_3d_set_by_tensor_3d_wrapper(
                            d1, d2, d3,
                            x.getDataAddressWrapper(), d1_x, d2_x, d3_x, stride1_x, stride2_x, stride3_x,
                            y.getDataAddressWrapper(), d1_y, d2_y, d3_y, stride1_y, stride2_y, stride3_y,
                            offset_x, offset_y, x.getSpace().type);
                }
            } else {
                if (x.is_cpu()) {
                    math21_generic_tensor_3d_f_set_by_tensor_3d_cpu(
                            fname, d1, d2, d3,
                            x.getDataAddress(), d1_x, d2_x, d3_x, stride1_x, stride2_x, stride3_x,
                            y.getDataAddress(), d1_y, d2_y, d3_y, stride1_y, stride2_y, stride3_y,
                            offset_x, offset_y, x.getSpace().type);
                } else {
                    math21_generic_tensor_3d_f_set_by_tensor_3d_wrapper(
                            fname, d1, d2, d3,
                            x.getDataAddressWrapper(), d1_x, d2_x, d3_x, stride1_x, stride2_x, stride3_x,
                            y.getDataAddressWrapper(), d1_y, d2_y, d3_y, stride1_y, stride2_y, stride3_y,
                            offset_x, offset_y, x.getSpace().type);
                }
            }
        }
    }
}