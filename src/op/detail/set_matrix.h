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
        // tensor-nd is regarded as matrix
        template<typename T>
        void imath21_op_matrix_like_f_set_by_matrix(
                NumN fname,
                const Tensor <T> &x, Tensor <T> &y,
                NumN d1_x, NumN d2_x, NumN d1_y, NumN d2_y,
                NumN stride1_x, NumN stride2_x,
                NumN stride1_y, NumN stride2_y,
                NumN offset1_x, NumN offset2_x,
                NumN offset1_y, NumN offset2_y,
                NumN d1 = 0, NumN d2 = 0) {
            MATH21_ASSERT(xjIsIn(offset1_x + 1, 1, d1_x));
            MATH21_ASSERT(xjIsIn(offset2_x + 1, 1, d2_x));
            MATH21_ASSERT(xjIsIn(offset1_y + 1, 1, d1_y));
            MATH21_ASSERT(xjIsIn(offset2_y + 1, 1, d2_y));
            NumN offset_x, offset_y;
            offset_x = offset1_x * d2_x + offset2_x;
            offset_y = offset1_y * d2_y + offset2_y;
            d1 = math21_number_container_assign_get_n(d1, d1_x, stride1_x, offset1_x, d1_y, stride1_y, offset1_y);
            d2 = math21_number_container_assign_get_n(d2, d2_x, stride2_x, offset2_x, d2_y, stride2_y, offset2_y);
            if (d1 * d2 == 0)return;
            if (fname == m21_fname_none) {
                if (x.is_cpu()) {
                    math21_generic_matrix_set_by_matrix_cpu(
                            d1, d2,
                            x.getDataAddress(), d1_x, d2_x, stride1_x, stride2_x,
                            y.getDataAddress(), d1_y, d2_y, stride1_y, stride2_y,
                            offset_x, offset_y, x.getSpace().type);
                } else {
                    math21_generic_matrix_set_by_matrix_wrapper(
                            d1, d2,
                            x.getDataAddressWrapper(), d1_x, d2_x, stride1_x, stride2_x,
                            y.getDataAddressWrapper(), d1_y, d2_y, stride1_y, stride2_y,
                            offset_x, offset_y, x.getSpace().type);
                }
            } else {
                if (x.is_cpu()) {
                    math21_generic_matrix_f_set_by_matrix_cpu(
                            fname,
                            d1, d2,
                            x.getDataAddress(), d1_x, d2_x, stride1_x, stride2_x,
                            y.getDataAddress(), d1_y, d2_y, stride1_y, stride2_y,
                            offset_x, offset_y, x.getSpace().type);
                } else {
                    math21_generic_matrix_f_set_by_matrix_wrapper(
                            fname,
                            d1, d2,
                            x.getDataAddressWrapper(), d1_x, d2_x, stride1_x, stride2_x,
                            y.getDataAddressWrapper(), d1_y, d2_y, stride1_y, stride2_y,
                            offset_x, offset_y, x.getSpace().type);
                }
            }
        }
    }
}