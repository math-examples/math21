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
        template<typename T>
        void imath21_op_vector_f_set_value(NumN fname, Tensor <T> &x, NumR value) {
            if (fname == m21_fname_none) {
                if (x.is_cpu()) {
                    math21_generic_vector_set_by_value_cpu(
                            x.size(), value, x.getDataAddress(), 1, x.getSpace().type);
                } else {
                    math21_generic_vector_set_by_value_wrapper(
                            x.size(), value, x.getDataAddressWrapper(), 1,
                            x.getSpace().type);
                }
            } else {
                if (x.is_cpu()) {
                    math21_generic_vector_f_set_by_value_cpu(
                            fname,
                            x.size(), value, x.getDataAddress(), 1, x.getSpace().type);
                } else {
                    math21_generic_vector_f_set_by_value_wrapper(
                            fname,
                            x.size(), value, x.getDataAddressWrapper(), 1,
                            x.getSpace().type);
                }
            }
        }

        template<typename T1, typename T2>
        void imath21_op_vector_f_set_by_vector(
                NumN fname,
                const Tensor <T1> &x, Tensor <T2> &y,
                NumN stride_x = 1, NumN stride_y = 1,
                NumN offset_x = 0, NumN offset_y = 0,
                NumN n = 0) {
            MATH21_ASSERT(x.isEmpty() || xjIsIn(offset_x + 1, 1, x.size()));
            MATH21_ASSERT(y.isEmpty() || xjIsIn(offset_y + 1, 1, y.size()));
            NumN n_x = x.size();
            NumN n_y = y.size();
            if (y.isEmpty()) {
                NumN n_max_x;
                n_max_x = math21_number_container_stride_get_n(n_x, stride_x, offset_x);
                MATH21_ASSERT(offset_y == 0 && stride_y == 1);
                n_y = n_max_x;
                y.setSize(n_y);
            }
            n = math21_number_container_assign_get_n(n, n_x, stride_x, offset_x, n_y, stride_y, offset_y);
            if (n == 0)return;
            if (fname == m21_fname_none) {
                if (x.is_cpu()) {
                    math21_generic_vector_set_by_vector_cpu(
                            n, x.getDataAddress(), stride_x, y.getDataAddress(), stride_y,
                            offset_x, offset_y, x.getSpace().type, y.getSpace().type);
                } else {
                    math21_generic_vector_set_by_vector_wrapper(
                            n, x.getDataAddressWrapper(), stride_x, y.getDataAddressWrapper(), stride_y,
                            offset_x, offset_y, x.getSpace().type, y.getSpace().type);
                }
            } else {
                if (x.is_cpu()) {
                    math21_generic_vector_f_set_by_vector_cpu(
                            fname,
                            n, x.getDataAddress(), stride_x, y.getDataAddress(), stride_y,
                            offset_x, offset_y, x.getSpace().type, y.getSpace().type);
                } else {
                    math21_generic_vector_f_set_by_vector_wrapper(
                            fname,
                            n, x.getDataAddressWrapper(), stride_x, y.getDataAddressWrapper(), stride_y,
                            offset_x, offset_y, x.getSpace().type, y.getSpace().type);
                }
            }
        }
    }
}