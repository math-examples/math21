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
#include "detail/set_vector.h"

namespace math21 {
    template<typename T>
    void math21_op_vector_set_value(Tensor<T> &x, NumR value) {
        detail::imath21_op_vector_f_set_value(m21_fname_none, x, value);
    }

    template<typename T1, typename T2>
    void math21_op_vector_set_by_vector(const Tensor<T1> &x, Tensor<T2> &y,
                                        NumN stride_x = 1, NumN stride_y = 1, NumN offset_x = 0, NumN offset_y = 0,
                                        NumN n = 0) {
        detail::imath21_op_vector_f_set_by_vector(
                m21_fname_none, x, y, stride_x, stride_y, offset_x, offset_y, n);
    }

    template<typename T>
    void math21_op_vector_sub_region_set(const Tensor<T> &x, Tensor<T> &y,
                                         NumN offset_x, NumN offset_y, NumN n) {
        math21_op_vector_set_by_vector(x, y, 1, 1, offset_x, offset_y, n);
    }
}