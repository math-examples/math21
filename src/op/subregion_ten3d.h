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
#include "detail/set_tensor_3d.h"

namespace math21 {
    // tensor-nd is regarded as tensor-3d
    template<typename T>
    void math21_op_tensor_3d_f_set_by_tensor_3d(
            NumN fname,
            const Tensor<T> &x, Tensor<T> &y,
            NumN d1_x, NumN d2_x, NumN d3_x, NumN d1_y, NumN d2_y, NumN d3_y,
            NumN stride1_x, NumN stride2_x, NumN stride3_x,
            NumN stride1_y, NumN stride2_y, NumN stride3_y,
            NumN offset1_x, NumN offset2_x, NumN offset3_x,
            NumN offset1_y, NumN offset2_y, NumN offset3_y,
            NumN d1 = 0, NumN d2 = 0, NumN d3 = 0) {
        detail::imath21_op_tensor_3d_f_set_by_tensor_3d(
                fname, x, y,
                d1_x, d2_x, d3_x, d1_y, d2_y, d3_y,
                stride1_x, stride2_x, stride3_x,
                stride1_y, stride2_y, stride3_y,
                offset1_x, offset2_x, offset3_x,
                offset1_y, offset2_y, offset3_y,
                d1, d2, d3);
    }

    template<typename T>
    void math21_op_tensor_3d_set_by_tensor_3d(
            const Tensor<T> &x, Tensor<T> &y,
            NumN d1_x, NumN d2_x, NumN d3_x, NumN d1_y, NumN d2_y, NumN d3_y,
            NumN stride1_x, NumN stride2_x, NumN stride3_x,
            NumN stride1_y, NumN stride2_y, NumN stride3_y,
            NumN offset1_x, NumN offset2_x, NumN offset3_x,
            NumN offset1_y, NumN offset2_y, NumN offset3_y,
            NumN d1 = 0, NumN d2 = 0, NumN d3 = 0) {
        math21_op_tensor_3d_f_set_by_tensor_3d(m21_fname_none, x, y,
                                               d1_x, d2_x, d3_x, d1_y, d2_y, d3_y,
                                               stride1_x, stride2_x, stride3_x,
                                               stride1_y, stride2_y, stride3_y,
                                               offset1_x, offset2_x, offset3_x,
                                               offset1_y, offset2_y, offset3_y,
                                               d1, d2, d3);
    }

    template<typename T>
    void math21_op_tensor_3d_sub_region_set(
            const Tensor<T> &x, Tensor<T> &y,
            NumN d1_x, NumN d2_x, NumN d3_x, NumN d1_y, NumN d2_y, NumN d3_y,
            NumN offset1_x, NumN offset2_x, NumN offset3_x,
            NumN offset1_y, NumN offset2_y, NumN offset3_y,
            NumN d1 = 0, NumN d2 = 0, NumN d3 = 0) {
        math21_op_tensor_3d_set_by_tensor_3d(
                x, y,
                d1_x, d2_x, d3_x,
                d1_y, d2_y, d3_y,
                1, 1, 1,
                1, 1, 1,
                offset1_x, offset2_x, offset3_x,
                offset1_y, offset2_y, offset3_y,
                d1, d2, d3);
    }

    template<typename T>
    void math21_op_tensor_3d_sub_region_addto(
            const Tensor<T> &x, Tensor<T> &y,
            NumN d1_x, NumN d2_x, NumN d3_x, NumN d1_y, NumN d2_y, NumN d3_y,
            NumN offset1_x, NumN offset2_x, NumN offset3_x,
            NumN offset1_y, NumN offset2_y, NumN offset3_y,
            NumN d1 = 0, NumN d2 = 0, NumN d3 = 0) {
        math21_op_tensor_3d_f_set_by_tensor_3d(
                m21_fname_addto,
                x, y,
                d1_x, d2_x, d3_x,
                d1_y, d2_y, d3_y,
                1, 1, 1,
                1, 1, 1,
                offset1_x, offset2_x, offset3_x,
                offset1_y, offset2_y, offset3_y,
                d1, d2, d3);
    }
}