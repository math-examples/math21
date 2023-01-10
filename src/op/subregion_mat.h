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
#include "detail/set_matrix.h"

namespace math21 {
    // tensor-nd is regarded as matrix
    template<typename T>
    void math21_op_matrix_like_set_by_matrix(
            const Tensor<T> &x, Tensor<T> &y,
            NumN d1_x, NumN d2_x, NumN d1_y, NumN d2_y,
            NumN stride1_x, NumN stride2_x,
            NumN stride1_y, NumN stride2_y,
            NumN offset1_x, NumN offset2_x,
            NumN offset1_y, NumN offset2_y,
            NumN d1 = 0, NumN d2 = 0) {
        detail::imath21_op_matrix_like_f_set_by_matrix(
                m21_fname_none,
                x, y, d1_x, d2_x, d1_y, d2_y,
                stride1_x, stride2_x,
                stride1_y, stride2_y,
                offset1_x, offset2_x,
                offset1_y, offset2_y,
                d1, d2);
    }

    template<typename T>
    void math21_op_matrix_set_by_matrix(
            const Tensor<T> &x, Tensor<T> &y,
            NumN stride1_x, NumN stride2_x,
            NumN stride1_y, NumN stride2_y,
            NumN offset1_x, NumN offset2_x,
            NumN offset1_y, NumN offset2_y,
            NumN d1 = 0, NumN d2 = 0) {
        math21_op_matrix_like_set_by_matrix(x, y,
                                            x.nrows(), x.ncols(), y.nrows(), y.ncols(),
                                            stride1_x, stride2_x,
                                            stride1_y, stride2_y,
                                            offset1_x, offset2_x,
                                            offset1_y, offset2_y,
                                            d1, d2);
    }

    template<typename T>
    void math21_op_matrix_set_by_matrix(
            const Tensor<T> &x, Tensor<T> &y,
            NumN stride1_x, NumN stride2_x, NumN stride1_y, NumN stride2_y) {
        math21_op_matrix_set_by_matrix(x, y, stride1_x, stride2_x, stride1_y, stride2_y, 0, 0, 0, 0, 0, 0);
    }

    template<typename T>
    void math21_op_matrix_like_sub_region_set(
            const Tensor<T> &x, Tensor<T> &y,
            NumN d1_x, NumN d2_x, NumN d1_y, NumN d2_y,
            NumN offset1_x, NumN offset2_x,
            NumN offset1_y, NumN offset2_y,
            NumN d1 = 0, NumN d2 = 0) {
        math21_op_matrix_like_set_by_matrix(x, y,
                                            d1_x, d2_x, d1_y, d2_y,
                                            1, 1, 1, 1,
                                            offset1_x, offset2_x,
                                            offset1_y, offset2_y, d1, d2);
    }

    template<typename T>
    void math21_op_matrix_sub_region_set(
            const Tensor<T> &x, Tensor<T> &y,
            NumN offset1_x, NumN offset2_x,
            NumN offset1_y, NumN offset2_y,
            NumN d1, NumN d2) {
        math21_op_matrix_set_by_matrix(x, y, 1, 1, 1, 1,
                                       offset1_x, offset2_x,
                                       offset1_y, offset2_y, d1, d2);
    }

    // top-left set
    template<typename T>
    void math21_op_matrix_sub_region_tl_set(
            const Tensor<T> &x, Tensor<T> &y,
            NumN d1 = 0, NumN d2 = 0) {
        math21_op_matrix_sub_region_set(x, y, 0, 0, 0, 0, d1, d2);
    }

    // top-right set
    template<typename T>
    void math21_op_matrix_sub_region_tr_set(
            const Tensor<T> &x, Tensor<T> &y,
            NumN d1 = 0, NumN d2 = 0) {
        NumN offset1_x, offset2_x, offset1_y, offset2_y;
        offset1_x = 0;
        offset1_y = 0;
        math21_number_get_offset_from_right(x.ncols(), y.ncols(), offset2_x, offset2_y);
        math21_op_matrix_sub_region_set(x, y, offset1_x, offset2_x, offset1_y, offset2_y, d1, d2);
    }

    // bottom-right set
    template<typename T>
    void math21_op_matrix_sub_region_br_set(
            const Tensor<T> &x, Tensor<T> &y,
            NumN d1 = 0, NumN d2 = 0) {
        NumN offset1_x, offset2_x, offset1_y, offset2_y;
        math21_number_get_offset_from_right(x.nrows(), y.nrows(), offset1_x, offset1_y);
        math21_number_get_offset_from_right(x.ncols(), y.ncols(), offset2_x, offset2_y);
        math21_op_matrix_sub_region_set(x, y, offset1_x, offset2_x, offset1_y, offset2_y, d1, d2);
    }

    // bottom-left set
    template<typename T>
    void math21_op_matrix_sub_region_bl_set(
            const Tensor<T> &x, Tensor<T> &y,
            NumN d1 = 0, NumN d2 = 0) {
        NumN offset1_x, offset2_x, offset1_y, offset2_y;
        math21_number_get_offset_from_right(x.nrows(), y.nrows(), offset1_x, offset1_y);
        offset2_x = 0;
        offset2_y = 0;
        math21_op_matrix_sub_region_set(x, y, offset1_x, offset2_x, offset1_y, offset2_y, d1, d2);
    }
}