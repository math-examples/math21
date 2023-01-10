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

#include "detail/set_subtensor.h"

namespace math21 {
    template<typename T>
    void math21_op_subtensor_like_set_or_get(Tensor<T> &x, Tensor<T> &y,
                                             const TenN &mask_, const Seqce<TenN> &maps, NumB isGet) {
        TenN mask;
        math21_operator_share_copy(mask_, mask);
        if (mask.isEmpty()) {
            mask.setSize(1);
            mask = 1;
        }
        detail::imath21_op_subtensor_like_f_set_or_get(
                m21_fname_none, x, y, mask, maps, isGet);
    }

    template<typename T>
    void imath21_op_subtensor_set_or_get(Tensor<T> &x, Tensor<T> &y,
                                         const Seqce<VecN> &indexes, NumB isGet) {
        // indexs -> maps
        Seqce<TenN> maps;
        maps.setSize(indexes.size());
        for (NumN i = 1; i <= indexes.size(); ++i) {
            maps.at(i) = indexes(i);
            math21_algorithm_sort(maps.at(i));
            VecN d(i);
            d = 1;
            d(i) = maps(i).size();
            maps.at(i).reshape(d);
        }
        math21_op_subtensor_like_set_or_get(x, y, TenN(), maps, isGet);
    }

    // tensor slice
    template<typename T>
    void math21_op_subtensor_get(Tensor<T> &x, const Tensor<T> &y,
                                 const Seqce<VecN> &indexes) {
        imath21_op_subtensor_set_or_get(x, (Tensor<T> &) y, indexes, 1);
    }

    // x -> y, x is sub-tensor of y
    template<typename T>
    void math21_op_subtensor_set(const Tensor<T> &x, Tensor<T> &y, const Seqce<VecN> &indexes) {
        imath21_op_subtensor_set_or_get((Tensor<T> &) x, y, indexes, 0);
    }

    template<typename T>
    void imath21_op_submatrix_set_or_get(
            Tensor<T> &x0, Tensor<T> &y0, const VecN &rowIndex, const VecN &colIndex, NumB isGet) {
        Tensor<T> x, y;
        math21_operator_share_copy(x0, x);
        math21_operator_share_copy(y0, y);
        VecN d(3);
        d = 1, x.nrows(), x.ncols();
        x.reshape(d);
        d = 1, y.nrows(), y.ncols();
        y.reshape(d);
        Seqce<VecN> indexes(3);
        indexes(1).setSize(1);
        indexes(1) = 1;
        math21_operator_share_copy(rowIndex, indexes.at(2));
        math21_operator_share_copy(colIndex, indexes.at(3));
        imath21_op_subtensor_set_or_get(x, y, indexes, isGet);
        if (isGet) {
            if (x0.isEmpty()) {
                MATH21_ASSERT(x.dims() == 3 && x.dim(1) == 1);
                if (x.dim(3) == 1) {
                    d.setSize(1);
                    d = x.dim(2);
                } else {
                    d.setSize(2);
                    d = x.dim(2), x.dim(3);
                }
                x.reshape(d);
                math21_operator_share_copy(x, x0);
            }
        }
    }

    // see math21_operator_matrix_submatrix
    template<typename T>
    void math21_op_submatrix_get(
            Tensor<T> &x, const Tensor<T> &y, const VecN &rowIndex, const VecN &colIndex) {
        imath21_op_submatrix_set_or_get(x, (Tensor<T> &) y, rowIndex, colIndex, 1);
    }

    template<typename T>
    void math21_op_submatrix_set(
            const Tensor<T> &x, Tensor<T> &y, const VecN &rowIndex, const VecN &colIndex) {
        imath21_op_submatrix_set_or_get((Tensor<T> &) x, y, rowIndex, colIndex, 0);
    }

    // mask of y
    template<typename T>
    void math21_op_submatrix_get_by_mask(
            Tensor<T> &x, const Tensor<T> &y, const VecN &rowMask, const VecN &colMask) {
        VecN rowIndex, colIndex;
        math21_operator_vector_mask_to_index(rowMask, rowIndex);
        math21_operator_vector_mask_to_index(colMask, colIndex);
        imath21_op_submatrix_set_or_get(x, (Tensor<T> &) y, rowIndex, colIndex, 1);
    }

    // Note: y is created and is set to 0 when empty.
    // mask of y
    template<typename T>
    void math21_op_submatrix_set_by_mask(
            const Tensor<T> &x, Tensor<T> &y, const VecN &rowMask, const VecN &colMask) {
        if (y.isEmpty()) {
            y.setDeviceType(x.getDeviceType());
            if (colMask.size() == 1) {
                y.setSize(rowMask.size());
            } else {
                y.setSize(rowMask.size(), colMask.size());
            }
            y = 0;
        }
        VecN rowIndex, colIndex;
        math21_operator_vector_mask_to_index(rowMask, rowIndex);
        math21_operator_vector_mask_to_index(colMask, colIndex);
        imath21_op_submatrix_set_or_get((Tensor<T> &) x, y, rowIndex, colIndex, 0);
    }

    // sub-vector
    template<typename T>
    void math21_op_subvector_get_by_mask(
            Tensor<T> &x, const Tensor<T> &y, const VecN &rowMask) {
        MATH21_ASSERT(y.isColVector());
        VecN colMask(1);
        colMask = 1;
        math21_op_submatrix_get_by_mask(x, y, rowMask, colMask);
    }

    // Note: y is created and is set to 0 when empty.
    // sub-vector
    template<typename T>
    void math21_op_subvector_set_by_mask(
            const Tensor<T> &x, Tensor<T> &y, const VecN &rowMask) {
        MATH21_ASSERT(x.isColVector());
        VecN colMask(1);
        colMask = 1;
        math21_op_submatrix_set_by_mask(x, y, rowMask, colMask);
    }
}