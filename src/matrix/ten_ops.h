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

#include "vector/files_c.h"
#include "matrix.h"
#include "ten.h"

namespace math21 {

    template<typename T>
    void math21_operator_reshape(Tensor <T> &v, const VecN &d) {
        if (!v.isSameSize(d)) {
            v.reshape(d);
        }
    }

    template<typename T>
    void math21_operator_reshape_to_vector(Tensor <T> &v) {
        VecN d(1);
        d = v.size();
        math21_operator_reshape(v, d);
    }

    template<typename T>
    void math21_operator_reshape_to_row_vector(Tensor <T> &v) {
        VecN d(2);
        d = 1, v.size();
        math21_operator_reshape(v, d);
    }

    template<typename T>
    void math21_operator_reshape_to_matrix(Tensor <T> &v, NumN d1, NumN d2) {
        VecN d(2);
        d = d1, d2;
        math21_operator_reshape(v, d);
    }

    template<typename T>
    void math21_operator_reshape_to_tensor3d(Tensor <T> &v, NumN d1, NumN d2, NumN d3) {
        VecN d(3);
        d = d1, d2, d3;
        math21_operator_reshape(v, d);
    }

    template<typename T>
    void math21_operator_reshape_to_tensor4d(Tensor <T> &v, NumN d1, NumN d2, NumN d3, NumN d4) {
        VecN d(4);
        d = d1, d2, d3, d4;
        math21_operator_reshape(v, d);
    }

    // Has error!
    template<typename T>
    const T *math21_memory_tensor_data_address(const Tensor <T> &A) {
        return A.getDataAddress();
    }

    // Has error!
    // But still using and testing!
    template<typename T>
    T *math21_memory_tensor_data_address(Tensor <T> &A) {
        return A.getDataAddress();
    }

    // deprecated, use math21_operator_tensor_copy_data_cpu instead.
    template<typename T>
    void math21_memory_tensor_data_copy_to_tensor_cpu(Tensor <T> &A, const T *data) {
        MATH21_ASSERT(A.isContinuous())
        SpaceParas paras = A.getSpace();
        if (data != (const T *) paras.start) {
            math21_memory_memcpy(paras.start, data, sizeof(T) * A.size());
        }
    }

    template<typename T>
    void math21_memory_tensor_data_copy_to_buffer_cpu(const Tensor <T> &A, T *data) {
        MATH21_ASSERT(A.isContinuous())
        SpaceParas paras = A.getSpace();
        if (data != (const T *) paras.start) {
            math21_memory_memcpy(data, paras.start, sizeof(T) * A.size());
        }
    }

    // start from k.
    template<typename T>
    NumB math21_operator_isContinuousIntegers(const Tensor <T> &x, NumZ k = 1) {
        MATH21_ASSERT(!x.isEmpty());
        return math21_operator_container_isContinuousIntegers(x, k);
    }

    template<typename T>
    NumB math21_operator_isEqual(const Tensor <T> &x, const Tensor <T> &y, NumR epsilon = 0) {
        if (y.isSameSize(x.shape()) == 0) {
            return 0;
        }
        if (x.isEmpty()) {
            return 1;
        }
        return math21_operator_container_isEqual(x, y, epsilon);
    }

    template<typename T>
    NumN math21_operator_number_of_equal(NumR k, const Tensor <T> &x) {
        MATH21_ASSERT(!x.isEmpty());
        return math21_operator_container_number_of_equal(k, x);
    }

    template<typename T>
    NumB math21_operator_is_not_less(const Tensor <T> &x, NumR k) {
        MATH21_ASSERT(!x.isEmpty());
        return math21_operator_container_is_not_less(x, k);
    }

    template<typename T>
    NumB math21_operator_is_less(const Tensor <T> &x, NumR k) {
        MATH21_ASSERT(!x.isEmpty());
        return math21_operator_container_is_less(x, k);
    }

    template<typename T>
    NumB math21_operator_is_not_larger(const Tensor <T> &x, NumR k) {
        MATH21_ASSERT(!x.isEmpty());
        return math21_operator_container_is_not_larger(x, k);
    }

    template<typename T>
    NumB math21_operator_is_larger(const Tensor <T> &x, NumR k) {
        MATH21_ASSERT(!x.isEmpty());
        return math21_operator_container_is_larger_number(x, k);
    }

    template<typename T>
    T math21_operator_multiply_all(const Tensor <T> &A) {
        MATH21_ASSERT(!A.isEmpty(), "empty matrix");
        return math21_operator_container_multiply_all(A);
    }

    // share data and use its own shape
    template<typename T>
    void math21_operator_share_data(const Tensor <T> &v, Tensor <T> &w) {
        MATH21_ASSERT(v.volume() == w.volume());
        SpaceParas paras = v.getSpace();
        w.setSpace(paras);
    }

    // share data and use new shape
    // vanilla share
    template<typename T>
    void math21_operator_share_copy(const Tensor <T> &v, Tensor <T> &w) {
        SpaceParas paras = v.getSpace();
        VecN d;
        w.setSize(v.shape(d), &paras);
    }

    // v is seen as vector
    template<typename T>
    void math21_operator_share_part_reshape(const Tensor <T> &v, Tensor <T> &w, const VecN &d,
                                            NumN offset) {
        NumN volume_all = v.size();
        NumN volume = math21_operator_container_multiply_all(d);
        MATH21_ASSERT(offset + volume <= volume_all)
        SpaceParas paras = v.getSpace();
        SpaceParas paras_dst;
        math21_memory_getSpace(paras, paras_dst, offset, volume, sizeof(T));
        w.setSize(d, &paras_dst);
    }

    // v is seen as vector
    template<typename T>
    void math21_operator_share_vector_part_using_from_to(
            const Tensor <T> &v, Tensor <T> &w, NumZ from_, NumZ to_) {
        NumN from = math21_number_container_pos_check(v.size(), from_);
        NumN to = math21_number_container_pos_check(v.size(), to_);
        NumN offset = from - 1;
        VecN d(1);
        d = to - offset;
        math21_operator_share_part_reshape(v, w, d, offset);
    }

    // Todo: Deep question: Is it safe if v and w have a piece of common space? (next version)
    // share data and use given shape
    template<typename T>
    void math21_operator_share_reshape(const Tensor <T> &v, Tensor <T> &w, const VecN &d,
                                       NumN offset = 0) {
        if (v.volume() == math21_operator_multiply_all(d)) {
            MATH21_ASSERT(offset == 0);
            SpaceParas paras = v.getSpace();
            w.setSize(d, &paras);
        } else {
            math21_operator_share_part_reshape(v, w, d, offset);
        }
    }

    template<typename T>
    void math21_operator_tensor_set_size_cpu(
            Tensor <T> &A, const VecN &d, const void *byte, NumB hasRef) {
        MATH21_ASSERT(A.isBasicType());
        if (!byte) {
            A.clear();
            return;
        }
        NumN n = math21_operator_container_multiply_all(d);
        SpaceParas paras;
        math21_memory_setSpace_cpu(paras, (void *) byte,
                                   n * sizeof(T), hasRef, math21_type_get<T>());
        A.setSize(d, &paras);
    }

    template<typename T>
    void math21_operator_tensor_set_size_wrapper(
            Tensor <T> &A, const VecN &d, PtrVoidInWrapper byte, NumB hasRef) {
        MATH21_ASSERT(A.isBasicType());
        if (math21_vector_isEmpty_wrapper(byte)) {
            A.clear();
            return;
        }
        NumN n = math21_operator_container_multiply_all(d);
        SpaceParas paras;
        math21_memory_setSpace_wrapper(paras, (PtrVoidWrapper) byte,
                                       n * sizeof(T), hasRef, math21_type_get<T>());
        A.setSize(d, &paras);
    }

    // not take in charge of data
    template<typename T>
    void math21_operator_tensor_set_data_cpu(
            Tensor <T> &A, const VecN &d, const void *data) {
        math21_operator_tensor_set_size_cpu(A, d, data, 0);
    }

    // not take in charge of data
    template<typename T>
    void math21_operator_tensor_set_data_cpu(
            Tensor <T> &A, NumN n, const void *data) {
        VecN d(1);
        d = n;
        math21_operator_tensor_set_data_cpu(A, d, data);
    }

    // not take in charge of data
    template<typename T>
    void math21_operator_tensor_set_data_wrapper(
            Tensor <T> &A, const VecN &d, PtrVoidInWrapper data) {
        math21_operator_tensor_set_size_wrapper(A, d, data, 0);
    }

    // not take in charge of data
    template<typename T>
    void math21_operator_tensor_set_data_wrapper(
            Tensor <T> &A, NumN n, PtrVoidInWrapper data) {
        VecN d(1);
        d = n;
        math21_operator_tensor_set_data_wrapper(A, d, data);
    }

    // take in charge of data
    template<typename T>
    void math21_operator_tensor_from_data_cpu(Tensor <T> &A, const VecN &d, const void *data) {
        math21_operator_tensor_set_size_cpu(A, d, data, 1);
    }

    // take in charge of data
    template<typename T>
    void math21_operator_tensor_from_data_cpu(Tensor <T> &A, NumN n, const void *data) {
        VecN d(1);
        d = n;
        math21_operator_tensor_from_data_cpu(A, d, data);
    }

    // take in charge of data
    template<typename T>
    void math21_operator_tensor_from_data_wrapper(Tensor <T> &A, const VecN &d, PtrVoidInWrapper data) {
        math21_operator_tensor_set_size_wrapper(A, d, data, 1);
    }

    // take in charge of data
    template<typename T>
    void math21_operator_tensor_from_data_wrapper(Tensor <T> &A, NumN n, PtrVoidInWrapper data) {
        VecN d(1);
        d = n;
        math21_operator_tensor_from_data_wrapper(A, d, data);
    }

    // see math21_memory_tensor_data_copy_to_tensor_cpu
    template<typename T>
    void math21_operator_tensor_copy_data_cpu(
            Tensor <T> &A, const VecN &d, const void *data) {
        Tensor<T> B;
        math21_operator_tensor_set_data_cpu(B, d, data);
        A = B;
    }

    template<typename T>
    void math21_operator_tensor_copy_data_cpu(
            Tensor <T> &A, NumN n, const void *data) {
        VecN d(1);
        d = n;
        math21_operator_tensor_copy_data_cpu(A, d, data);
    }

    template<typename T>
    void math21_operator_tensor_copy_data_wrapper(Tensor <T> &A, const VecN &d, PtrVoidInWrapper data) {
        Tensor<T> B;
        math21_operator_tensor_set_data_wrapper(B, d, data);
        A = B;
    }

    template<typename T>
    void math21_operator_tensor_copy_data_wrapper(Tensor <T> &A, NumN n, PtrVoidInWrapper data) {
        VecN d(1);
        d = n;
        math21_operator_tensor_copy_data_wrapper(A, d, data);
    }

    template<typename T>
    void math21_operator_share_to_vector(const Tensor <T> &v, Tensor <T> &w, NumN d1 = 0,
                                         NumN offset = 0) {
        NumN n = v.volume();
        MATH21_ASSERT(offset < n);
        n -= offset;
        if (d1 == 0)d1 = n;
        MATH21_ASSERT(d1 <= n);
        VecN d(1);
        d = d1;
        math21_operator_share_reshape(v, w, d, offset);
    }

    template<typename T>
    void math21_operator_share_to_row_vector(const Tensor <T> &v, Tensor <T> &w, NumN d2 = 0,
                                             NumN offset = 0) {
        NumN n = v.volume();
        MATH21_ASSERT(offset < n);
        n -= offset;
        if (d2 == 0)d2 = n;
        MATH21_ASSERT(d2 <= n);
        VecN d(2);
        d = 1, d2;
        math21_operator_share_reshape(v, w, d, offset);
    }

    // Deprecated. To be removed in version 8
    template<typename T>
    void math21_operator_share_reshape_to_row_vector(const Tensor <T> &v, Tensor <T> &w) {
        math21_operator_share_to_row_vector(v, w);
    }

    template<typename T>
    void math21_operator_share_to_matrix(const Tensor <T> &v, Tensor <T> &w,
                                         NumN d1, NumN d2, NumN offset = 0) {
        MATH21_ASSERT(d1 + d2);
        NumN n = v.volume();
        MATH21_ASSERT(offset < n);
        n -= offset;
        if (d1 == 0)d1 = n / d2;
        else if (d2 == 0)d2 = n / d1;
        MATH21_ASSERT(d1 * d2 <= n);
        VecN d(2);
        d = d1, d2;
        math21_operator_share_reshape(v, w, d, offset);
    }

    template<typename T>
    void math21_operator_share_to_tensor3d(const Tensor <T> &v, Tensor <T> &w,
                                           NumN d1, NumN d2, NumN d3, NumN offset = 0) {
        MATH21_ASSERT((d1 + d2) && (d1 + d3) && (d2 + d3));
        NumN n = v.volume();
        MATH21_ASSERT(offset < n);
        n -= offset;
        if (d1 == 0)d1 = n / (d2 * d3);
        else if (d2 == 0)d2 = n / (d1 * d3);
        else if (d3 == 0)d3 = n / (d1 * d2);
        MATH21_ASSERT(d1 * d2 * d3 <= n);
        VecN d(3);
        d = d1, d2, d3;
        math21_operator_share_reshape(v, w, d, offset);
    }

    // Deprecated. To be removed in version 8
    template<typename T>
    void math21_operator_share_reshape_to_tensor(const Tensor <T> &v, Tensor <T> &w, NumN d1, NumN d2, NumN d3) {
        math21_operator_share_to_tensor3d(v, w, d1, d2, d3);
    }

    template<typename T>
    void math21_operator_share_to_tensor4d(const Tensor <T> &v, Tensor <T> &w,
                                           NumN d1, NumN d2, NumN d3, NumN d4, NumN offset = 0) {
        MATH21_ASSERT(d1 * d2 * d3 * d4);
        NumN n = v.volume();
        MATH21_ASSERT(offset < n);
        n -= offset;
        MATH21_ASSERT(d1 * d2 * d3 * d4 <= n);
        VecN d(4);
        d = d1, d2, d3, d4;
        math21_operator_share_reshape(v, w, d, offset);
    }

    template<typename T>
    void math21_operator_share_2d_to_3d(const Tensor <T> &v, Tensor <T> &w) {
        MATH21_ASSERT(v.dims() == 2)
        VecN d(3);
        d = 1, v.dim(1), v.dim(2);
        math21_operator_share_reshape(v, w, d, 0);
    }

    template<typename T>
    void math21_operator_share_remove_dim_1(const Tensor <T> &v, Seqce <Tensor<T>> &ws) {
        MATH21_ASSERT(v.dims() >= 2, "v.dims() = " << v.dims())
        NumN n = v.dim(1);
        VecN d(v.dims() - 1);
        math21_operator_container_set_partially(v.shape(), d, 1);
        NumN volume = math21_operator_container_multiply_all(d);

        ws.setSize(n);
        SpaceParas paras = v.getSpace();

        NumN offset = 0;
        SpaceParas paras_dst;
        for (NumN i = 1; i <= n; ++i) {
            math21_memory_getSpace(paras, paras_dst, offset, volume, sizeof(T));
            ws(i).setSize(d, &paras_dst);
            offset = offset + volume;
        }
    }

    // dim 1 removed
    template<typename T>
    void math21_operator_share_tensor_row_i(const Tensor <T> &v, NumN i, Tensor <T> &w) {
        MATH21_ASSERT(v.dims() >= 2, "v.dims() = " << v.dims());
        MATH21_ASSERT(xjIsIn(i, 1, v.dim(1)));
        VecN d(v.dims() - 1);
        math21_operator_container_set_partially(v.shape(), d, 1);
        NumN volume = math21_operator_container_multiply_all(d);
        math21_operator_share_part_reshape(v, w, d, (i - 1) * volume);
    }

    // get sub, dim 1 kept
    template<typename T>
    void
    math21_operator_share_tensor_rows(const Tensor <T> &v, NumN offset, NumN n, Tensor <T> &w) {
        MATH21_ASSERT(v.dims() >= 1, "v.dims() = " << v.dims());
        MATH21_ASSERT(offset < v.dim(1));
        n = math21_number_container_get_n(n, v.dim(1), 1, offset);
        NumN unit = v.size() / v.dim(1);
        VecN d;
        v.shape(d);
        d.at(1) = n;
        math21_operator_share_part_reshape(v, w, d, offset * unit);
    }

    // v is seen as vector
    // get i-th part from the whole space, each part having shape d.
    template<typename T>
    void math21_operator_share_part_to_i_tensor(const Tensor <T> &v, NumN i, const VecN &d,
                                                Tensor <T> &w) {
        NumN volume = math21_operator_container_multiply_all(d);
        NumN offset = (i - 1) * volume;
        math21_operator_share_part_reshape(v, w, d, offset);
    }

    // get i-th part from the whole space, each part having shape (nr,nc).
    template<typename T>
    void math21_operator_share_part_to_i_mat(const Tensor <T> &v, NumN i, NumN nr, NumN nc,
                                             Tensor <T> &w) {
        VecN d(2);
        d = nr, nc;
        math21_operator_share_part_to_i_tensor(v, i, d, w);
    }

    // see math21_operator_share_remove_dim_1
    template<typename T>
    void math21_operator_share_mat_to_mats(const Tensor <T> &v, Seqce <Tensor<T>> &ws, NumN nr,
                                           NumN nc) {
        MATH21_ASSERT(v.dims() == 2, "v.dims() = " << v.dims())
        NumN n = v.dim(1);
        VecN d(v.dims() - 1);
        math21_operator_container_set_partially(v.shape(), d, 1);
        NumN volume = math21_operator_container_multiply_all(d);
        MATH21_ASSERT(volume == nr * nc,
                      "volume = " << volume << ", nr =  " << nr << ", nc = " << nc);
        VecN shape(2);
        shape = nr, nc;

        ws.setSize(n);
        SpaceParas paras = v.getSpace();

        NumN offset = 0;
        SpaceParas paras_dst;
        for (NumN i = 1; i <= n; ++i) {
            math21_memory_getSpace(paras, paras_dst, offset, volume, sizeof(T));
            ws(i).setSize(shape, &paras_dst);
            offset = offset + volume;
        }
    }

    // b is a kind of shape mask, and non-zero element denotes removing.
    // return shrinked shape
    // e.x., b = 2, 1, 0.
    template<typename VecType>
    void
    math21_operator_tensor_shrink_shape_using_shape(const VecType &shape, const VecN &b, VecN &d) {
        NumN n = math21_operator_number_of_equal(0, b);
        if (n == 0) {
            d.clear();
            return;
        }
        d.setSize(n);
        for (NumN i = 1, j = 1; i <= b.size(); ++i) {
            if (b(i) == 0) {
                d(j) = shape(i);
                ++j;
            }
        }
    }

    template<typename T>
    void math21_operator_tensor_shrink_shape_using_axes_with_dim_kept(const Tensor <T> &A,
                                                                      const VecN &axes, VecN &d) {
        if (axes.isEmpty()) {
            d.setSize(A.dims());
            d = 1;
        } else {
            A.shape(d);
            for (NumN i = 1; i <= axes.size(); ++i) {
                math21_tool_assert(axes(i) <= d.size());
                d(axes(i)) = 1;
            }
        }
    }
}