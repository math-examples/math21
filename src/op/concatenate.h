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

#include "subregion_vec.h"
#include "subregion_mat.h"

namespace math21 {

    template<typename T>
    void math21_op_vector_concatenate(const Seqce<const Tensor <T> *> &xs, Tensor <T> &y) {
        if (xs.isEmpty()) {
            return;
        }
        const Tensor<T> &x1 = *xs(1);
        VecN d2s(xs.size());
        for (NumN i = 1; i <= xs.size(); ++i) {
            const Tensor<T> &xi = *xs(i);
            d2s(i) = xi.size();
        }
        NumN di = static_cast<NumN>(math21_operator_container_sum(d2s, 1));

        y.setDeviceType(x1.getDeviceType());
        if (y.size() != di)y.setSize(di);

        VecN offsets(xs.size());
        math21_operator_container_cdf_like(d2s, offsets, 1);

        for (NumN i = 1; i <= xs.size(); ++i) {
            const Tensor<T> &xi = *xs(i);
            math21_op_vector_sub_region_set(xi, y, 0, offsets(i), xi.size());
        }
    }

    template<typename T>
    void math21_op_vector_concatenate(const Tensor <T> &x1, const Tensor <T> &x2, Tensor <T> &y) {
        Seqce<const Tensor<T> *> xs(2);
        xs(1) = &x1;
        xs(2) = &x2;
        math21_op_vector_concatenate(xs, y);
    }

    template<typename T>
    void math21_op_tensor_concatenate(const Seqce<const Tensor <T> *> &xs, Tensor <T> &y, NumZ _axis) {
        if (xs.isEmpty()) {
            return;
        }
        const Tensor<T> &x1 = *xs(1);
        for (NumN i = 2; i <= xs.size(); ++i) {
            const Tensor<T> &xi = *xs(i);
            MATH21_ASSERT(x1.dims() == xi.dims(),
                          "x1.dims() = " << x1.dims() << ", xi.dims() = " << xi.dims())
        }
        NumN axis = math21_number_container_pos_check(x1.dims(), _axis);

        VecN d2s(xs.size());
        for (NumN i = 1; i <= xs.size(); ++i) {
            const Tensor<T> &xi = *xs(i);
            d2s(i) = xi.dim(axis);
        }
        NumN di = static_cast<NumN>(math21_operator_container_sum(d2s, 1));

        VecN dx1;
        x1.shape(dx1);
        dx1(axis) = di;
        for (NumN i = 2; i <= xs.size(); ++i) {
            const Tensor<T> &xi = *xs(i);
            VecN dxi;
            xi.shape(dxi);
            dxi(axis) = di;
            MATH21_ASSERT(math21_operator_container_isEqual(dx1, dxi));
        }
        y.setDeviceType(x1.getDeviceType());
        y.setSize(dx1);

        VecN d(2);
        d(1) = math21_operator_container_multiply_some(x1.shape(), axis - 1);
        d(2) = math21_operator_container_multiply_some(x1.shape(), x1.shape().size() - axis, axis);
        for (NumN i = 1; i <= d.size(); ++i) {
            if (d(i) == 0) {
                d(i) = 1;
            }
        }
        math21_operator_container_linear_to_A(d(2), d2s);
        VecN offsets(xs.size());
        math21_operator_container_cdf_like(d2s, offsets, 1);

        NumN d2_y = di * d(2);
        for (NumN i = 1; i <= xs.size(); ++i) {
            const Tensor<T> &xi = *xs(i);
            math21_op_matrix_like_sub_region_set(xi, y,
                                                 d(1), d2s(i), d(1), d2_y,
                                                 0, 0, 0, offsets(i));
        }

//    const T **data_xs = new const T *[xs.size()];
//    for (NumN i = 1; i <= xs.size(); ++i) {
//        const Tensor<T> &xi = *xs(i);
//        data_xs[i - 1] = xi.getDataAddress();
//    }
//    math21_vector_concatenate_axis_2_in_d2_cpu(data_xs, y.getDataAddress(), xs.size(),
//                                               d(1), d2s.getDataAddress());
//    delete[] data_xs;
    }

    template<typename T>
    void math21_op_tensor_concatenate(const Seqce<Tensor <T>> &xs, Tensor <T> &y, NumZ _axis) {
        Seqce<const Tensor<T> *> pxs(xs.size());
        for (NumN i = 1; i <= xs.size(); ++i) {
            pxs(i) = &xs.operator()(i);
        }
        math21_op_tensor_concatenate(pxs, y, _axis);
    }

    // z = (x, y)
    // axis>=1
    template<typename T>
    void math21_op_tensor_concatenate(const Tensor <T> &x1, const Tensor <T> &x2,
                                      Tensor <T> &y, NumZ axis) {
        Seqce<const Tensor<T> *> xs(2);
        xs(1) = &x1;
        xs(2) = &x2;
        math21_op_tensor_concatenate(xs, y, axis);
    }

    template<typename T>
    void math21_op_tensor_concatenate(const Tensor <T> &x1, const Tensor <T> &x2, const Tensor <T> &x3,
                                      Tensor <T> &y, NumZ axis) {
        Seqce<const Tensor<T> *> xs(3);
        xs(1) = &x1;
        xs(2) = &x2;
        xs(3) = &x3;
        math21_op_tensor_concatenate(xs, y, axis);
    }

    // todo: support when dis is empty.
    // use TenType instead of Tensor <T> to avoid code format error.
    // TenType is
    // y = (x1, x2, ...) -> x1, x2, ...
    // axis>=1
    template<typename TenType>
    void math21_op_tensor_split(const TenType &y, Seqce<TenType *> &xs, const VecN &dis, NumZ _axis) {
        if (dis.isEmpty()) {
            return;
        }
        NumN axis = math21_number_container_pos_check(y.dims(), _axis);
        NumN di = static_cast<NumN>(math21_operator_container_sum(dis, 1));
        MATH21_ASSERT(di == y.dim(axis))
        MATH21_ASSERT(xs.size() == dis.size())
        VecN dy;
        y.shape(dy);
        for (NumN i = 1; i <= xs.size(); ++i) {
            math21_tool_assert(xs.at(i));
            auto &xi = *xs.at(i);
            dy(axis) = dis(i);
            xi.setDeviceType(y.getDeviceType());
            if (!xi.isSameSize(dy)) {
                xi.setSize(dy);
            }
        }

        const auto &x1 = *xs(1);
        VecN d(2);
        d(1) = math21_operator_container_multiply_some(x1.shape(), axis - 1);
        d(2) = math21_operator_container_multiply_some(x1.shape(), x1.shape().size() - axis, axis);
        for (NumN i = 1; i <= d.size(); ++i) {
            if (d(i) == 0) {
                d(i) = 1;
            }
        }

        VecN d2s(xs.size());
        math21_operator_container_linear(d(2), dis, d2s);
        VecN offsets(xs.size());
        math21_operator_container_cdf_like(d2s, offsets, 1);
        NumN d2_y = di * d(2);
        for (NumN i = 1; i <= xs.size(); ++i) {
            auto &xi = *xs(i);
            math21_op_matrix_like_sub_region_set(y, xi,
                                                 d(1), d2_y, d(1), d2s(i),
                                                 0, offsets(i), 0, 0);
        }
//    T *data_xs[xs.size()];
//    for (NumN i = 1; i <= xs.size(); ++i) {
//        auto &xi = xs.at(i);
//        data_xs[i - 1] = xi.getDataAddress();
//    }
//    math21_vector_split_axis_2_in_d2_cpu(data_xs, y.getDataAddress(), xs.size(),
//                                         d(1), d2s.getDataAddress());
    }

    template<typename T>
    void math21_op_tensor_split(const Tensor <T> &y, Seqce <Tensor<T> > &xs, const VecN &dis, NumZ axis) {
        if (xs.size() != dis.size()) {
            xs.setSize(dis.size());
        }
        Seqce<Tensor<T> *> pxs;
        pxs.setSize(xs.size());
        for (NumN i = 1; i <= xs.size(); ++i)pxs(i) = &xs(i);
        math21_op_tensor_split(y, pxs, dis, axis);
    }

    template<typename T>
    void math21_op_tensor_split_addto(const Tensor <T> &y, Seqce <Tensor<T> > &xs, const VecN &dis, NumZ axis) {
        MATH21_ASSERT(0, "TODO");
        if (xs.size() != dis.size()) {
            xs.setSize(dis.size());
        }
        Seqce<Tensor<T> *> pxs;
        pxs.setSize(xs.size());
        for (NumN i = 1; i <= xs.size(); ++i)pxs(i) = &xs(i);
        math21_op_tensor_split(y, pxs, dis, axis);
    }

    template<typename T>
    void math21_op_tensor_split(const Tensor <T> &y, Tensor <T> &x1, Tensor <T> &x2, NumN dx1, NumN dx2, NumZ axis) {
        Seqce<Tensor<T> *> xs(2);
        xs(1) = &x1;
        xs(2) = &x2;
        VecN dis;
        dis.setSize(2);
        dis = dx1, dx2;
        math21_op_tensor_split(y, xs, dis, axis);
    }
}