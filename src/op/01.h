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
#include "subregion_vec.h"
#include "subregion_ten.h"

namespace math21 {
    template<typename T>
    void
    math21_op_tensor_sub_axis_i_set(const Tensor <T> &x, Tensor <T> &y, NumN offset, NumZ axis) {
        NumN ax = math21_number_container_pos_check(y.dims(), axis);
        VecN offsets(y.dims());
        offsets = 0;
        offsets(ax) = offset;
        math21_op_subregion_set(x, y, offsets);
    }

    template<typename T>
    void math21_op_tensor_sub_axis_i_get(Tensor <T> &x, const Tensor <T> &y, NumN offset, NumN di,
                                         NumZ axis) {
        NumN ax = math21_number_container_pos_check(y.dims(), axis);
        VecN offsets(y.dims());
        offsets = 0;
        offsets(ax) = offset;
        VecN dx;
        y.shape(dx);
        dx(ax) = di;
        math21_op_subregion_get(x, y, offsets, dx);
    }

    template<typename T>
    void math21_op_tensor_sub_axis_i_expand_and_set(const Tensor <T> &x, Tensor <T> &y, NumN offset,
                                                    NumZ axis) {
        Tensor<T> x_share;
        math21_operator_tensor_share_add_axis(x, x_share, axis);
        math21_op_tensor_sub_axis_i_set(x_share, y, offset, axis);
    }

    template<typename T>
    void math21_op_tensor_sub_axis_i_get_and_shrink(Tensor <T> &x, const Tensor <T> &y, NumN offset,
                                                    NumZ axis) {
        Tensor<T> x0;
        NumN di = 1;
        math21_op_tensor_sub_axis_i_get(x0, y, offset, di, axis);
        math21_operator_tensor_share_remove_axis(x0, x, axis);
    }

    // x, y are matrices.
    // set part x to part y.
    // axis: {1, 2}
    template<typename T>
    void math21_op_matrix_axis_i_set_part_by_mat(
            const Tensor <T> &x, Tensor <T> &y,
            NumN offset1_x, NumN offset2_x, NumN axis_x,
            NumN offset1_y, NumN offset2_y, NumN axis_y,
            NumN n = 0) {
        MATH21_ASSERT(!x.isEmpty() && !y.isEmpty());
        MATH21_ASSERT(x.isMatrixInMath());
        MATH21_ASSERT(y.isMatrixInMath());
        MATH21_ASSERT(axis_x == 1 || axis_x == 2);
        MATH21_ASSERT(axis_y == 1 || axis_y == 2);
        MATH21_ASSERT(xjIsIn(offset1_x + 1, 1, x.nrows()));
        MATH21_ASSERT(xjIsIn(offset2_x + 1, 1, x.ncols()));
        MATH21_ASSERT(xjIsIn(offset1_y + 1, 1, y.nrows()));
        MATH21_ASSERT(xjIsIn(offset2_y + 1, 1, y.ncols()));
        NumN n_max_x;
        if (axis_x == 1) {
            n_max_x = x.ncols() - offset2_x;
        } else {
            n_max_x = x.nrows() - offset1_x;
        }
        NumN n_max_y;
        if (axis_y == 1) {
            n_max_y = y.ncols() - offset2_y;
        } else {
            n_max_y = y.nrows() - offset1_y;
        }
        NumN n_max = xjmin(n_max_x, n_max_y);
        if (n == 0) n = n_max;
        n = xjmin(n_max, n);
        NumN offset_x = offset1_x * x.ncols() + offset2_x;
        NumN offset_y = offset1_y * y.ncols() + offset2_y;
        math21_op_vector_set_by_vector(
                x, y,
                axis_x == 1 ? 1 : x.ncols(),
                axis_y == 1 ? 1 : y.ncols(),
                offset_x, offset_y, n);
    }

    // x is can be tensor, but is regarded as vector
    // set x to y at axis_i
    // axis: {1, 2}
    template<typename T>
    void math21_op_matrix_axis_i_set_part_vector(
            const Tensor <T> &x, Tensor <T> &y, NumN offset_x, NumN offset1_y, NumN offset2_y,
            NumN axis_y,
            NumN n = 0) {
        if (x.dims() > 1) {
            Tensor<T> x_copy;
            math21_operator_share_copy(x, x_copy);
            x_copy.toVector();
            math21_op_matrix_axis_i_set_part_by_mat(x_copy, y, offset_x, 0, 2, offset1_y, offset2_y,
                                                    axis_y, n);
        } else {
            math21_op_matrix_axis_i_set_part_by_mat(x, y, offset_x, 0, 2, offset1_y, offset2_y,
                                                    axis_y, n);
        }
    }

    // x is can be tensor, but is regarded as vector
    template<typename T>
    void math21_op_matrix_axis_i_get_part_vector(
            Tensor <T> &x, const Tensor <T> &y, NumN offset_x,
            NumN offset1_y, NumN offset2_y, NumN axis_y, NumN n = 0) {
        if (x.dims() > 1) {
            Tensor<T> x_copy;
            math21_operator_share_copy(x, x_copy);
            x_copy.toVector();
            math21_op_matrix_axis_i_set_part_by_mat(y, x_copy, offset1_y, offset2_y, axis_y,
                                                    offset_x, 0, 2, n);
        } else {
            math21_op_matrix_axis_i_set_part_by_mat(y, x, offset1_y, offset2_y, axis_y, offset_x, 0,
                                                    2, n);
        }
    }

    template<typename T>
    void math21_op_matrix_set_row(const Tensor <T> &x, Tensor <T> &y, NumN i, NumN offset_x = 0,
                                  NumN offset_y = 0,
                                  NumN n = 0) {
        math21_op_matrix_axis_i_set_part_vector(x, y, offset_x, i - 1, offset_y, 1, n);
    }

    template<typename T>
    void math21_op_matrix_get_row(
            Tensor <T> &x, const Tensor <T> &y, NumN i, NumN offset_x = 0, NumN offset_y = 0,
            NumN n = 0) {
        x.setDeviceType(y.getDeviceType());
        if (x.isEmpty())x.setSize(y.ncols());
        math21_op_matrix_axis_i_get_part_vector(x, y, offset_x, i - 1, offset_y, 1, n);
    }

    template<typename T>
    void math21_op_matrix_set_col(const Tensor <T> &x, Tensor <T> &y, NumN i, NumN offset_x = 0,
                                  NumN offset_y = 0,
                                  NumN n = 0) {
        math21_op_matrix_axis_i_set_part_vector(x, y, offset_x, offset_y, i - 1, 2, n);
    }

    template<typename T>
    void math21_op_matrix_get_col(
            Tensor <T> &x, const Tensor <T> &y, NumN i, NumN offset_x = 0, NumN offset_y = 0,
            NumN n = 0) {
        x.setDeviceType(y.getDeviceType());
        if (x.isEmpty())x.setSize(y.nrows());
        math21_op_matrix_axis_i_get_part_vector(x, y, offset_x, offset_y, i - 1, 2, n);
    }

    template<typename T>
    void math21_op_vector_kx_onto(NumR k, Tensor <T> &x) {
        if (x.is_cpu()) {
            math21_generic_vector_kx_cpu(x.size(), k, (void *) x.getDataAddress(), 1,
                                         x.getSpace().type);
        } else {
            math21_generic_vector_kx_wrapper(x.size(), k, x.getDataAddressWrapper(), 1,
                                             x.getSpace().type);
        }
    }

    template<typename T>
    void math21_op_vector_kx_add_y(NumR k, const Tensor <T> &x, Tensor <T> &y) {
        if (x.is_cpu()) {
            math21_generic_vector_kx_add_y_cpu(
                    x.size(), k, x.getDataAddress(), 1, y.getDataAddress(), 1, x.getSpace().type);
        } else {
            math21_generic_vector_kx_add_y_wrapper(
                    x.size(), k, x.getDataAddressWrapper(), 1, y.getDataAddressWrapper(), 1,
                    x.getSpace().type);
        }
    }

    template<typename T>
    void math21_op_vector_xy(const Tensor <T> &x, Tensor <T> &y) {
        if (x.is_cpu()) {
            math21_generic_vector_xy_cpu(
                    x.size(), x.getDataAddress(), 1, y.getDataAddress(), 1, x.getSpace().type);
        } else {
            math21_generic_vector_xy_wrapper(
                    x.size(), x.getDataAddressWrapper(), 1, y.getDataAddressWrapper(), 1,
                    x.getSpace().type);
        }
    }

    template<typename VecType1, typename VecType2>
    void math21_op_container_sin(const VecType1 &x, VecType2 &y) {
        math21_tool_container_size_assert(x.size() == y.size());
        if (x.is_cpu()) {
            math21_generic_vector_sin_cpu(x.size(), x.getDataAddress(), y.getDataAddress(),
                                          x.getSpace().type);
        } else {
            math21_generic_vector_sin_wrapper(x.size(), x.getDataAddressWrapper(),
                                              y.getDataAddressWrapper(),
                                              x.getSpace().type);
        }
    }

    template<typename VecType1, typename VecType2>
    void math21_op_container_cos(const VecType1 &x, VecType2 &y) {
        math21_tool_container_size_assert(x.size() == y.size());
        if (x.is_cpu()) {
            math21_generic_vector_cos_cpu(x.size(), x.getDataAddress(), y.getDataAddress(),
                                          x.getSpace().type);
        } else {
            math21_generic_vector_cos_wrapper(x.size(), x.getDataAddressWrapper(),
                                              y.getDataAddressWrapper(),
                                              x.getSpace().type);
        }
    }

    template<typename T>
    const Tensor <T> &math21_operator_tensor_to_cpu(const Tensor <T> &x, Tensor <T> &x_new) {
        if (!x.is_cpu()) {
            x_new = x;
            return x_new;
        } else {
            return x;
        }
    }

    template<typename T>
    Tensor <T> &math21_operator_tensor_to_cpu(Tensor <T> &x, Tensor <T> &x_new) {
        if (!x.is_cpu()) {
            x_new = x;
            return x_new;
        } else {
            return x;
        }
    }

    template<typename T>
    NumB math21_op_isEqual(const Tensor <T> &x, const Tensor <T> &y, NumR epsilon = 0) {
        if (x.is_cpu() && y.is_cpu()) {
            return math21_operator_isEqual(x, y, epsilon);
        } else {
            if (y.isSameSize(x.shape()) == 0) {
                return 0;
            }
            if (x.isEmpty()) {
                return 1;
            }
            Tensor<T> x2, y2;
            return math21_operator_container_isEqual(
                    math21_operator_tensor_to_cpu(x, x2),
                    math21_operator_tensor_to_cpu(y, y2),
                    epsilon);
        }
    }

    // C=A+B
    template<typename VecType>
    void math21_op_container_addToC(const VecType &A, const VecType &B, VecType &C) {
        math21_tool_container_size_assert(A.size() == B.size());
        math21_tool_container_size_assert(A.size() == C.size());
        if (A.is_cpu()) {
            math21_generic_vector_addToC_cpu(A.size(), A.getDataAddress(), B.getDataAddress(),
                                             C.getDataAddress(),
                                             A.getSpace().type);
        } else {
            math21_generic_vector_addToC_wrapper(A.size(), A.getDataAddressWrapper(),
                                                 B.getDataAddressWrapper(),
                                                 C.getDataAddressWrapper(), A.getSpace().type);
        }
    }

    // A=A+B
    template<typename VecType>
    void math21_op_container_addToA(VecType &A, const VecType &B) {
        math21_op_container_addToC(A, B, A);
    }

    // B=A+B
    template<typename VecType>
    void math21_op_container_addToB(const VecType &A, VecType &B) {
        math21_op_container_addToC(A, B, B);
    }

    // C=A*B
    template<typename VecType>
    void math21_op_container_mulToC(const VecType &A, const VecType &B, VecType &C) {
        math21_tool_container_size_assert(A.size() == B.size());
        math21_tool_container_size_assert(A.size() == C.size());
        if (A.is_cpu()) {
            math21_generic_vector_mulToC_cpu(A.size(), A.getDataAddress(), B.getDataAddress(),
                                             C.getDataAddress(),
                                             A.getSpace().type);
        } else {
            math21_generic_vector_mulToC_wrapper(A.size(), A.getDataAddressWrapper(),
                                                 B.getDataAddressWrapper(),
                                                 C.getDataAddressWrapper(), A.getSpace().type);
        }
    }

    // A=A*B
    template<typename VecType>
    void math21_op_container_mulToA(VecType &A, const VecType &B) {
        math21_op_container_mulToC(A, B, A);
    }

    // B=A*B
    template<typename VecType>
    void math21_op_container_mulToB(const VecType &A, VecType &B) {
        math21_op_container_mulToC(A, B, B);
    }

    // X -> Y
    template<typename T>
    void math21_op_tensor_broadcast(const Tensor <T> &X, Tensor <T> &Y, const VecN &d) {
        TensorBroadcast<T> A;
        A.set(X, d);
        Y.setSize(A.shape());
        VecN dx, dy;
        // todo: use cpu when opencl>=2
        dx.setDeviceType(X.getDeviceType());
        dy.setDeviceType(X.getDeviceType());
        X.shape(dx);
        Y.shape(dy);
        NumN n = Y.size();
        if (X.is_cpu()) {
            math21_generic_broadcast_in_dn_cpu(n, X.getDataAddress(), Y.getDataAddress(),
                                               X.dims(), dx.getDataAddress(), Y.dims(),
                                               dy.getDataAddress(),
                                               X.getSpace().type);
        } else {
            math21_generic_broadcast_in_dn_wrapper(n, X.getDataAddressWrapper(),
                                                   Y.getDataAddressWrapper(),
                                                   X.dims(),
                                                   (PtrNInWrapper) dx.getDataAddressWrapper(),
                                                   Y.dims(),
                                                   (PtrNInWrapper) dy.getDataAddressWrapper(),
                                                   X.getSpace().type);
        }
    }

// cuda: USE Function pointers to __device__ functions, See https://stackoverflow.com/questions/15644261/cuda-function-pointers
// opencl: https://stackoverflow.com/questions/7391166/does-opencl-support-function-pointers
// fname: m21_fname
// see tf.reduce_sum
// X(i) is 1 or 0.
// 1 variable, 0 parameter.
    template<typename T>
    void math21_op_tensor_f_shrink(const Tensor <T> &A, Tensor <T> &B, const VecN &X,
                                   NumN fname, NumB isKeepingDims = 0) {
        MATH21_ASSERT(X.size() == A.dims())

        // get parameter shape
        VecN dp;
        math21_operator_tensor_shrink_shape_using_shape(A.shape(), X, dp);
        MATH21_ASSERT(dp.size() != A.dims(), "No variable specified!")
        VecN b;
        b.setSize(X.size());
        b = 1;
        // b = 1 - X, because 0 denotes var in shrinkView
        math21_operator_container_linear_to_A(1, b, -1, X);

        VecN y;
        B.setDeviceType(A.getDeviceType());
        if (dp.isEmpty()) {
            B.setSize(1);
        } else {
            B.setSize(dp);
            y.setSize(b.size());
        }

        NumN np = B.size();
        VecN dv;
        math21_operator_tensor_shrink_shape_using_shape(A.shape(), b, dv);
        NumN nv = math21_operator_container_multiply_all(dv);

        VecN dx, dy;
        dx.setDeviceType(A.getDeviceType());
        dy.setDeviceType(A.getDeviceType());
        A.shape(dx);
        B.shape(dy);
        if (A.is_cpu()) {
            math21_generic_tensor_f_shrink_cpu(
                    fname, np, A.getDataAddress(), B.getDataAddress(),
                    A.dims(), dx.getDataAddress(), B.dims(), dy.getDataAddress(),
                    b.size(), b.getDataAddress(),
                    nv, dv.size(), dv.getDataAddress(), A.getSpace().type);
        } else {
            b.convertDeviceType(A.getDeviceType());
            dv.convertDeviceType(A.getDeviceType());

            math21_generic_tensor_f_shrink_wrapper(
                    fname, np, A.getDataAddressWrapper(), B.getDataAddressWrapper(),
                    A.dims(), (PtrNInWrapper) dx.getDataAddressWrapper(), B.dims(),
                    (PtrNInWrapper) dy.getDataAddressWrapper(),
                    b.size(), (PtrNInWrapper) b.getDataAddressWrapper(),
                    nv, dv.size(), (PtrNInWrapper) dv.getDataAddressWrapper(), A.getSpace().type);
        }

        if (isKeepingDims) {
            VecN d(A.dims());
            math21_operator_container_replace_inc(X, d, dp, (NumN) 0);
            B.reshape(d);
        }
    }

    template<typename T>
    void math21_op_tensor_f_shrink_using_axes(const Tensor <T> &x, Tensor <T> &y, const VecN &axes,
                                              NumN fname, NumB isKeepingDims = 0) {
        VecN index;
        math21_operator_tensor_f_shrink_axes_to_index(x.dims(), axes, index);
        math21_op_tensor_f_shrink(x, y, index, fname, isKeepingDims);
    }

    // see math21_op_tensor_f_shrink
    template<typename T>
    void math21_op_tensor_f_inner_product_like_shrink(
            const Tensor <T> &A, const Tensor <T> &A2, Tensor <T> &B, const VecN &X,
            NumN fname, NumB isKeepingDims = 0) {
        VecN dx;
        Seqce<VecN> shapes(2);
        shapes.at(1) = A.shape(dx);
        shapes.at(2) = A2.shape(dx);
        NumB flag = math21_broadcast_is_compatible_in_ele_op(shapes, dx);
        MATH21_ASSERT(flag, "shape not compatible when broadcasting\n"
                << shapes.log("shapes"));

        MATH21_ASSERT(X.size() == dx.size());

        // get parameter shape
        VecN dp;
        math21_operator_tensor_shrink_shape_using_shape(dx, X, dp);
        MATH21_ASSERT(dp.size() != dx.size(), "No variable specified!")
        VecN b;
        b.setSize(X.size());
        b = 1;
        // b = 1 - X, because 0 denotes var in shrinkView
        math21_operator_container_linear_to_A(1, b, -1, X);

        VecN y;
        B.setDeviceType(A.getDeviceType());
        if (dp.isEmpty()) {
            B.setSize(1);
        } else {
            B.setSize(dp);
            y.setSize(b.size());
        }

        NumN np = B.size();
        VecN dv;
        math21_operator_tensor_shrink_shape_using_shape(dx, b, dv);
        NumN nv = math21_operator_container_multiply_all(dv);

        VecN dx1, dx2, dy;
        dx1.setDeviceType(A.getDeviceType());
        dx2.setDeviceType(A.getDeviceType());
        dy.setDeviceType(A.getDeviceType());
        A.shape(dx1);
        A2.shape(dx2);
        B.shape(dy);

        dx.convertDeviceType(A.getDeviceType());
        b.convertDeviceType(A.getDeviceType());
        dv.convertDeviceType(A.getDeviceType());
        if (math21_operator_tensor_is_shape_nearly_same(A.shape(), A2.shape())) {
            if (A.is_cpu()) {
                math21_generic_tensor_f_inner_product_like_shrink_cpu(
                        fname, np,
                        A.getDataAddress(), A2.getDataAddress(), B.getDataAddress(),
                        dx.size(), dx.getDataAddress(), B.dims(), dy.getDataAddress(),
                        b.size(), b.getDataAddress(),
                        nv, dv.size(), dv.getDataAddress(), A.getSpace().type);
            } else {
                math21_generic_tensor_f_inner_product_like_shrink_wrapper(
                        fname, np,
                        A.getDataAddressWrapper(), A2.getDataAddressWrapper(),
                        B.getDataAddressWrapper(),
                        dx.size(), (PtrNInWrapper) dx.getDataAddressWrapper(), B.dims(),
                        (PtrNInWrapper) dy.getDataAddressWrapper(),
                        b.size(), (PtrNInWrapper) b.getDataAddressWrapper(),
                        nv, dv.size(), (PtrNInWrapper) dv.getDataAddressWrapper(),
                        A.getSpace().type);
            }
        } else {
            if (A.is_cpu()) {
                math21_generic_tensor_f_inner_product_like_bcshrink_cpu(
                        fname, np,
                        A.getDataAddress(), A2.getDataAddress(), B.getDataAddress(),
                        dx1.size(), dx1.getDataAddress(), dx2.size(), dx2.getDataAddress(),
                        dx.size(), dx.getDataAddress(), dy.size(), dy.getDataAddress(),
                        b.size(), b.getDataAddress(),
                        nv, dv.size(), dv.getDataAddress(), A.getSpace().type);
            } else {
                math21_generic_tensor_f_inner_product_like_bcshrink_wrapper(
                        fname, np,
                        A.getDataAddressWrapper(), A2.getDataAddressWrapper(),
                        B.getDataAddressWrapper(),
                        dx1.size(), (PtrNInWrapper) dx1.getDataAddressWrapper(),
                        dx2.size(), (PtrNInWrapper) dx2.getDataAddressWrapper(),
                        dx.size(), (PtrNInWrapper) dx.getDataAddressWrapper(),
                        dy.size(), (PtrNInWrapper) dy.getDataAddressWrapper(),
                        b.size(), (PtrNInWrapper) b.getDataAddressWrapper(),
                        nv, dv.size(), (PtrNInWrapper) dv.getDataAddressWrapper(),
                        A.getSpace().type);
            }
        }
        if (isKeepingDims) {
            VecN d(A.dims());
            math21_operator_container_replace_inc(X, d, dp, (NumN) 0);
            B.reshape(d);
        }
    }

    template<typename T>
    void math21_op_tensor_f_inner_product_like_shrink_using_axes
            (const Tensor <T> &x1, const Tensor <T> &x2,
             Tensor <T> &y, const VecN &axes,
             NumN fname, NumB isKeepingDims = 0) {
        VecN dx;
        Seqce<VecN> shapes(2);
        shapes.at(1) = x1.shape(dx);
        shapes.at(2) = x2.shape(dx);
        NumB flag = math21_broadcast_is_compatible_in_ele_op(shapes, dx);
        MATH21_ASSERT(flag, "shape not compatible when broadcasting\n"
                << shapes.log("shapes"));

        VecN index;
        math21_operator_tensor_f_shrink_axes_to_index(dx.size(), axes, index);
        math21_op_tensor_f_inner_product_like_shrink(x1, x2, y, index, fname, isKeepingDims);
    }

    // Y = f(X1, X2)
    template<typename T>
    void math21_op_tensor_f_with_broadcast(NumN fname,
                                           const Tensor <T> &X1,
                                           const Tensor <T> &X2,
                                           Tensor <T> &Y, NumB notSetSize = 0,
                                           NumB useCompatible = 0) {
        VecN d;
        Seqce<VecN> shapes(2);
        shapes.at(1) = X1.shape(d);
        shapes.at(2) = X2.shape(d);
        NumB flag = math21_broadcast_is_compatible_in_ele_op(shapes, d);
        MATH21_ASSERT(flag, "shape not compatible when broadcasting\n"
                << shapes.log("shapes"));
        if (Y.isEmpty()) {
            Y.setDeviceType(X1.getDeviceType());
        } else {
            MATH21_ASSERT(Y.getDeviceType() == X1.getDeviceType());
        }
        if (!useCompatible) {
            if (!notSetSize) {
                Y.setSize(d);
            } else {
                MATH21_ASSERT(math21_operator_tensor_is_shape_nearly_same(Y.shape(), d));
            }
        } else {
            if (Y.isEmpty()) {
                if (!notSetSize) {
                    Y.setSize(d);
                }
            } else {
                MATH21_ASSERT(math21_broadcast_is_compatible_to(d, Y.shape()));
            }
        }
        NumN n = Y.size();
        if (n == 0)return;
        if (math21_operator_tensor_is_shape_nearly_same(X1.shape(), X2.shape())
            && math21_operator_tensor_is_shape_nearly_same(Y.shape(), d)) {
            if (X1.is_cpu()) {
                math21_generic_vector_f_add_like_cpu(fname, n,
                                                     X1.getDataAddress(),
                                                     X2.getDataAddress(), Y.getDataAddress(),
                                                     X1.getSpace().type);
            } else {
                math21_generic_vector_f_add_like_wrapper(fname, n,
                                                         X1.getDataAddressWrapper(),
                                                         X2.getDataAddressWrapper(),
                                                         Y.getDataAddressWrapper(),
                                                         X1.getSpace().type);
            }
        } else {
            VecN dx1, dx2, dy;
            // todo: use cpu when opencl>=2
            dx1.setDeviceType(X1.getDeviceType());
            dx2.setDeviceType(X1.getDeviceType());
            dy.setDeviceType(X1.getDeviceType());
            X1.shape(dx1);
            X2.shape(dx2);
            Y.shape(dy);
            if (X1.is_cpu()) {
                math21_generic_tensor_f_with_broadcast_in_dn_cpu(fname, n,
                                                                 X1.getDataAddress(),
                                                                 X2.getDataAddress(),
                                                                 Y.getDataAddress(),
                                                                 X1.dims(),
                                                                 dx1.getDataAddress(),
                                                                 X2.dims(),
                                                                 dx2.getDataAddress(),
                                                                 Y.dims(),
                                                                 dy.getDataAddress(),
                                                                 X1.getSpace().type);
            } else {
                math21_generic_tensor_f_with_broadcast_in_dn_wrapper(fname, n,
                                                                     X1.getDataAddressWrapper(),
                                                                     X2.getDataAddressWrapper(),
                                                                     Y.getDataAddressWrapper(),
                                                                     X1.dims(),
                                                                     (PtrNInWrapper) dx1.getDataAddressWrapper(),
                                                                     X2.dims(),
                                                                     (PtrNInWrapper) dx2.getDataAddressWrapper(),
                                                                     Y.dims(),
                                                                     (PtrNInWrapper) dy.getDataAddressWrapper(),
                                                                     X1.getSpace().type);
            }
        }
    }

    template<typename T>
    void math21_op_tensor_f_onto_1_with_broadcast(NumN fname,
                                                  Tensor <T> &X1,
                                                  const Tensor <T> &X2) {
        math21_op_tensor_f_with_broadcast(fname, X1, X2, X1, 1);
    }

    template<typename T>
    void math21_op_tensor_f_onto_2_with_broadcast(NumN fname,
                                                  const Tensor <T> &X1,
                                                  Tensor <T> &X2) {
        math21_op_tensor_f_with_broadcast(fname, X1, X2, X2, 1);
    }

    // Y = f(X1, X2)
    template<typename T>
    void math21_op_tensor_f_sin_like(NumN fname,
                                     const Tensor <T> &X1,
                                     Tensor <T> &Y) {
        if (Y.isEmpty()) {
            Y.setDeviceType(X1.getDeviceType());
        } else {
            MATH21_ASSERT(Y.getDeviceType() == X1.getDeviceType());
        }
        Y.setSize(X1.shape());
        NumN n = Y.size();
        if (X1.is_cpu()) {
            math21_generic_vector_f_sin_like_cpu(fname, n,
                                                 X1.getDataAddress(), Y.getDataAddress(),
                                                 X1.getSpace().type);
        } else {
            math21_generic_vector_f_sin_like_wrapper(fname, n,
                                                     X1.getDataAddressWrapper(),
                                                     Y.getDataAddressWrapper(),
                                                     X1.getSpace().type);
        }
    }

    template<typename T>
    void math21_op_tensor_f_sin_like_onto(NumN fname, Tensor <T> &X) {
        math21_op_tensor_f_sin_like(fname, X, X);
    }

    // Y = f(k, X1)
    template<typename T>
    void math21_op_tensor_f_kx_like(NumN fname, NumR k,
                                    const Tensor <T> &X1,
                                    Tensor <T> &Y) {
        if (Y.isEmpty()) {
            Y.setDeviceType(X1.getDeviceType());
        } else {
            MATH21_ASSERT(Y.getDeviceType() == X1.getDeviceType());
        }
        Y.setSize(X1.shape());
        NumN n = Y.size();
        if (X1.is_cpu()) {
            math21_generic_vector_f_kx_like_cpu(fname, n, k,
                                                X1.getDataAddress(), Y.getDataAddress(),
                                                X1.getSpace().type);
        } else {
            math21_generic_vector_f_kx_like_wrapper(fname, n, k,
                                                    X1.getDataAddressWrapper(),
                                                    Y.getDataAddressWrapper(),
                                                    X1.getSpace().type);
        }
    }

    template<typename T>
    void math21_op_tensor_f_kx_like_onto(NumN fname, NumR k, Tensor <T> &X) {
        math21_op_tensor_f_kx_like(fname, k, X, X);
    }

    // see math21_operator_matrix_mul_with_trans_option
    // C = k1*(A*B) + k2*C or similar
    template<typename T>
    void math21_op_mat_mul_linear(NumR k1, NumR k2,
                                  const Tensor <T> &A, const Tensor <T> &B, Tensor <T> &C,
                                  NumB isTransA = 0,
                                  NumB isTransB = 0,
                                  NumB isSettingC = 0) {
        if (C.isEmpty()) {
            C.setDeviceType(A.getDeviceType());
        } else {
            MATH21_ASSERT(C.getDeviceType() == A.getDeviceType());
        }

        NumN nr_C, nc_C, n_common;
        nr_C = !isTransA ? A.nrows() : A.ncols();
        nc_C = !isTransB ? B.ncols() : B.nrows();
        n_common = !isTransA ? A.ncols() : A.nrows();
        MATH21_ASSERT(n_common == (!isTransB ? B.nrows() : B.ncols())); // error-prone

        if (isSettingC) {
            C.setSize(nr_C, nc_C);
            C = 0;
            k2 = 0;
        } else {
            MATH21_ASSERT(C.nrows() == nr_C && C.ncols() == nc_C); // C may be vector
        }

        NumN stride_a, stride_b, stride_c;
        stride_a = A.ncols();
        stride_b = B.ncols();
        stride_c = C.ncols();

        if (A.is_cpu()) {
            math21_generic_matrix_multiply_onto_k1AB_add_k2C_similar_cpu(
                    isTransA, isTransB, nr_C, nc_C, n_common,
                    k1, A.getDataAddress(), stride_a,
                    B.getDataAddress(), stride_b,
                    k2, C.getDataAddress(), stride_c, A.getSpace().type);

        } else {
            math21_generic_matrix_multiply_onto_k1AB_add_k2C_similar_wrapper(
                    isTransA, isTransB, nr_C, nc_C, n_common,
                    k1, A.getDataAddressWrapper(), stride_a,
                    B.getDataAddressWrapper(), stride_b,
                    k2, C.getDataAddressWrapper(), stride_c, A.getSpace().type);
        }
    }

    template<typename T>
    void math21_op_mat_mul(NumR k, const Tensor <T> &A, const Tensor <T> &B, Tensor <T> &C,
                           NumB isTransA = 0,
                           NumB isTransB = 0) {
        math21_op_mat_mul_linear(k, 0, A, B, C, isTransA, isTransB, 1);
    }

    template<typename T>
    void math21_op_mat_mul(const Tensor <T> &A, const Tensor <T> &B, Tensor <T> &C,
                           NumB isTransA = 0,
                           NumB isTransB = 0) {
        math21_op_mat_mul_linear(1, 0, A, B, C, isTransA, isTransB, 1);
    }

    template<typename T>
    void math21_op_tensor_repeat(const Tensor <T> &x, Tensor <T> &y, const VecN &repeats,
                                 NumZ axis = 0) {
        MATH21_ASSERT(axis != 0, "no implement")
        NumN _axis;
        if (axis) {
            _axis = math21_number_container_pos_check(x.dims(), axis);
        } else {
            _axis = 0; // todo: this
        }
        MATH21_ASSERT(repeats.size() == x.dim(_axis) || repeats.size() == 1)

        VecN d;
        math21_operator_container_dn_to_d3(x.shape(), _axis, d);

        VecN d_y;
        x.shape(d_y);
        NumN d2_y;
        if (repeats.size() == 1) {
            d2_y = x.dim(_axis) * repeats(1);
        } else {
            d2_y = (NumN) math21_operator_container_sum(repeats, 1);
        }
        d_y(_axis) = d2_y;
        y.setDeviceType(x.getDeviceType());
        y.setSize(d_y);

        VecN _repeats;
        if (repeats.size() == 1) {
            _repeats.setSize(d(2));
            _repeats = repeats(1);
        } else {
            _repeats = repeats;
        }
        VecN offsets(d(2));
        math21_operator_container_cdf_like(_repeats, offsets, 1);
        for (NumN i = 1; i <= d(2); ++i) {
            for (NumN j = 1; j <= _repeats(i); ++j) {
                math21_op_tensor_3d_sub_region_set(
                        x, y,
                        d(1), d(2), d(3), d(1), d2_y, d(3),
                        0, i - 1, 0,
                        0, offsets(i) + j - 1, 0,
                        d(1), 1, d(3));
            }
        }

//        if (repeats.size() == 1) {
//            math21_vector_repeat_axis_2_in_d3_cpu(x.getDataAddress(), y.getDataAddress(),
//                                                  d(1), d(2), d(3), repeats(1));
//        } else {
//            math21_vector_repeats_axis_2_in_d3_cpu(x.getDataAddress(), y.getDataAddress(),
//                                                   d(1), d(2), d(3), repeats.getDataAddress());
//        }
    }

    template<typename T>
    void math21_op_tensor_sum_undo_repeat(
            Tensor <T> &x, const Tensor <T> &y, const VecN &repeats, NumZ axis = 0) {
        MATH21_ASSERT(axis != 0, "no implement")
        NumN _axis;
        if (axis) {
            _axis = math21_number_container_pos_check(y.dims(), axis);
        } else {
            _axis = 0; // todo: this
        }
        MATH21_ASSERT((NumN) math21_operator_container_sum(repeats, 1) == y.dim(_axis) ||
                      repeats.size() == 1)

        VecN d_x;
        y.shape(d_x);
        NumN d2_x;
        if (repeats.size() == 1) {
            MATH21_ASSERT(y.dim(_axis) % repeats(1) == 0)
            d2_x = y.dim(_axis) / repeats(1);
        } else {
            d2_x = repeats.size();
        }
        d_x(_axis) = d2_x;
        x.setSize(d_x);

        VecN d;
        math21_operator_container_dn_to_d3(d_x, _axis, d);

        NumN d2_y = y.dim(_axis);
        VecN _repeats;
        if (repeats.size() == 1) {
            _repeats.setSize(d(2));
            _repeats = repeats(1);
        } else {
            _repeats = repeats;
        }
        VecN offsets(d(2));
        math21_operator_container_cdf_like(_repeats, offsets, 1);
        x = 0;
        for (NumN i = 1; i <= d(2); ++i) {
            for (NumN j = 1; j <= _repeats(i); ++j) {
                math21_op_tensor_3d_sub_region_addto(
                        y, x,
                        d(1), d2_y, d(3), d(1), d(2), d(3),
                        0, offsets(i) + j - 1, 0,
                        0, i - 1, 0,
                        d(1), 1, d(3));
            }
        }

//    if (repeats.size() == 1) {
//        math21_vector_sum_undo_repeat_axis_2_in_d3_cpu(x.getDataAddress(), y.getDataAddress(),
//                                                       d(1), d(2), d(3), repeats(1));
//    } else {
//        math21_vector_sum_undo_repeats_axis_2_in_d3_cpu(x.getDataAddress(), y.getDataAddress(),
//                                                        d(1), d(2), d(3), repeats.getDataAddress());
//    }
    }

    // todo: see math21_op_tensor_swap_axes
    template<typename T1, typename T2>
    void math21_op_matrix_like_transpose(const Tensor <T1> &X, Tensor <T2> &Y,
                                         NumN d1_x, NumN d2_x) {
        NumN d1_y, d2_y;
        d1_y = d2_x;
        d2_y = d1_x;
        if (Y.isEmpty()) {
            Y.setDeviceType(X.getDeviceType());
        } else {
            MATH21_ASSERT(Y.getDeviceType() == X.getDeviceType());
        }
        MATH21_ASSERT(X.size() == d1_x * d2_x);
        if (Y.isEmpty()) {
            Y.setSize(X.ncols(), X.nrows());
        } else {
            MATH21_ASSERT(Y.size() == d1_y * d2_y);
        }
        NumN n = Y.size();
        if (X.is_cpu()) {
            math21_generic_matrix_transpose_cpu(n,
                                                X.getDataAddress(), Y.getDataAddress(),
                                                d1_x, d2_x,
                                                X.getSpace().type, Y.getSpace().type);
        } else {
            math21_generic_matrix_transpose_wrapper(n,
                                                    X.getDataAddressWrapper(),
                                                    Y.getDataAddressWrapper(),
                                                    d1_x, d2_x,
                                                    X.getSpace().type, Y.getSpace().type);
        }
    }

    template<typename T1, typename T2>
    void math21_op_matrix_like_trans_reverse_axis(const Tensor <T1> &X, Tensor <T2> &Y,
                                                  NumN d1_x, NumN d2_x, NumB isXAxis) {
        NumN d1_y, d2_y;
        d1_y = d2_x;
        d2_y = d1_x;
        if (Y.isEmpty()) {
            Y.setDeviceType(X.getDeviceType());
        } else {
            MATH21_ASSERT(Y.getDeviceType() == X.getDeviceType());
        }
        MATH21_ASSERT(X.size() == d1_x * d2_x);
        if (Y.isEmpty()) {
            Y.setSize(X.ncols(), X.nrows());
        } else {
            MATH21_ASSERT(Y.size() == d1_y * d2_y);
        }
        NumN n = Y.size();
        if (X.is_cpu()) {
            math21_generic_matrix_trans_reverse_axis_cpu(n,
                                                         X.getDataAddress(), Y.getDataAddress(),
                                                         d1_x, d2_x, isXAxis,
                                                         X.getSpace().type, Y.getSpace().type);
        } else {
            math21_generic_matrix_trans_reverse_axis_wrapper(n,
                                                             X.getDataAddressWrapper(),
                                                             Y.getDataAddressWrapper(),
                                                             d1_x, d2_x, isXAxis,
                                                             X.getSpace().type, Y.getSpace().type);
        }
    }

    template<typename T1, typename T2>
    void math21_op_matrix_transpose(const Tensor <T1> &X, Tensor <T2> &Y) {
        if (Y.isEmpty()) {
            Y.setDeviceType(X.getDeviceType());
        } else {
            MATH21_ASSERT(Y.getDeviceType() == X.getDeviceType());
        }
        Y.setSize(X.ncols(), X.nrows());
        math21_op_matrix_like_transpose(X, Y, X.nrows(), X.ncols());
    }

    // rotate pi/2 counter-clockwise when isXAxis = 1 or clockwise when isXAxis = 0
    template<typename T1, typename T2>
    void math21_op_matrix_trans_reverse_axis(const Tensor <T1> &X, Tensor <T2> &Y, NumB isXAxis) {
        if (Y.isEmpty()) {
            Y.setDeviceType(X.getDeviceType());
        } else {
            MATH21_ASSERT(Y.getDeviceType() == X.getDeviceType());
        }
        Y.setSize(X.ncols(), X.nrows());
        math21_op_matrix_like_trans_reverse_axis(X, Y, X.nrows(), X.ncols(), isXAxis);
    }

    // todo: optimize and use container assign when d2 = d4, or |S| =2 with S = {i | di = 1, i in [2, 4]}.
    template<typename T1, typename T2>
    void math21_op_tensor_swap_axes(const Tensor <T1> &x, Tensor <T2> &y, NumZ pos, NumZ pos2) {
        if (y.isEmpty()) {
            y.setDeviceType(x.getDeviceType());
        } else {
            MATH21_ASSERT(y.getDeviceType() == x.getDeviceType());
        }
        NumN _pos = math21_number_container_pos_check(x.dims(), pos);
        NumN _pos2 = math21_number_container_pos_check(x.dims(), pos2);
        NumN p1, p2;
        p1 = xjmin(_pos, _pos2);
        p2 = xjmax(_pos, _pos2);
        MATH21_ASSERT(p1 != p2, "swap same axis");
        VecN d;
        math21_operator_container_dn_to_d5_fix_24(x.shape(), p1, p2, d);
        VecN d_y;
        x.shape(d_y);
        math21_operator_container_swap(d_y, p1, p2);
        y.setSize(d_y);
        if (x.is_cpu()) {
            math21_generic_tensor_swap_axes_24_in_d5_cpu(
                    x.getDataAddress(), y.getDataAddress(),
                    d(1), d(2), d(3), d(4), d(5),
                    x.getSpace().type, y.getSpace().type);
        } else {
            math21_generic_tensor_swap_axes_24_in_d5_wrapper(
                    x.getDataAddressWrapper(),
                    y.getDataAddressWrapper(),
                    d(1), d(2), d(3), d(4), d(5),
                    x.getSpace().type, y.getSpace().type);
        }
    }

    // <=> multiple math21_op_tensor_swap_axes, by Beauty
    template<typename T1, typename T2>
    void math21_op_tensor_move_axis(const Tensor <T1> &x, Tensor <T2> &y, NumZ from, NumZ to) {
        NumN _from = math21_number_container_pos_check(x.dims(), from);
        NumN _to = math21_number_container_pos_check(x.dims(), to);
        MATH21_ASSERT(_from != _to, "move same axis");
        VecN d;
        math21_operator_container_dn_to_d4_fix_2or3(x.shape(), _from, _to, d);
        VecN d_y;
        x.shape(d_y);
        math21_operator_container_move(d_y, _from, _to);
        Tensor<T1> x_s;
        math21_operator_share_reshape(x, x_s, d);
        math21_op_tensor_swap_axes(x_s, y, 2, 3);
        y.reshape(d_y);
    }

    template<typename T>
    void math21_op_tensor_swap_rows(Tensor <T> &x, NumZ i, NumZ j, NumZ axis) {
        if (x.isEmpty())return;
        NumN ax = math21_number_container_pos_check(x.dims(), axis);
        NumN pos = math21_number_container_pos_check(x.dim(ax), i);
        NumN pos2 = math21_number_container_pos_check(x.dim(ax), j);
        if (pos == pos2)return;
        VecN d;
        math21_operator_container_dn_to_d3(x.shape(), ax, d);
        NumN n = x.size() / x.dim(ax);
        if (x.is_cpu()) {
            math21_generic_tensor_3d_swap_row_in_d2_cpu(
                    n, x.getDataAddress(), pos, pos2,
                    d(1), d(2), d(3), x.getSpace().type);
        } else {
            math21_generic_tensor_3d_swap_row_in_d2_wrapper(
                    n, x.getDataAddressWrapper(), pos, pos2,
                    d(1), d(2), d(3), x.getSpace().type);
        }
    }
}