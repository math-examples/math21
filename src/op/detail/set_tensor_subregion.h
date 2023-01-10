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
        // see math21_op_subtensor_set, math21_op_tensor_3d_f_set_by_tensor_3d
        // set y using sub-tensor x
        // x -> y, x is sub-tensor of y
        template<typename T>
        void imath21_op_subregion_f_set(
                NumN fname,
                const Tensor <T> &x, Tensor <T> &y, const VecN &offset) {
            MATH21_ASSERT(!x.isEmpty() && !y.isEmpty());
            MATH21_ASSERT(x.dims() == y.dims());
            MATH21_ASSERT(x.dims() == offset.size());
            VecN dx, dy;
            dx.setDeviceType(x.getDeviceType());
            dy.setDeviceType(x.getDeviceType());
            x.shape(dx);
            y.shape(dy);
            VecN offset_new;
            offset_new.setDeviceType(x.getDeviceType());
            offset_new = offset;

            // check
            VecN d(x.dims());
            math21_operator_container_addToC(x.shape(), offset, d);
            MATH21_ASSERT(math21_operator_container_is_not_less(y.shape(), d));
            if (fname == m21_fname_none) {
                if (x.is_cpu()) {
                    math21_generic_tensor_subregion_set_or_get_cpu(
                            x.size(), (void *) x.getDataAddress(),
                            y.getDataAddress(),
                            x.dims(),
                            dx.getDataAddress(), dy.getDataAddress(),
                            offset_new.getDataAddress(), 0,
                            x.getSpace().type);
                } else {
                    math21_generic_tensor_subregion_set_or_get_wrapper(
                            x.size(),
                            (PtrVoidWrapper) x.getDataAddressWrapper(),
                            y.getDataAddressWrapper(),
                            x.dims(),
                            (PtrNInWrapper) dx.getDataAddressWrapper(),
                            (PtrNInWrapper) dy.getDataAddressWrapper(),
                            (PtrNInWrapper) offset_new.getDataAddressWrapper(),
                            0,
                            x.getSpace().type);
                }
            } else {
                if (x.is_cpu()) {
                    math21_generic_tensor_subregion_f_set_or_get_cpu(
                            fname,
                            x.size(), (void *) x.getDataAddress(),
                            y.getDataAddress(),
                            x.dims(),
                            dx.getDataAddress(), dy.getDataAddress(),
                            offset_new.getDataAddress(), 0,
                            x.getSpace().type);
                } else {
                    math21_generic_tensor_subregion_f_set_or_get_wrapper(
                            fname,
                            x.size(),
                            (PtrVoidWrapper) x.getDataAddressWrapper(),
                            y.getDataAddressWrapper(),
                            x.dims(),
                            (PtrNInWrapper) dx.getDataAddressWrapper(),
                            (PtrNInWrapper) dy.getDataAddressWrapper(),
                            (PtrNInWrapper) offset_new.getDataAddressWrapper(),
                            0,
                            x.getSpace().type);
                }
            }
        }

        // see math21_op_subtensor_get
        // a special kind of sub, region sub.
        // x <- y, x is sub-tensor of y, here x, y can be empty
        template<typename T>
        void imath21_op_subregion_f_get(
                NumN fname,
                Tensor <T> &x, const Tensor <T> &y, const VecN &offset, const VecN &_dx) {
            x.setDeviceType(y.getDeviceType());
            if (!_dx.isEmpty()) {
                x.setSize(_dx);
            }
            if (x.isEmpty() || y.isEmpty()) {
                return;
            }
            MATH21_ASSERT(x.dims() == y.dims());
            MATH21_ASSERT(x.dims() == offset.size());

            VecN dx, dy;
            dx.setDeviceType(x.getDeviceType());
            dy.setDeviceType(x.getDeviceType());
            x.shape(dx);
            y.shape(dy);
            VecN offset_new;
            offset_new.setDeviceType(x.getDeviceType());
            offset_new = offset;

            // check
            VecN d(x.dims());
            math21_operator_container_addToC(x.shape(), offset, d);
            MATH21_ASSERT(math21_operator_container_is_not_less(y.shape(), d));
            if (fname == m21_fname_none) {
                if (x.is_cpu()) {
                    math21_generic_tensor_subregion_set_or_get_cpu(
                            x.size(), x.getDataAddress(),
                            (void *) y.getDataAddress(),
                            x.dims(),
                            dx.getDataAddress(), dy.getDataAddress(),
                            offset_new.getDataAddress(),
                            1,
                            x.getSpace().type);
                } else {
                    math21_generic_tensor_subregion_set_or_get_wrapper(
                            x.size(),
                            x.getDataAddressWrapper(),
                            (PtrVoidWrapper) y.getDataAddressWrapper(),
                            x.dims(),
                            (PtrNInWrapper) dx.getDataAddressWrapper(),
                            (PtrNInWrapper) dy.getDataAddressWrapper(),
                            (PtrNInWrapper) offset_new.getDataAddressWrapper(),
                            1,
                            x.getSpace().type);
                }
            } else {
                if (x.is_cpu()) {
                    math21_generic_tensor_subregion_f_set_or_get_cpu(
                            fname,
                            x.size(), x.getDataAddress(),
                            (void *) y.getDataAddress(),
                            x.dims(),
                            dx.getDataAddress(), dy.getDataAddress(),
                            offset_new.getDataAddress(),
                            1,
                            x.getSpace().type);
                } else {
                    math21_generic_tensor_subregion_f_set_or_get_wrapper(
                            fname,
                            x.size(),
                            x.getDataAddressWrapper(),
                            (PtrVoidWrapper) y.getDataAddressWrapper(),
                            x.dims(),
                            (PtrNInWrapper) dx.getDataAddressWrapper(),
                            (PtrNInWrapper) dy.getDataAddressWrapper(),
                            (PtrNInWrapper) offset_new.getDataAddressWrapper(),
                            1,
                            x.getSpace().type);
                }
            }
        }
    }
}