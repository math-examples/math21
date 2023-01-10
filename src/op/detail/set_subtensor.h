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
        // see math21_op_set_using_mask, math21_op_subregion_get
        // see math21_geometry_warp_image_using_indexes_cpu_kernel
        // Caller must make sure element in index lies in y.
        // x <- y, x is sub-tensor of y
        // general way of get sub-tensor but is more flexible, because:
        // 1) x can be larger than y
        // 2) index can repeat.
        // 3) mask
        template<typename T>
        void imath21_op_subtensor_like_f_set_or_get(
                NumN fname,
                Tensor <T> &x, Tensor <T> &y,
                const TenN &mask, const Seqce <TenN> &maps, NumB isGet) {
            MATH21_ASSERT(maps.size() == 3);

            VecN d;
            Seqce<VecN> shapes(maps.size());
            for (NumN i = 1; i <= maps.size(); ++i) {
                shapes.at(i) = maps(i).shape(d);
            }
            NumB flag = math21_broadcast_is_compatible_in_ele_op(shapes, d);
            MATH21_ASSERT(flag, "shape not compatible when broadcasting\n"
                    << shapes.log("shapes"));
            if (isGet && x.isEmpty()) {
                x.setDeviceType(y.getDeviceType());
                x.setSize(d);
            }
            if (x.isEmpty())return;
            {
                MATH21_ASSERT(x.getDeviceType() == y.getDeviceType());
                MATH21_ASSERT(math21_broadcast_is_compatible_to(d, x.shape()));
                MATH21_ASSERT(math21_broadcast_is_compatible_to(mask.shape(), x.shape()));
                MATH21_ASSERT(x.dims() == y.dims());
            }

            NumN n = x.size();
            VecN dx1, dx2, dy, dmap1, dmap2, dmap3;
            const TenN &map1 = maps(1);
            const TenN &map2 = maps(2);
            const TenN &map3 = maps(3);
            auto &x1 = x;
            auto &x2 = mask;
            // todo: use cpu when opencl>=2
            dx1.setDeviceType(x.getDeviceType());
            dx2.setDeviceType(x.getDeviceType());
            dy.setDeviceType(y.getDeviceType());
            dmap1.setDeviceType(y.getDeviceType());
            dmap2.setDeviceType(y.getDeviceType());
            dmap3.setDeviceType(y.getDeviceType());
            x.shape(dx1);
            x2.shape(dx2);
            y.shape(dy);
            map1.shape(dmap1);
            map2.shape(dmap2);
            map3.shape(dmap3);
            if (fname == m21_fname_none) {
                if (x.is_cpu()) {
                    math21_generic_subtensor_like_set_or_get_using_mask_in_d3_cpu(
                            n, x1.getDataAddress(), x2.getDataAddress(), y.getDataAddress(), map1.getDataAddress(),
                            map2.getDataAddress(), map3.getDataAddress(),
                            x1.dims(), dx1.getDataAddress(), x2.dims(), dx2.getDataAddress(), y.dims(),
                            dy.getDataAddress(),
                            map1.dims(), dmap1.getDataAddress(), map2.dims(), dmap2.getDataAddress(), map3.dims(),
                            dmap3.getDataAddress(), isGet, x1.getSpace().type);
                } else {
                    math21_generic_subtensor_like_set_or_get_using_mask_in_d3_wrapper(
                            n, x1.getDataAddressWrapper(), (PtrNInWrapper) x2.getDataAddressWrapper(),
                            y.getDataAddressWrapper(), (PtrNInWrapper) map1.getDataAddressWrapper(),
                            (PtrNInWrapper) map2.getDataAddressWrapper(),
                            (PtrNInWrapper) map3.getDataAddressWrapper(),
                            x1.dims(), (PtrNInWrapper) dx1.getDataAddressWrapper(), x2.dims(),
                            (PtrNInWrapper) dx2.getDataAddressWrapper(), y.dims(),
                            (PtrNInWrapper) dy.getDataAddressWrapper(),
                            map1.dims(), (PtrNInWrapper) dmap1.getDataAddressWrapper(), map2.dims(),
                            (PtrNInWrapper) dmap2.getDataAddressWrapper(), map3.dims(),
                            (PtrNInWrapper) dmap3.getDataAddressWrapper(), isGet, x1.getSpace().type);
                }
            } else {
                if (x.is_cpu()) {
                    math21_generic_subtensor_like_f_set_or_get_using_mask_in_d3_cpu(
                            fname,
                            n, x1.getDataAddress(), x2.getDataAddress(), y.getDataAddress(), map1.getDataAddress(),
                            map2.getDataAddress(), map3.getDataAddress(),
                            x1.dims(), dx1.getDataAddress(), x2.dims(), dx2.getDataAddress(), y.dims(),
                            dy.getDataAddress(),
                            map1.dims(), dmap1.getDataAddress(), map2.dims(), dmap2.getDataAddress(), map3.dims(),
                            dmap3.getDataAddress(), isGet, x1.getSpace().type);
                } else {
                    math21_generic_subtensor_like_f_set_or_get_using_mask_in_d3_wrapper(
                            fname,
                            n, x1.getDataAddressWrapper(), (PtrNInWrapper) x2.getDataAddressWrapper(),
                            y.getDataAddressWrapper(), (PtrNInWrapper) map1.getDataAddressWrapper(),
                            (PtrNInWrapper) map2.getDataAddressWrapper(),
                            (PtrNInWrapper) map3.getDataAddressWrapper(),
                            x1.dims(), (PtrNInWrapper) dx1.getDataAddressWrapper(), x2.dims(),
                            (PtrNInWrapper) dx2.getDataAddressWrapper(), y.dims(),
                            (PtrNInWrapper) dy.getDataAddressWrapper(),
                            map1.dims(), (PtrNInWrapper) dmap1.getDataAddressWrapper(), map2.dims(),
                            (PtrNInWrapper) dmap2.getDataAddressWrapper(), map3.dims(),
                            (PtrNInWrapper) dmap3.getDataAddressWrapper(), isGet, x1.getSpace().type);
                }
            }
        }
    }
}