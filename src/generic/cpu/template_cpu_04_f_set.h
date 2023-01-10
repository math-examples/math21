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
#include "common.h"

#define MATH21_IS_FROM_CPU

#include "../kernels/generic_04_f_set.kl"

#undef MATH21_IS_FROM_CPU

namespace math21 {

    // todo: use index 1 for x, y
    // x is sub-tensor of y
    template<typename T>
    void math21_template_subtensor_like_f_set_or_get_using_mask_in_d3_cpu(
            NumN fname, NumN n,
            T *x1,
            const NumN *x2,
            T *y,
            const NumN *map1,
            const NumN *map2,
            const NumN *map3,
            NumN dims_x1, const NumN *dx1,
            NumN dims_x2, const NumN *dx2,
            NumN dims_y, const NumN *dy,
            NumN dims_map1, const NumN *dmap1,
            NumN dims_map2, const NumN *dmap2,
            NumN dims_map3, const NumN *dmap3,
            NumB isGet) {
        x1 -= 1;
        x2 -= 1;
        y -= 1;
        map1 -= 1;
        map2 -= 1;
        map3 -= 1;
        dx1 -= 1;
        dx2 -= 1;
        dy -= 1;
        dmap1 -= 1;
        dmap2 -= 1;
        dmap3 -= 1;
        NumN id;
        auto f = math21_generic_fname_to_f_addto_like_name_cpu(fname);
#pragma omp parallel for
        for (id = 1; id <= n; ++id) {
            math21_template_subtensor_like_f_set_or_get_using_mask_in_d3_cpu_kernel(
                    f, n, x1, x2, y, map1, map2, map3,
                    dims_x1, dx1, dims_x2, dx2, dims_y, dy,
                    dims_map1, dmap1, dims_map2, dmap2, dims_map3, dmap3,
                    isGet, id);
        }
    }
}