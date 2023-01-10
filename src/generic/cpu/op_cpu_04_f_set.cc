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

#include "template_cpu_04_f_set.h"
#include "op_cpu_04_f_set.h"

using namespace math21;

// Do not edit.
// file generated from set file.

void math21_generic_subtensor_like_f_set_or_get_using_mask_in_d3_cpu(
        NumN fname,
        NumN n, void *x1, const NumN *x2, void *y,
        const NumN *map1, const NumN *map2, const NumN *map3,
        NumN dims_x1, const NumN *dx1, NumN dims_x2, const NumN *dx2,
        NumN dims_y, const NumN *dy, NumN dims_map1, const NumN *dmap1,
        NumN dims_map2, const NumN *dmap2, NumN dims_map3, const NumN *dmap3,
        NumB isGet, NumN type) {
    if (type == m21_type_NumN8) {
        math21_template_subtensor_like_f_set_or_get_using_mask_in_d3_cpu(fname,
                                                                         n, (NumN8 *) x1, x2, (NumN8 *) y, map1, map2,
                                                                         map3,
                                                                         dims_x1, dx1, dims_x2, dx2, dims_y, dy,
                                                                         dims_map1, dmap1, dims_map2, dmap2, dims_map3,
                                                                         dmap3, isGet);
    } else if (type == m21_type_NumR) {
        math21_template_subtensor_like_f_set_or_get_using_mask_in_d3_cpu(fname,
                                                                         n, (NumR *) x1, x2, (NumR *) y, map1, map2,
                                                                         map3,
                                                                         dims_x1, dx1, dims_x2, dx2, dims_y, dy,
                                                                         dims_map1, dmap1, dims_map2, dmap2, dims_map3,
                                                                         dmap3, isGet);
    } else if (type == m21_type_NumR32) {
        math21_template_subtensor_like_f_set_or_get_using_mask_in_d3_cpu(fname,
                                                                         n, (NumR32 *) x1, x2, (NumR32 *) y, map1, map2,
                                                                         map3,
                                                                         dims_x1, dx1, dims_x2, dx2, dims_y, dy,
                                                                         dims_map1, dmap1, dims_map2, dmap2, dims_map3,
                                                                         dmap3, isGet);
    } else if (type == m21_type_NumR64) {
        math21_template_subtensor_like_f_set_or_get_using_mask_in_d3_cpu(fname,
                                                                         n, (NumR64 *) x1, x2, (NumR64 *) y, map1, map2,
                                                                         map3,
                                                                         dims_x1, dx1, dims_x2, dx2, dims_y, dy,
                                                                         dims_map1, dmap1, dims_map2, dmap2, dims_map3,
                                                                         dmap3, isGet);
    } else {
        math21_tool_assert(0);
    }
}
