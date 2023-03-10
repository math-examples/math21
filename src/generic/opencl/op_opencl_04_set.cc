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

// Do not edit.
// file generated by replacing "_cpu(" in cpu file to "_opencl(".

#include "template_opencl_04_set.h"
#include "op_opencl_04_set.h"

#ifdef MATH21_FLAG_USE_OPENCL

using namespace math21;

void math21_generic_subtensor_like_set_or_get_using_mask_in_d3_opencl(
        NumN n, PtrVoidWrapper x1, PtrNInWrapper x2, PtrVoidWrapper y,
        PtrNInWrapper map1, PtrNInWrapper map2, PtrNInWrapper map3,
        NumN dims_x1, PtrNInWrapper dx1, NumN dims_x2, PtrNInWrapper dx2,
        NumN dims_y, PtrNInWrapper dy, NumN dims_map1, PtrNInWrapper dmap1,
        NumN dims_map2, PtrNInWrapper dmap2,
        NumN dims_map3, PtrNInWrapper dmap3, NumB isGet, NumN type) {
    if (type == m21_type_NumN8) {
        math21_template_subtensor_like_set_or_get_using_mask_in_d3_opencl<NumN8>(
                n, x1, x2, y, map1, map2, map3,
                dims_x1, dx1, dims_x2, dx2, dims_y, dy,
                dims_map1, dmap1, dims_map2, dmap2, dims_map3, dmap3, isGet);
    } else if (type == m21_type_NumR) {
        math21_template_subtensor_like_set_or_get_using_mask_in_d3_opencl<NumR>(
                n, x1, x2, y, map1, map2, map3,
                dims_x1, dx1, dims_x2, dx2, dims_y, dy,
                dims_map1, dmap1, dims_map2, dmap2, dims_map3, dmap3, isGet);
    } else if (type == m21_type_NumR32) {
        math21_template_subtensor_like_set_or_get_using_mask_in_d3_opencl<NumR32>(
                n, x1, x2, y, map1, map2, map3,
                dims_x1, dx1, dims_x2, dx2, dims_y, dy,
                dims_map1, dmap1, dims_map2, dmap2, dims_map3, dmap3, isGet);
    } else if (type == m21_type_NumR64) {
        math21_template_subtensor_like_set_or_get_using_mask_in_d3_opencl<NumR64>(
                n, x1, x2, y, map1, map2, map3,
                dims_x1, dx1, dims_x2, dx2, dims_y, dy,
                dims_map1, dmap1, dims_map2, dmap2, dims_map3, dmap3, isGet);
    } else {
        math21_tool_assert(0);
    }
}

#endif