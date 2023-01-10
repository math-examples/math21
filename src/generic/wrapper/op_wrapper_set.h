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

#include "inner_c.h"

#ifdef __cplusplus
extern "C" {
#endif

void math21_generic_tensor_subregion_set_or_get_wrapper(NumN n, PtrVoidWrapper x, PtrVoidWrapper y, NumN dims,
                                                        PtrNInWrapper dx, PtrNInWrapper dy,
                                                        PtrNInWrapper offset, NumB isGet, NumN type);

void math21_generic_matrix_set_by_matrix_wrapper(NumN d1, NumN d2,
                                                 PtrVoidInWrapper x, NumN d1_x, NumN d2_x, NumN stride1_x,
                                                 NumN stride2_x,
                                                 PtrVoidWrapper y, NumN d1_y, NumN d2_y, NumN stride1_y,
                                                 NumN stride2_y,
                                                 NumN offset_x, NumN offset_y, NumN type);

void math21_generic_tensor_3d_set_by_tensor_3d_wrapper(NumN d1, NumN d2, NumN d3,
                                                       PtrVoidInWrapper x, NumN d1_x, NumN d2_x, NumN d3_x,
                                                       NumN stride1_x, NumN stride2_x, NumN stride3_x,
                                                       PtrVoidWrapper y, NumN d1_y, NumN d2_y, NumN d3_y,
                                                       NumN stride1_y, NumN stride2_y, NumN stride3_y,
                                                       NumN offset_x, NumN offset_y, NumN type);

void math21_generic_vector_set_by_value_wrapper(
        NumN n, NumR value, PtrVoidWrapper x, NumN stride_x, NumN type);

void math21_generic_vector_set_by_vector_wrapper(
        NumN n, PtrVoidInWrapper x, NumN stride_x, PtrVoidWrapper y,
        NumN stride_y, NumN offset_x, NumN offset_y, NumN type1, NumN type2);

void math21_generic_subtensor_like_set_or_get_using_mask_in_d3_wrapper(
        NumN n, PtrVoidWrapper x1, PtrNInWrapper x2, PtrVoidWrapper y,
        PtrNInWrapper map1, PtrNInWrapper map2, PtrNInWrapper map3,
        NumN dims_x1, PtrNInWrapper dx1, NumN dims_x2, PtrNInWrapper dx2,
        NumN dims_y, PtrNInWrapper dy,
        NumN dims_map1, PtrNInWrapper dmap1,
        NumN dims_map2, PtrNInWrapper dmap2,
        NumN dims_map3, PtrNInWrapper dmap3,
        NumB isGet, NumN type);

#ifdef __cplusplus
}
#endif
