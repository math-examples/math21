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

void math21_generic_subtensor_like_f_set_or_get_using_mask_in_d3_cpu(
        NumN fname,
        NumN n, void *x1, const NumN *x2, void *y,
        const NumN *map1, const NumN *map2, const NumN *map3,
        NumN dims_x1, const NumN *dx1, NumN dims_x2, const NumN *dx2,
        NumN dims_y, const NumN *dy, NumN dims_map1, const NumN *dmap1,
        NumN dims_map2, const NumN *dmap2, NumN dims_map3, const NumN *dmap3,
        NumB isGet, NumN type);

#ifdef __cplusplus
}
#endif
