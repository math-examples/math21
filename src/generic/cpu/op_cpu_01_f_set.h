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

void math21_generic_tensor_subregion_f_set_or_get_cpu(NumN fname, NumN n, void *x, void *y, NumN dims,
                                                      const NumN *dx, const NumN *dy,
                                                      const NumN *offset, NumB isGet, NumN type);

void math21_generic_matrix_f_set_by_matrix_cpu(NumN fname, NumN d1, NumN d2,
                                               const void *x, NumN d1_x, NumN d2_x, NumN stride1_x, NumN stride2_x,
                                               void *y, NumN d1_y, NumN d2_y, NumN stride1_y, NumN stride2_y,
                                               NumN offset_x, NumN offset_y, NumN type);

void math21_generic_tensor_3d_f_set_by_tensor_3d_cpu(NumN fname, NumN d1, NumN d2, NumN d3,
                                                     const void *x, NumN d1_x, NumN d2_x, NumN d3_x,
                                                     NumN stride1_x, NumN stride2_x, NumN stride3_x,
                                                     void *y, NumN d1_y, NumN d2_y, NumN d3_y,
                                                     NumN stride1_y, NumN stride2_y, NumN stride3_y,
                                                     NumN offset_x, NumN offset_y, NumN type);

void math21_generic_vector_f_set_by_value_cpu(NumN fname, NumN n, NumR value, void *x, NumN stride_x, NumN type);

#ifdef __cplusplus
}
#endif
