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

#include "template_cpu_01_set.h"
#include "op_cpu_01_set.h"

using namespace math21;

void math21_generic_tensor_subregion_set_or_get_cpu(NumN n, void *x, void *y, NumN dims,
                                                    const NumN *dx, const NumN *dy,
                                                    const NumN *offset, NumB isGet, NumN type) {
    if (type == m21_type_NumR) {
        math21_template_tensor_subregion_set_or_get_cpu(n, (NumR *) x, (NumR *) y, dims, dx, dy, offset, isGet);
    } else if (type == m21_type_NumR32) {
        math21_template_tensor_subregion_set_or_get_cpu(n, (NumR32 *) x, (NumR32 *) y, dims, dx, dy, offset, isGet);
    } else if (type == m21_type_NumR64) {
        math21_template_tensor_subregion_set_or_get_cpu(n, (NumR64 *) x, (NumR64 *) y, dims, dx, dy, offset, isGet);
    } else {
        math21_tool_assert(0);
    }
}

void math21_generic_matrix_set_by_matrix_cpu(NumN d1, NumN d2,
                                             const void *x, NumN d1_x, NumN d2_x, NumN stride1_x, NumN stride2_x,
                                             void *y, NumN d1_y, NumN d2_y, NumN stride1_y, NumN stride2_y,
                                             NumN offset_x, NumN offset_y, NumN type) {
    if (type == m21_type_NumR) {
        math21_template_matrix_set_by_matrix_cpu(d1, d2,
                                                 (const NumR *) x, d1_x, d2_x, stride1_x, stride2_x,
                                                 (NumR *) y, d1_y, d2_y, stride1_y, stride2_y,
                                                 offset_x, offset_y);
    } else if (type == m21_type_NumR32) {
        math21_template_matrix_set_by_matrix_cpu(d1, d2,
                                                 (const NumR32 *) x, d1_x, d2_x, stride1_x, stride2_x,
                                                 (NumR32 *) y, d1_y, d2_y, stride1_y, stride2_y,
                                                 offset_x, offset_y);
    } else if (type == m21_type_NumR64) {
        math21_template_matrix_set_by_matrix_cpu(d1, d2,
                                                 (const NumR64 *) x, d1_x, d2_x, stride1_x, stride2_x,
                                                 (NumR64 *) y, d1_y, d2_y, stride1_y, stride2_y,
                                                 offset_x, offset_y);
    } else {
        math21_tool_assert(0);
    }
}

void math21_generic_tensor_3d_set_by_tensor_3d_cpu(NumN d1, NumN d2, NumN d3,
                                                   const void *x, NumN d1_x, NumN d2_x, NumN d3_x,
                                                   NumN stride1_x, NumN stride2_x, NumN stride3_x,
                                                   void *y, NumN d1_y, NumN d2_y, NumN d3_y,
                                                   NumN stride1_y, NumN stride2_y, NumN stride3_y,
                                                   NumN offset_x, NumN offset_y, NumN type) {
    if (type == m21_type_NumR) {
        math21_template_tensor_3d_set_by_tensor_3d_cpu(d1, d2, d3,
                                                       (const NumR *) x, d1_x, d2_x, d3_x,
                                                       stride1_x, stride2_x, stride3_x,
                                                       (NumR *) y, d1_y, d2_y, d3_y,
                                                       stride1_y, stride2_y, stride3_y,
                                                       offset_x, offset_y);
    } else if (type == m21_type_NumR32) {
        math21_template_tensor_3d_set_by_tensor_3d_cpu(d1, d2, d3,
                                                       (const NumR32 *) x, d1_x, d2_x, d3_x,
                                                       stride1_x, stride2_x, stride3_x,
                                                       (NumR32 *) y, d1_y, d2_y, d3_y,
                                                       stride1_y, stride2_y, stride3_y,
                                                       offset_x, offset_y);
    } else if (type == m21_type_NumR64) {
        math21_template_tensor_3d_set_by_tensor_3d_cpu(d1, d2, d3,
                                                       (const NumR64 *) x, d1_x, d2_x, d3_x,
                                                       stride1_x, stride2_x, stride3_x,
                                                       (NumR64 *) y, d1_y, d2_y, d3_y,
                                                       stride1_y, stride2_y, stride3_y,
                                                       offset_x, offset_y);
    } else {
        math21_tool_assert(0);
    }
}

void math21_generic_vector_set_by_value_cpu(NumN n, NumR value, void *x, NumN stride_x, NumN type) {
    if (type == m21_type_NumN8) {
        math21_template_vector_set_by_value_cpu(n, (NumN8) value, (NumN8 *) x, stride_x);
    } else if (type == m21_type_NumN) {
        math21_template_vector_set_by_value_cpu(n, (NumN) value, (NumN *) x, stride_x);
    } else if (type == m21_type_NumR) {
        math21_template_vector_set_by_value_cpu(n, (NumR) value, (NumR *) x, stride_x);
    } else if (type == m21_type_NumR32) {
        math21_template_vector_set_by_value_cpu(n, (NumR32) value, (NumR32 *) x, stride_x);
    } else if (type == m21_type_NumR64) {
        math21_template_vector_set_by_value_cpu(n, (NumR64) value, (NumR64 *) x, stride_x);
    } else {
        math21_tool_assert(0);
    }
}
