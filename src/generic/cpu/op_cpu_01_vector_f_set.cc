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

#include "template_cpu_01_vector_f_set.h"
#include "op_cpu_01_vector_f_set.h"

using namespace math21;

// Do not edit.
// file generated from set file.

void math21_generic_vector_f_set_by_vector_cpu(
        NumN fname,
        NumN n, const void *x, NumN stride_x, void *y, NumN stride_y,
        NumN offset_x, NumN offset_y, NumN type1, NumN type2) {
    if (type1 == m21_type_NumN8) {
        if (type2 == m21_type_NumN8) {
            math21_template_vector_f_set_by_vector_cpu(fname, n, (const NumN8 *) x, stride_x, (NumN8 *) y, stride_y,
                                                       offset_x, offset_y);
        } else if (type2 == m21_type_NumR) {
            math21_template_vector_f_set_by_vector_cpu(fname, n, (const NumN8 *) x, stride_x, (NumR *) y, stride_y,
                                                       offset_x, offset_y);
        } else {
            math21_tool_assert(0);
        }
    } else if (type1 == m21_type_NumN) {
        if (type2 == m21_type_NumN8) {
            math21_template_vector_f_set_by_vector_cpu(fname, n, (const NumN *) x, stride_x, (NumN8 *) y, stride_y,
                                                       offset_x, offset_y);
        } else if (type2 == m21_type_NumN) {
            math21_template_vector_f_set_by_vector_cpu(fname, n, (const NumN *) x, stride_x, (NumN *) y, stride_y,
                                                       offset_x, offset_y);
        } else if (type2 == m21_type_NumR) {
            math21_template_vector_f_set_by_vector_cpu(fname, n, (const NumN *) x, stride_x, (NumR *) y, stride_y,
                                                       offset_x, offset_y);
        } else {
            math21_tool_assert(0);
        }
    } else if (type1 == m21_type_NumR) {
        if (type2 == m21_type_NumN8) {
            math21_template_vector_f_set_by_vector_cpu(fname, n, (const NumR *) x, stride_x, (NumN8 *) y, stride_y,
                                                       offset_x, offset_y);
        } else if (type2 == m21_type_NumN) {
            math21_template_vector_f_set_by_vector_cpu(fname, n, (const NumR *) x, stride_x, (NumN *) y, stride_y,
                                                       offset_x, offset_y);
        } else if (type2 == m21_type_NumR) {
            math21_template_vector_f_set_by_vector_cpu(fname, n, (const NumR *) x, stride_x, (NumR *) y, stride_y,
                                                       offset_x, offset_y);
        } else {
            math21_tool_assert(0);
        }
    } else {
        math21_tool_assert(0);
    }
}
