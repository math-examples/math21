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

#include "template_cuda_01.h"
#include "op_cuda.h"

using namespace math21;

// Do not edit.
// file generated by replacing "_cpu(" in cpu file to "_cuda(".

void math21_generic_vector_kx_cuda(NumN n, NumR k, void *x, NumN stride_x, NumN type) {
    if (type == m21_type_NumR) {
        math21_template_vector_kx_cuda(n, (NumR) k, (NumR *) x, stride_x);
    } else if (type == m21_type_NumR32) {
        math21_template_vector_kx_cuda(n, (NumR32) k, (NumR32 *) x, stride_x);
    } else if (type == m21_type_NumR64) {
        math21_template_vector_kx_cuda(n, (NumR64) k, (NumR64 *) x, stride_x);
    } else {
        math21_tool_assert(0);
    }
}

void math21_generic_vector_kx_add_y_cuda(NumN n, NumR k, const void *x, NumN stride_x,
                                         void *y, NumN stride_y, NumN type) {
    if (type == m21_type_NumR) {
        math21_template_vector_kx_add_y_cuda(n, (NumR) k, (const NumR *) x, stride_x, (NumR *) y, stride_y);
    } else if (type == m21_type_NumR32) {
        math21_template_vector_kx_add_y_cuda(n, (NumR32) k, (const NumR32 *) x, stride_x, (NumR32 *) y, stride_y);
    } else if (type == m21_type_NumR64) {
        math21_template_vector_kx_add_y_cuda(n, (NumR64) k, (const NumR64 *) x, stride_x, (NumR64 *) y, stride_y);
    } else {
        math21_tool_assert(0);
    }
}

void math21_generic_vector_xy_cuda(NumN n, const void *x, NumN stride_x,
                                   void *y, NumN stride_y, NumN type) {
    if (type == m21_type_NumR) {
        math21_template_vector_xy_cuda(n, (const NumR *) x, stride_x, (NumR *) y, stride_y);
    } else if (type == m21_type_NumR32) {
        math21_template_vector_xy_cuda(n, (const NumR32 *) x, stride_x, (NumR32 *) y, stride_y);
    } else if (type == m21_type_NumR64) {
        math21_template_vector_xy_cuda(n, (const NumR64 *) x, stride_x, (NumR64 *) y, stride_y);
    } else {
        math21_tool_assert(0);
    }
}

void math21_generic_vector_sin_cuda(NumN n, const void *x, void *y, NumN type) {
    if (type == m21_type_NumR) {
        math21_template_vector_sin_cuda(n, (const NumR *) x, (NumR *) y);
    } else if (type == m21_type_NumR32) {
        math21_template_vector_sin_cuda(n, (const NumR32 *) x, (NumR32 *) y);
    } else if (type == m21_type_NumR64) {
        math21_template_vector_sin_cuda(n, (const NumR64 *) x, (NumR64 *) y);
    } else {
        math21_tool_assert(0);
    }
}

void math21_generic_vector_cos_cuda(NumN n, const void *x, void *y, NumN type) {
    if (type == m21_type_NumR) {
        math21_template_vector_cos_cuda(n, (const NumR *) x, (NumR *) y);
    } else if (type == m21_type_NumR32) {
        math21_template_vector_cos_cuda(n, (const NumR32 *) x, (NumR32 *) y);
    } else if (type == m21_type_NumR64) {
        math21_template_vector_cos_cuda(n, (const NumR64 *) x, (NumR64 *) y);
    } else {
        math21_tool_assert(0);
    }
}

void math21_generic_tensor_3d_swap_row_in_d2_cuda(
        NumN n, void *x, NumN i, NumN j, NumN d1, NumN d2, NumN d3, NumN type) {
    if (type == m21_type_NumN) {
        math21_template_tensor_3d_swap_row_in_d2_cuda(n, (NumN *) x, i, j, d1, d2, d3);
    } else if (type == m21_type_NumR) {
        math21_template_tensor_3d_swap_row_in_d2_cuda(n, (NumR *) x, i, j, d1, d2, d3);
    } else if (type == m21_type_NumR32) {
        math21_template_tensor_3d_swap_row_in_d2_cuda(n, (NumR32 *) x, i, j, d1, d2, d3);
    } else if (type == m21_type_NumR64) {
        math21_template_tensor_3d_swap_row_in_d2_cuda(n, (NumR64 *) x, i, j, d1, d2, d3);
    } else {
        math21_tool_assert(0);
    }
}

void math21_generic_vector_addToC_cuda(NumN n, const void *A, const void *B, void *C, NumN type) {
    if (type == m21_type_NumN) {
        math21_template_vector_addToC_cuda(n, (const NumN *) A, (const NumN *) B, (NumN *) C);
    } else if (type == m21_type_NumR) {
        math21_template_vector_addToC_cuda(n, (const NumR *) A, (const NumR *) B, (NumR *) C);
    } else if (type == m21_type_NumR32) {
        math21_template_vector_addToC_cuda(n, (const NumR32 *) A, (const NumR32 *) B, (NumR32 *) C);
    } else if (type == m21_type_NumR64) {
        math21_template_vector_addToC_cuda(n, (const NumR64 *) A, (const NumR64 *) B, (NumR64 *) C);
    } else {
        math21_tool_assert(0);
    }
}

void math21_generic_vector_mulToC_cuda(NumN n, const void *A, const void *B, void *C, NumN type) {
    if (type == m21_type_NumR) {
        math21_template_vector_mulToC_cuda(n, (const NumR *) A, (const NumR *) B, (NumR *) C);
    } else if (type == m21_type_NumR32) {
        math21_template_vector_mulToC_cuda(n, (const NumR32 *) A, (const NumR32 *) B, (NumR32 *) C);
    } else if (type == m21_type_NumR64) {
        math21_template_vector_mulToC_cuda(n, (const NumR64 *) A, (const NumR64 *) B, (NumR64 *) C);
    } else {
        math21_tool_assert(0);
    }
}

void math21_generic_broadcast_in_dn_cuda(NumN n, const void *x, void *y,
                                         NumN dims_x, const NumN *dx,
                                         NumN dims_y, const NumN *dy,
                                         NumN type) {
    if (type == m21_type_NumR) {
        math21_template_vector_broadcast_in_dn_cuda(n, (const NumR *) x, (NumR *) y, dims_x, dx, dims_y, dy);
    } else if (type == m21_type_NumR32) {
        math21_template_vector_broadcast_in_dn_cuda(n, (const NumR32 *) x, (NumR32 *) y, dims_x, dx, dims_y, dy);
    } else if (type == m21_type_NumR64) {
        math21_template_vector_broadcast_in_dn_cuda(n, (const NumR64 *) x, (NumR64 *) y, dims_x, dx, dims_y, dy);
    } else {
        math21_tool_assert(0);
    }
}

void math21_generic_optimization_adam_update_part_2_cuda(
        NumN x_size, void *x, const void *m, const void *v,
        NumR beta1, NumR beta2, NumR alpha, NumR eps, NumN t, NumN type) {
    if (type == m21_type_NumR) {
        math21_template_optimization_adam_update_part_2_cuda(
                x_size, (NumR *) x, (const NumR *) m, (const NumR *) v, (NumR) beta1, (NumR) beta2,
                (NumR) alpha, (NumR) eps, t);
    } else if (type == m21_type_NumR32) {
        math21_template_optimization_adam_update_part_2_cuda(
                x_size, (NumR32 *) x, (const NumR32 *) m, (const NumR32 *) v, (NumR32) beta1, (NumR32) beta2,
                (NumR32) alpha, (NumR32) eps, t);
    } else if (type == m21_type_NumR64) {
        math21_template_optimization_adam_update_part_2_cuda(
                x_size, (NumR64 *) x, (const NumR64 *) m, (const NumR64 *) v, (NumR64) beta1, (NumR64) beta2,
                (NumR64) alpha, (NumR64) eps, t);
    } else {
        math21_tool_assert(0);
    }
}