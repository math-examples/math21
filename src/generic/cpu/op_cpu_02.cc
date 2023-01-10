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

#include "template_cpu_02.h"
#include "op_cpu.h"

using namespace math21;

void math21_generic_tensor_f_shrink_cpu(NumN fname, NumN n, const void *x, void *y,
                                        NumN dims_x, const NumN *dx, NumN dims_y, const NumN *dy,
                                        NumN nb, const NumN *b,
                                        NumN nv, NumN dims_v, const NumN *dv, NumN type) {
    if (type == m21_type_NumN) {
        math21_template_tensor_f_shrink_cpu(fname, n, (const NumN *) x, (NumN *) y,
                                            dims_x, dx, dims_y, dy, nb, b, nv, dims_v, dv);
    } else if (type == m21_type_NumR) {
        math21_template_tensor_f_shrink_cpu(fname, n, (const NumR *) x, (NumR *) y,
                                            dims_x, dx, dims_y, dy, nb, b, nv, dims_v, dv);
    } else if (type == m21_type_NumR32) {
        math21_template_tensor_f_shrink_cpu(fname, n, (const NumR32 *) x, (NumR32 *) y,
                                            dims_x, dx, dims_y, dy, nb, b, nv, dims_v, dv);
    } else if (type == m21_type_NumR64) {
        math21_template_tensor_f_shrink_cpu(fname, n, (const NumR64 *) x, (NumR64 *) y,
                                            dims_x, dx, dims_y, dy, nb, b, nv, dims_v, dv);
    } else {
        math21_tool_assert(0);
    }
}

void math21_generic_tensor_f_inner_product_like_shrink_cpu(
        NumN fname, NumN n,
        const void *x1, const void *x2, void *y,
        NumN dims_x, const NumN *dx, NumN dims_y, const NumN *dy,
        NumN nb, const NumN *b,
        NumN nv, NumN dims_v, const NumN *dv, NumN type) {
    if (type == m21_type_NumR) {
        math21_template_tensor_f_inner_product_like_shrink_cpu(
                fname, n, (const NumR *) x1, (const NumR *) x2, (NumR *) y,
                dims_x, dx, dims_y, dy, nb, b, nv, dims_v, dv);
    } else if (type == m21_type_NumR32) {
        math21_template_tensor_f_inner_product_like_shrink_cpu(
                fname, n, (const NumR32 *) x1, (const NumR32 *) x2, (NumR32 *) y,
                dims_x, dx, dims_y, dy, nb, b, nv, dims_v, dv);
    } else if (type == m21_type_NumR64) {
        math21_template_tensor_f_inner_product_like_shrink_cpu(
                fname, n, (const NumR64 *) x1, (const NumR64 *) x2, (NumR64 *) y,
                dims_x, dx, dims_y, dy, nb, b, nv, dims_v, dv);
    } else {
        math21_tool_assert(0);
    }
}

void math21_generic_tensor_f_inner_product_like_bcshrink_cpu(
        NumN fname, NumN n,
        const void *x1, const void *x2, void *y,
        NumN dims_x1, const NumN *dx1, NumN dims_x2, const NumN *dx2,
        NumN dims_x, const NumN *dx, NumN dims_y, const NumN *dy,
        NumN nb, const NumN *b,
        NumN nv, NumN dims_v, const NumN *dv, NumN type) {
    if (type == m21_type_NumR) {
        math21_template_tensor_f_inner_product_like_bcshrink_cpu(
                fname, n, (const NumR *) x1, (const NumR *) x2, (NumR *) y,
                dims_x1, dx1, dims_x2, dx2,
                dims_x, dx, dims_y, dy, nb, b, nv, dims_v, dv);
    } else if (type == m21_type_NumR32) {
        math21_template_tensor_f_inner_product_like_bcshrink_cpu(
                fname, n, (const NumR32 *) x1, (const NumR32 *) x2, (NumR32 *) y,
                dims_x1, dx1, dims_x2, dx2,
                dims_x, dx, dims_y, dy, nb, b, nv, dims_v, dv);
    } else if (type == m21_type_NumR64) {
        math21_template_tensor_f_inner_product_like_bcshrink_cpu(
                fname, n, (const NumR64 *) x1, (const NumR64 *) x2, (NumR64 *) y,
                dims_x1, dx1, dims_x2, dx2,
                dims_x, dx, dims_y, dy, nb, b, nv, dims_v, dv);
    } else {
        math21_tool_assert(0);
    }
}

void math21_generic_tensor_f_with_broadcast_in_dn_cpu(NumN fname, NumN n,
                                                      const void *x1,
                                                      const void *x2,
                                                      void *y,
                                                      NumN dims_x1, const NumN *dx1,
                                                      NumN dims_x2, const NumN *dx2,
                                                      NumN dims_y, const NumN *dy, NumN type) {
    if (type == m21_type_NumN8) {
        math21_template_tensor_f_with_broadcast_in_dn_cpu(fname, n,
                                                          (const NumN8 *) x1, (const NumN8 *) x2,
                                                          (NumN8 *) y, dims_x1, dx1, dims_x2, dx2,
                                                          dims_y, dy);
    } else if (type == m21_type_NumN) {
        math21_template_tensor_f_with_broadcast_in_dn_cpu(fname, n,
                                                          (const NumN *) x1, (const NumN *) x2,
                                                          (NumN *) y, dims_x1, dx1, dims_x2, dx2,
                                                          dims_y, dy);
    } else if (type == m21_type_NumR) {
        math21_template_tensor_f_with_broadcast_in_dn_cpu(fname, n,
                                                          (const NumR *) x1, (const NumR *) x2,
                                                          (NumR *) y, dims_x1, dx1, dims_x2, dx2,
                                                          dims_y, dy);
    } else if (type == m21_type_NumR32) {
        math21_template_tensor_f_with_broadcast_in_dn_cpu(fname, n,
                                                          (const NumR32 *) x1, (const NumR32 *) x2,
                                                          (NumR32 *) y, dims_x1, dx1, dims_x2, dx2,
                                                          dims_y, dy);
    } else if (type == m21_type_NumR64) {
        math21_template_tensor_f_with_broadcast_in_dn_cpu(fname, n,
                                                          (const NumR64 *) x1, (const NumR64 *) x2,
                                                          (NumR64 *) y, dims_x1, dx1, dims_x2, dx2,
                                                          dims_y, dy);
    } else {
        math21_tool_assert(0);
    }
}

void math21_generic_vector_f_add_like_cpu(NumN fname, NumN n,
                                          const void *x1,
                                          const void *x2,
                                          void *y, NumN type) {
    if (type == m21_type_NumN8) {
        math21_template_vector_f_add_like_cpu(fname, n,
                                              (const NumN8 *) x1, (const NumN8 *) x2,
                                              (NumN8 *) y);
    } else if (type == m21_type_NumN) {
        math21_template_vector_f_add_like_cpu(fname, n,
                                              (const NumN *) x1, (const NumN *) x2,
                                              (NumN *) y);
    } else if (type == m21_type_NumR) {
        math21_template_vector_f_add_like_cpu(fname, n,
                                              (const NumR *) x1, (const NumR *) x2,
                                              (NumR *) y);
    } else if (type == m21_type_NumR32) {
        math21_template_vector_f_add_like_cpu(fname, n,
                                              (const NumR32 *) x1, (const NumR32 *) x2,
                                              (NumR32 *) y);
    } else if (type == m21_type_NumR64) {
        math21_template_vector_f_add_like_cpu(fname, n,
                                              (const NumR64 *) x1, (const NumR64 *) x2,
                                              (NumR64 *) y);
    } else {
        math21_tool_assert(0);
    }
}

void math21_generic_vector_f_sin_like_cpu(NumN fname, NumN n,
                                          const void *x1,
                                          void *y, NumN type) {
    if (type == m21_type_NumR) {
        math21_template_vector_f_sin_like_cpu(fname, n,
                                              (const NumR *) x1,
                                              (NumR *) y);
    } else if (type == m21_type_NumR32) {
        math21_template_vector_f_sin_like_cpu(fname, n,
                                              (const NumR32 *) x1,
                                              (NumR32 *) y);
    } else if (type == m21_type_NumR64) {
        math21_template_vector_f_sin_like_cpu(fname, n,
                                              (const NumR64 *) x1,
                                              (NumR64 *) y);
    } else {
        math21_tool_assert(0);
    }
}

void math21_generic_vector_f_kx_like_cpu(NumN fname, NumN n,
                                         NumR k,
                                         const void *x1,
                                         void *y, NumN type) {
    if (type == m21_type_NumR) {
        math21_template_vector_f_kx_like_cpu(fname, n, (NumR) k,
                                             (const NumR *) x1,
                                             (NumR *) y);
    } else if (type == m21_type_NumR32) {
        math21_template_vector_f_kx_like_cpu(fname, n, (NumR32) k,
                                             (const NumR32 *) x1,
                                             (NumR32 *) y);
    } else if (type == m21_type_NumR64) {
        math21_template_vector_f_kx_like_cpu(fname, n, (NumR64) k,
                                             (const NumR64 *) x1,
                                             (NumR64 *) y);
    } else {
        math21_tool_assert(0);
    }
}