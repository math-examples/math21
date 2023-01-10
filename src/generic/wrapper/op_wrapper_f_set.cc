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

#include "../cpu/files_c.h"
#include "../cuda/files_c.h"
#include "../opencl/files_c.h"
#include "op_wrapper_f_set.h"

void
math21_generic_tensor_subregion_f_set_or_get_wrapper(NumN fname, NumN n, PtrVoidWrapper x, PtrVoidWrapper y, NumN dims,
                                                     PtrNInWrapper dx, PtrNInWrapper dy,
                                                     PtrNInWrapper offset, NumB isGet, NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_tensor_subregion_f_set_or_get_cpu(fname, n, x, y, dims, dx, dy, offset, isGet, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_tensor_subregion_f_set_or_get_cuda(fname, n, x, y, dims, dx, dy, offset, isGet, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_tensor_subregion_f_set_or_get_opencl(fname, n, x, y, dims, dx, dy, offset, isGet, type);
#endif
}

void math21_generic_matrix_f_set_by_matrix_wrapper(NumN fname, NumN d1, NumN d2,
                                                   PtrVoidInWrapper x, NumN d1_x, NumN d2_x, NumN stride1_x,
                                                   NumN stride2_x,
                                                   PtrVoidWrapper y, NumN d1_y, NumN d2_y, NumN stride1_y,
                                                   NumN stride2_y,
                                                   NumN offset_x, NumN offset_y, NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_matrix_f_set_by_matrix_cpu(fname, d1, d2,
                                            x, d1_x, d2_x, stride1_x, stride2_x,
                                            y, d1_y, d2_y, stride1_y, stride2_y,
                                            offset_x, offset_y, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_matrix_f_set_by_matrix_cuda(fname, d1, d2,
                                             x, d1_x, d2_x, stride1_x, stride2_x,
                                             y, d1_y, d2_y, stride1_y, stride2_y,
                                             offset_x, offset_y, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_matrix_f_set_by_matrix_opencl(fname, d1, d2,
                                                 x, d1_x, d2_x, stride1_x, stride2_x,
                                                 y, d1_y, d2_y, stride1_y, stride2_y,
                                                 offset_x, offset_y, type);
#endif
}

void math21_generic_tensor_3d_f_set_by_tensor_3d_wrapper(NumN fname, NumN d1, NumN d2, NumN d3,
                                                         PtrVoidInWrapper x, NumN d1_x, NumN d2_x, NumN d3_x,
                                                         NumN stride1_x, NumN stride2_x, NumN stride3_x,
                                                         PtrVoidWrapper y, NumN d1_y, NumN d2_y, NumN d3_y,
                                                         NumN stride1_y, NumN stride2_y, NumN stride3_y,
                                                         NumN offset_x, NumN offset_y, NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_tensor_3d_f_set_by_tensor_3d_cpu(fname, d1, d2, d3,
                                                  x, d1_x, d2_x, d3_x, stride1_x, stride2_x, stride3_x,
                                                  y, d1_y, d2_y, d3_y, stride1_y, stride2_y, stride3_y,
                                                  offset_x, offset_y, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_tensor_3d_f_set_by_tensor_3d_cuda(fname, d1, d2, d3,
                                                   x, d1_x, d2_x, d3_x, stride1_x, stride2_x, stride3_x,
                                                   y, d1_y, d2_y, d3_y, stride1_y, stride2_y, stride3_y,
                                                   offset_x, offset_y, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_tensor_3d_f_set_by_tensor_3d_opencl(fname, d1, d2, d3,
                                                       x, d1_x, d2_x, d3_x, stride1_x, stride2_x, stride3_x,
                                                       y, d1_y, d2_y, d3_y, stride1_y, stride2_y, stride3_y,
                                                       offset_x, offset_y, type);
#endif
}

void math21_generic_vector_f_set_by_value_wrapper(NumN fname,
                                                  NumN n, NumR value, PtrVoidWrapper x, NumN stride_x, NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_vector_f_set_by_value_cpu(fname, n, value, x, stride_x, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_vector_f_set_by_value_cuda(fname, n, value, x, stride_x, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_vector_f_set_by_value_opencl(fname, n, value, x, stride_x, type);
#endif
}

void math21_generic_vector_f_set_by_vector_wrapper(NumN fname,
                                                   NumN n, PtrVoidInWrapper x, NumN stride_x, PtrVoidWrapper y,
                                                   NumN stride_y, NumN offset_x, NumN offset_y, NumN type1,
                                                   NumN type2) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_vector_f_set_by_vector_cpu(fname, n, x, stride_x, y, stride_y, offset_x, offset_y, type1, type2);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_vector_f_set_by_vector_cuda(fname, n, x, stride_x, y, stride_y, offset_x, offset_y, type1, type2);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_vector_f_set_by_vector_opencl(fname, n, x, stride_x, y, stride_y, offset_x, offset_y, type1, type2);
#endif
}

void math21_generic_subtensor_like_f_set_or_get_using_mask_in_d3_wrapper(NumN fname,
                                                                         NumN n, PtrVoidWrapper x1, PtrNInWrapper x2,
                                                                         PtrVoidWrapper y,
                                                                         PtrNInWrapper map1, PtrNInWrapper map2,
                                                                         PtrNInWrapper map3,
                                                                         NumN dims_x1, PtrNInWrapper dx1, NumN dims_x2,
                                                                         PtrNInWrapper dx2,
                                                                         NumN dims_y, PtrNInWrapper dy,
                                                                         NumN dims_map1, PtrNInWrapper dmap1,
                                                                         NumN dims_map2, PtrNInWrapper dmap2,
                                                                         NumN dims_map3, PtrNInWrapper dmap3,
                                                                         NumB isGet, NumN type) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_generic_subtensor_like_f_set_or_get_using_mask_in_d3_cpu(fname, 
            n, x1, x2, y, map1, map2, map3,
            dims_x1, dx1, dims_x2, dx2, dims_y, dy,
            dims_map1, dmap1, dims_map2, dmap2, dims_map3, dmap3, isGet, type);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_generic_subtensor_like_f_set_or_get_using_mask_in_d3_cuda(fname, 
            n, x1, x2, y, map1, map2, map3,
            dims_x1, dx1, dims_x2, dx2, dims_y, dy,
            dims_map1, dmap1, dims_map2, dmap2, dims_map3, dmap3, isGet, type);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_generic_subtensor_like_f_set_or_get_using_mask_in_d3_opencl(fname,
                                                                       n, x1, x2, y, map1, map2, map3,
                                                                       dims_x1, dx1, dims_x2, dx2, dims_y, dy,
                                                                       dims_map1, dmap1, dims_map2, dmap2, dims_map3,
                                                                       dmap3, isGet, type);
#endif
}