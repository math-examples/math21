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

#include "inner_cc.h"
#include "../../algebra/files.h"
#include "common.h"

#ifdef MATH21_FLAG_USE_OPENCL

namespace math21 {
    const static std::string kernel_file_set = "generic_01_set.kl";
    Map_ <std::string, std::shared_ptr<m21clprogram>> programs_set;

// a special kind of sub, region sub.
// x is sub-tensor of y
    template<typename T>
    void math21_template_tensor_subregion_set_or_get_opencl(NumN n, PtrVoidWrapper x, PtrVoidWrapper y, NumN dims,
                                                            PtrNInWrapper dx, PtrNInWrapper dy, PtrNInWrapper offset,
                                                            NumB isGet) {
        math21_tool_assert(dims <= MATH21_KERNEL_ARRAY_MAX_LENGTH);
//    x -= 1;
//    y -= 1;
//    dx -= 1;
//    dy -= 1;
//    offset -= 1;
        cl_kernel kernel = math21_opencl_kernel_get<T>(
                "math21_template_tensor_subregion_set_or_get_opencl_kernel", kernel_file_set, programs_set);
        math21_opencl_kernel_arg_set(kernel, n, x, y, dims, dx, dy, offset, isGet);
        math21_opencl_kernel_run(kernel, n);
    }

    template<typename T>
    void math21_template_matrix_set_by_matrix_opencl(NumN d1, NumN d2,
                                                     PtrVoidInWrapper x, NumN d1_x, NumN d2_x, NumN stride1_x,
                                                     NumN stride2_x,
                                                     PtrVoidWrapper y, NumN d1_y, NumN d2_y, NumN stride1_y,
                                                     NumN stride2_y,
                                                     NumN offset_x, NumN offset_y) {
        d2_x = stride1_x * d2_x; // stride absorbed into next dim, so stride will become 1.
        d2_y = stride1_y * d2_y;
//    x += offset_x;
//    y += offset_y;
//    x -= 1;
//    y -= 1;
        NumN n = d1 * d2;
        cl_kernel kernel = math21_opencl_kernel_get<T>(
                "math21_template_matrix_set_by_matrix_opencl_kernel", kernel_file_set, programs_set);
        math21_opencl_kernel_arg_set(kernel, n, d2, x, d2_x, stride2_x, y, d2_y, stride2_y, offset_x, offset_y);
        math21_opencl_kernel_run(kernel, n);
    }

    template<typename T>
    void math21_template_tensor_3d_set_by_tensor_3d_opencl(NumN d1, NumN d2, NumN d3,
                                                           PtrVoidInWrapper x, NumN d1_x, NumN d2_x, NumN d3_x,
                                                           NumN stride1_x, NumN stride2_x, NumN stride3_x,
                                                           PtrVoidWrapper y, NumN d1_y, NumN d2_y, NumN d3_y,
                                                           NumN stride1_y, NumN stride2_y, NumN stride3_y,
                                                           NumN offset_x, NumN offset_y) {
        d2_x = stride1_x * d2_x;
        d2_y = stride1_y * d2_y;
        d3_x = stride2_x * d3_x;
        d3_y = stride2_y * d3_y;
//    x += offset_x;
//    y += offset_y;
//    x -= 1;
//    y -= 1;
        NumN n = d1 * d2 * d3;
        cl_kernel kernel = math21_opencl_kernel_get<T>(
                "math21_template_tensor_3d_set_by_tensor_3d_opencl_kernel", kernel_file_set, programs_set);
        math21_opencl_kernel_arg_set(kernel, n, d2, d3, x, d2_x, d3_x, stride3_x, y, d2_y, d3_y, stride3_y, offset_x,
                                     offset_y);
        math21_opencl_kernel_run(kernel, n);
    }

    template<typename T>
    void math21_template_vector_set_by_value_opencl(NumN n, T value, PtrVoidWrapper x, NumN stride_x) {
//    x -= 1;

        cl_kernel kernel = math21_opencl_kernel_get<T>(
                "math21_template_vector_set_by_value_opencl_kernel", kernel_file_set, programs_set);
        math21_opencl_kernel_arg_set(kernel, n, value, x, stride_x);
        math21_opencl_kernel_run(kernel, n);
    }
}

#endif