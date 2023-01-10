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
    const static std::string kernel_file_vector_set = "generic_01_vector_set.kl";
    Map_ <std::string, std::shared_ptr<m21clprogram>> programs_vector_set;

// see math21_vector_assign_from_vector_byte_opencl
    template<typename T1, typename T2>
    void
    math21_template_vector_set_by_vector_opencl(NumN n, PtrVoidInWrapper x, NumN stride_x, PtrVoidWrapper y,
                                                NumN stride_y,
                                                NumN offset_x, NumN offset_y) {
//    x += offset_x;
//    y += offset_y;
//    x -= 1;
//    y -= 1;

        cl_kernel kernel = math21_opencl_kernel_get<T1, T2>(
                "math21_template_vector_set_by_vector_opencl_kernel", kernel_file_vector_set, programs_vector_set);
        math21_opencl_kernel_arg_set(kernel, n, x, stride_x, y, stride_y, offset_x, offset_y);
        math21_opencl_kernel_run(kernel, n);
    }
}

#endif