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
    const static std::string kernel_file_set = "generic_04_set.kl";
    static Map_<std::string, std::shared_ptr<m21clprogram>> programs_set;

    template<typename T>
    void math21_template_subtensor_like_set_or_get_using_mask_in_d3_opencl(
            NumN n,
            PtrVoidWrapper x1,
            PtrNInWrapper x2,
            PtrVoidWrapper y,
            PtrNInWrapper map1,
            PtrNInWrapper map2,
            PtrNInWrapper map3,
            NumN dims_x1, PtrNInWrapper dx1,
            NumN dims_x2, PtrNInWrapper dx2,
            NumN dims_y, PtrNInWrapper dy,
            NumN dims_map1, PtrNInWrapper dmap1,
            NumN dims_map2, PtrNInWrapper dmap2,
            NumN dims_map3, PtrNInWrapper dmap3,
            NumB isGet) {
//    x1 -= 1;
//    x2 -= 1;
//    y -= 1;
//    map1 -= 1;
//    map2 -= 1;
//    map3 -= 1;
//    dx1 -= 1;
//    dx2 -= 1;
//    dy -= 1;
//    dmap1 -= 1;
//    dmap2 -= 1;
//    dmap3 -= 1;
        std::string functionName;
        functionName = "math21_template_subtensor_like_set_or_get_using_mask_in_d3_opencl_kernel";
        cl_kernel kernel = math21_opencl_kernel_get<T>(functionName, kernel_file_set, programs_set);
        math21_opencl_kernel_arg_set(kernel, n, x1, x2, y, map1, map2, map3,
                                     dims_x1, dx1, dims_x2, dx2, dims_y, dy,
                                     dims_map1, dmap1, dims_map2, dmap2, dims_map3, dmap3, isGet);
        math21_opencl_kernel_run(kernel, n);
    }
}

#endif