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

#include "inner.h"
#include "../../gpu/files.h"
#include "common.h"
#include "../kernels/generic_01_vector_set.kl"

namespace math21 {
    // see math21_vector_assign_from_vector_byte_cuda
    template<typename T1, typename T2>
    void math21_template_vector_set_by_vector_cuda(NumN n, const T1 *x, NumN stride_x, T2 *y, NumN stride_y,
                                                   NumN offset_x, NumN offset_y) {
        x += offset_x;
        y += offset_y;

        x -= 1;
        y -= 1;
        math21_cuda_kernel_arg_set_and_run(
                n,
                math21_template_vector_set_by_vector_cuda_kernel<T1, T2>,
                n, x, stride_x, y, stride_y);
    }
}