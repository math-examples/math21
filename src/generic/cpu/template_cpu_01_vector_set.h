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

#define MATH21_IS_FROM_CPU

#include "../kernels/generic_01_vector_set.kl"

#undef MATH21_IS_FROM_CPU

namespace math21 {

    // see math21_vector_assign_from_vector_byte_cpu
    template<typename T1, typename T2>
    void math21_template_vector_set_by_vector_cpu(NumN n, const T1 *x, NumN stride_x, T2 *y, NumN stride_y,
                                                  NumN offset_x, NumN offset_y) {
        x += offset_x;
        y += offset_y;

        x -= 1;
        y -= 1;
        NumN id;
#pragma omp parallel for
        for (id = 1; id <= n; ++id) {
            math21_template_vector_set_by_vector_cpu_kernel(n, x, stride_x, y, stride_y, id);
        }
    }

}