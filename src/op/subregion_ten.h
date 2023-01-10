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
#include "detail/set_tensor_subregion.h"

namespace math21 {
    // see math21_op_subtensor_set, math21_op_tensor_3d_f_set_by_tensor_3d
    // set y using sub-tensor x
    // x -> y, x is sub-tensor of y
    template<typename T>
    void math21_op_subregion_set(const Tensor<T> &x, Tensor<T> &y, const VecN &offset) {
        detail::imath21_op_subregion_f_set(m21_fname_none, x, y, offset);
    }

    // see math21_op_subtensor_get
    // a special kind of sub, region sub.
    // x <- y, x is sub-tensor of y, here x, y can be empty
    template<typename T>
    void math21_op_subregion_get(Tensor<T> &x, const Tensor<T> &y, const VecN &offset, const VecN &dx) {
        detail::imath21_op_subregion_f_get(m21_fname_none, x, y, offset, dx);
    }
}