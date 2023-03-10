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

#include "../matrix/files.h"

namespace math21 {
    template<typename T>
    void math21_operator_tensor_share_add_axis(const Tensor <T> &x, Tensor <T> &y, NumZ pos);

    void math21_operator_tensor_f_shrink_axes_to_index(NumN dims, const VecN &axes, VecN &index);
}