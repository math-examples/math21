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

#include "inner_c.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef MATH21_FLAG_USE_OPENCL

void math21_optimization_adam_update_part_2_opencl(int x_size, PtrR32Wrapper x, PtrR32Wrapper m, PtrR32Wrapper v, float beta1, float beta2, float alpha, float eps, int t);

#endif

#ifdef __cplusplus
}
#endif
