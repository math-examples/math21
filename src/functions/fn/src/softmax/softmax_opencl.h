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
#include "inner.h"

#ifdef MATH21_FLAG_USE_OPENCL

void math21_ml_function_softmax_tree_opencl(PtrR32Wrapper input, int in_class_size, int mini_batch_size, int stride, float temp, PtrR32Wrapper output, m21tree hier);

void math21_ml_function_softmax_opencl(PtrR32Wrapper input, int n, int mini_batch_size, int batch_offset, int groups, int group_offset, int stride, float temp, PtrR32Wrapper output);

void math21_ml_function_softmax_x_ent_opencl(int n, PtrR32Wrapper pred, PtrR32Wrapper truth, PtrR32Wrapper delta, PtrR32Wrapper error);

#endif
