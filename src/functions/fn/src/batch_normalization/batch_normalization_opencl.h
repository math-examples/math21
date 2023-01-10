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

void math21_ml_batchnormalization_backward_mu_fast_opencl(PtrR32InWrapper dX_hat, PtrR32InWrapper variance, int mini_batch_size, int features_size, int in_class_size, PtrR32Wrapper dmu);

void
math21_ml_batchnormalization_backward_sigma_square_fast_opencl(PtrR32InWrapper X, PtrR32InWrapper dX_hat,
                                                           PtrR32InWrapper mu,
                                                           PtrR32InWrapper variance, int mini_batch_size,
                                                           int features_size, int in_class_size,
                                                           PtrR32Wrapper dvariance);
void math21_ml_batchnormalization_backward_input_opencl(PtrR32InWrapper X, PtrR32InWrapper mu, PtrR32InWrapper variance,
                                                        PtrR32InWrapper dmu, PtrR32InWrapper dvariance, int mini_batch_size,
                                                        int features_size, int in_class_size, PtrR32Wrapper dX_hat);

#endif
