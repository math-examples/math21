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

void math21_function_conv2d_X_to_X_prime_wrapper(PtrVoidInWrapper X, PtrVoidWrapper X_prime,
                                                 NumN nch_X, NumN nr_X, NumN nc_X,
                                                 NumN ksize, NumN stride, NumN pad, NumN type);

void math21_function_conv2d_dX_prime_to_dX_wrapper(PtrVoidInWrapper dX_prime, PtrVoidWrapper dX,
                                                   NumN nch_X, NumN nr_X, NumN nc_X,
                                                   NumN ksize, NumN stride, NumN pad, NumN type);

NumB math21_function_conv2d_is_X_equal_to_X_prime(NumN k_size, NumN stride, NumN pad);

void math21_function_conv2d_forward_wrapper(
        PtrVoidWrapper Y,
        PtrVoidInWrapper W,
        PtrVoidInWrapper X_input,
        NumN nr_X,
        NumN nc_X,
        NumN nch_X,
        NumN nr_Y,
        NumN nc_Y,
        NumN nch_Y,
        NumN y_size,
        NumN n_W,
        NumN batch,
        NumN k_size,
        NumN stride,
        NumN pad,
        NumN n_group,
        PtrVoidWrapper workspace, NumN type);

void math21_function_conv2d_backward_wrapper(
        PtrVoidInWrapper dY,
        PtrVoidInWrapper W,
        PtrVoidWrapper dW,
        PtrVoidInWrapper X_input,
        PtrVoidWrapper dX_input,
        NumN nr_X,
        NumN nc_X,
        NumN nch_X,
        NumN nr_Y,
        NumN nc_Y,
        NumN nch_Y,
        NumN y_size,
        NumN n_W,
        NumN batch,
        NumN k_size,
        NumN stride,
        NumN pad,
        NumN n_group,
        PtrVoidWrapper workspace, NumN type);


void math21_function_conv2d_bias_forward_wrapper(
        PtrR32Wrapper x, PtrR32InWrapper b, NumN mini_batch_size,
        NumN features_size, NumN in_class_size);

void math21_function_conv2d_bias_backward_wrapper(
        PtrR32Wrapper db, PtrR32InWrapper dY, NumN mini_batch_size,
        NumN features_size, NumN in_class_size);

#ifdef __cplusplus
}
#endif
