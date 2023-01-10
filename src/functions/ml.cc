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

#include "../gpu/files.h"
#include "../generic/files.h"
#include "ml.h"

void math21_function_conv2d_X_to_X_prime_wrapper(PtrVoidInWrapper X, PtrVoidWrapper X_prime,
                                                 NumN nch_X, NumN nr_X, NumN nc_X,
                                                 NumN ksize, NumN stride, NumN pad, NumN type) {
    math21_generic_cross_correlation_X_to_X_prime_wrapper(
            X, X_prime, nch_X, nr_X, nc_X,
            ksize, ksize, pad, pad, stride, stride, 1, 1, type);
}

void math21_function_conv2d_dX_prime_to_dX_wrapper(PtrVoidInWrapper dX_prime, PtrVoidWrapper dX,
                                                   NumN nch_X, NumN nr_X, NumN nc_X,
                                                   NumN ksize, NumN stride, NumN pad, NumN type) {
    math21_generic_cross_correlation_dX_prime_to_dX_wrapper(
            dX_prime, dX, nch_X, nr_X, nc_X,
            ksize, ksize, pad, pad, stride, stride, 1, 1, type);
}

NumB math21_function_conv2d_is_X_equal_to_X_prime(NumN k_size, NumN stride, NumN pad) {
    if (k_size == 1 && stride == 1 && pad == 0) {
        return 1;
    } else {
        return 0;
    }
}

//  Z = h(Y), Y = W*X + b, or Y = X*W.t + b
//  Y_m = K_m * X_prime + b ...
// float *workspace;// X_prime or dL/dX_prime
// workspace is global space, and has size at least workspace_size.
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
        PtrVoidWrapper workspace, NumN type) {
    int imb, igroup;

    // Y_m = 0
    math21_generic_vector_set_by_value_wrapper(y_size * batch, 0, Y, 1, type);

    // nr_Y_m = group_size_Y
    int nr_Y_m = nch_Y / n_group;
    // n_common = group_size_X * nr_K * nc_K
    int n_common = (nch_X / n_group) * k_size * k_size;
    int nc_Y_m = nc_Y * nr_Y;
    for (imb = 0; imb < batch; ++imb) {
        for (igroup = 0; igroup < n_group; ++igroup) {
            // K shape: nch_Y * group_size_X * nr_K * nc_K
            // K shape: (num_group * group_size_Y) * group_size_X * nr_K * nc_K
            // K shape: num_group * group_size_Y * n_common
            // K shape: num_group * nr_Y_m * n_common
            PtrVoidInWrapper K_m = math21_number_pointer_input_increase(W, igroup * (n_W / n_group), type);
            PtrVoidWrapper X_prime = workspace;
            // X shape: mbs * num_group * group_size_X * nr_X * nc_X
            PtrVoidInWrapper X = math21_number_pointer_input_increase(
                    X_input, (imb * n_group + igroup) * (nch_X / n_group) * nr_X * nc_X, type);
            // Y shape: mbs * nch_Y * nr_Y * nc_Y
            // Y shape: mbs * num_group * group_size_Y * nr_Y * nc_Y
            // Y shape: mbs * num_group * nr_Y_m * nr_Y * nc_Y
            // Y shape: mbs * num_group * nr_Y_m * nc_Y_m
            PtrVoidWrapper Y_m = math21_number_pointer_increase(Y, (imb * n_group + igroup) * nr_Y_m * nc_Y_m,
                                                                    type);

            if (math21_function_conv2d_is_X_equal_to_X_prime(k_size, stride, pad)) {
                X_prime = (PtrVoidWrapper) X;
            } else {
                // X -> X_prime
                // X_prime shape: (group_size_X * nr_K * nc_K ) * (nr_Y * nc_Y)
                // X_prime shape: (group_size_X * nr_K * nc_K ) * nc_Y_m
                // X_prime shape: n_common * nc_Y_m
                math21_function_conv2d_X_to_X_prime_wrapper(X, X_prime, nch_X / n_group, nr_X, nc_X, k_size, stride,
                                                            pad, type);
            }
            // Y_m = K_m * X_prime + Y_m, K_m: nr_Y_m*n_common, X_prime: n_common*nc_Y_m, Y_m: nr_Y_m*nc_Y_m
            math21_generic_matrix_multiply_onto_k1AB_add_k2C_similar_wrapper(0, 0, nr_Y_m, nc_Y_m, n_common, 1, K_m,
                                                                             n_common,
                                                                             X_prime, nc_Y_m, 1, Y_m, nc_Y_m,
                                                                             type);
        }
    }
}

// dL/dZ => dL/dK, dL/dX
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
        PtrVoidWrapper workspace, NumN type) {

    int imb, igroup;
    int nr_dK_m = nch_Y / n_group;
    int nc_dK_m = k_size * k_size * nch_X / n_group;
    int nc_dY_m = nc_Y * nr_Y;

    for (imb = 0; imb < batch; ++imb) {
        for (igroup = 0; igroup < n_group; ++igroup) {
            // dK shape: nch_Y * group_size_X * nr_K * nc_K
            // dK shape: (num_group * group_size_Y) * group_size_X * nr_K * nc_K
            // dK shape: num_group * group_size_Y * n_common
            // dK shape: num_group * nr_dY_m * n_common
            // dK shape: num_group * nr_dK_m * n_common
            PtrVoidWrapper dK_m = math21_number_pointer_increase(dW, igroup * (n_W / n_group), type);
            PtrVoidWrapper X_prime = workspace;
            // X shape: mbs * num_group * group_size_X * nr_X * nc_X
            PtrVoidInWrapper X =
                    math21_number_pointer_input_increase(
                            X_input, (imb * n_group + igroup) * (nch_X / n_group) * nr_X * nc_X, type);
            // dY shape: mbs * nch_Y * nr_Y * nc_Y
            // dY shape: mbs * num_group * group_size_Y * nr_Y * nc_Y
            // dY shape: mbs * num_group * nr_dK_m * nr_Y * nc_Y
            // dY shape: mbs * num_group * nr_dK_m * nc_dY_m
            // dY shape: mbs * num_group * nr_dY_m * nc_dY_m
            PtrVoidInWrapper dY_m =
                    math21_number_pointer_input_increase(
                            dY, (imb * n_group + igroup) * nr_dK_m * nc_dY_m, type);
            if (math21_function_conv2d_is_X_equal_to_X_prime(k_size, stride, pad)) {
                X_prime = (PtrVoidWrapper) X;
            } else {
                // X -> X_prime
                math21_function_conv2d_X_to_X_prime_wrapper(X, X_prime, nch_X / n_group, nr_X, nc_X,
                                                            k_size, stride, pad, type);
            }
            // dL/dW += dL/dY * X.t
            // dL/dK_m += dL/dY_m * X_prime.t
            math21_generic_matrix_multiply_onto_k1AB_add_k2C_similar_wrapper(0, 1, nr_dK_m, nc_dK_m, nc_dY_m, 1, dY_m,
                                                                             nc_dY_m,
                                                                             X_prime, nc_dY_m, 1, dK_m, nc_dK_m,
                                                                             type);

            if (!math21_vector_isEmpty_wrapper(dX_input)) {
                PtrVoidInWrapper K_m = math21_number_pointer_input_increase(W, igroup * n_W / n_group, type);
                PtrVoidWrapper dX_prime = workspace;
                PtrVoidWrapper dX =
                        math21_number_pointer_increase(
                                dX_input, (imb * n_group + igroup) * (nch_X / n_group) * nr_X * nc_X, type);
                if (math21_function_conv2d_is_X_equal_to_X_prime(k_size, stride, pad)) {
                    dX_prime = dX;
                }

                // dL/dX = W.t * dL/dY
                // dL/dX_prime = K_m.t * dL/dY_m
                math21_generic_matrix_multiply_onto_k1AB_add_k2C_similar_wrapper(1, 0, nc_dK_m, nc_dY_m, nr_dK_m, 1,
                                                                                 K_m, nc_dK_m,
                                                                                 dY_m, nc_dY_m, 0, dX_prime, nc_dY_m,
                                                                                 type);

                if (!math21_function_conv2d_is_X_equal_to_X_prime(k_size, stride, pad)) {
                    // dX_prime -> dX
                    math21_function_conv2d_dX_prime_to_dX_wrapper(dX_prime, dX, nch_X / n_group, nr_X, nc_X,
                                                                  k_size,
                                                                  stride, pad, type);
                }
            }
        }
    }
}

void math21_function_conv2d_bias_forward_wrapper(
        PtrR32Wrapper x, PtrR32InWrapper b, NumN mini_batch_size,
        NumN features_size, NumN in_class_size) {
    math21_vector_x_add_b_with_in_class_wrapper(x, b, mini_batch_size, features_size, in_class_size);
}

void math21_function_conv2d_bias_backward_wrapper(
        PtrR32Wrapper db, PtrR32InWrapper dY, NumN mini_batch_size,
        NumN features_size, NumN in_class_size) {
    math21_vector_sum_with_in_class_wrapper(db, dY, mini_batch_size, features_size, in_class_size);
}
