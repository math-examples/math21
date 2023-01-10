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

PtrR32Wrapper math21_vector_deserialize_c_wrapper(FILE *f, size_t *n);

PtrR32Wrapper math21_vector_deserialize_from_file_wrapper(const char *name, size_t *n);

void math21_vector_serialize_c_wrapper(FILE *f, PtrR32InWrapper v, size_t n);

void math21_vector_serialize_to_file_wrapper(const char *name, PtrR32InWrapper v, size_t n);

void math21_vector_save_wrapper(const char *name, PtrR32InWrapper v, size_t from, size_t to);

void math21_vector_log_wrapper(const char *name, PtrR32InWrapper v, size_t from, size_t to);

PtrR32Wrapper math21_vector_create_with_default_value_wrapper(size_t n, float value);

PtrVoidWrapper math21_vector_create_buffer_wrapper(size_t n, size_t elementSize);

// data not kept
PtrR32Wrapper math21_vector_resize_with_default_value_wrapper(PtrR32Wrapper v, size_t n, float value);

// data not kept
PtrVoidWrapper math21_vector_resize_buffer_wrapper(PtrVoidWrapper v, size_t n, size_t elementSize);

// data not kept
PtrIntWrapper math21_vector_resize_with_default_value_int_wrapper(PtrIntWrapper v, size_t n, int value);

PtrR32Wrapper math21_vector_create_from_cpuvector_wrapper(size_t n, const float *x, int stride_x);

PtrIntWrapper math21_vector_create_from_cpuvector_int_wrapper(size_t n, const int *x, int stride_x);

void math21_vector_free_wrapper(PtrVoidWrapper x);

void
math21_vector_mean_wrapper(PtrR32InWrapper x, int batch, int filters, int spatial, PtrR32Wrapper mean);

void
math21_vector_variance_wrapper(PtrR32InWrapper X, PtrR32InWrapper mean, int mini_batch_size,
                               int features_size,
                               int in_class_size,
                               PtrR32Wrapper variance);

void math21_vector_assign_from_vector_N8_wrapper(int n, PtrN8InWrapper x, PtrN8Wrapper y);

void math21_vector_assign_from_vector_wrapper(int n, PtrR32InWrapper x, int stride_x, PtrR32Wrapper y,
                                              int stride_y);

void math21_vector_kx_wrapper(int n, float k, PtrR32Wrapper x, int stride_x);

void math21_vector_k_add_x_wrapper(int n, float k, PtrR32Wrapper x, int stride_x);

void math21_vector_kx_add_y_wrapper(int n, float k, PtrR32InWrapper x, int stride_x, PtrR32Wrapper y,
                                    int stride_y);

void
math21_vector_normalize_wrapper(PtrR32Wrapper x, PtrR32InWrapper mean, PtrR32InWrapper variance,
                                int mini_batch_size, int features_size,
                                int in_class_size);

void math21_vector_kx_with_in_class_wrapper(PtrR32Wrapper x, PtrR32InWrapper k, int mini_batch_size,
                                            int features_size,
                                            int in_class_size);

void math21_vector_x_add_b_with_in_class_wrapper(PtrR32Wrapper x, PtrR32InWrapper b, int mini_batch_size,
                                                 int features_size,
                                                 int in_class_size);

float math21_vector_sum(const float *x, int n);

void math21_vector_sum_with_in_class_wrapper(PtrR32Wrapper db, PtrR32InWrapper dY, int mini_batch_size,
                                             int features_size,
                                             int in_class_size);

void math21_vector_sum_SchurProduct_with_in_class_wrapper(PtrR32InWrapper X, PtrR32InWrapper dY,
                                                          int mini_batch_size, int features_size,
                                                          int in_class_size, PtrR32Wrapper dk);

void math21_vector_set_wrapper(int n, float value, PtrR32Wrapper X, int stride_x);

void math21_vector_set_int_wrapper(int n, int value, PtrIntWrapper X, int stride_x);

void math21_vector_assign_3d_d2_wrapper(PtrR32InWrapper data1, PtrR32Wrapper data2,
                                        int d1, int d2, int d3, int d2y, int offset2, int isToSmall);

void math21_vector_transpose_d1234_to_d1324_wrapper(PtrR32InWrapper x, PtrR32Wrapper y,
                                                    int d1, int d2, int d3, int d4);

void math21_vector_feature2d_add_2_wrapper(
        int mini_batch_size,
        float kx, PtrR32InWrapper X, int nch_X, int nr_X, int nc_X,
        float ky, PtrR32Wrapper Y, int nch_Y, int nr_Y, int nc_Y);

void math21_vector_feature2d_add_3_wrapper(
        int mini_batch_size,
        float kx, PtrR32InWrapper X, int nch_X, int nr_X, int nc_X,
        float kx2, PtrR32InWrapper X2, int nch_X2, int nr_X2, int nc_X2,
        float ky, PtrR32Wrapper Y, int nch_Y, int nr_Y, int nc_Y);

void math21_vector_feature2d_sample_wrapper(
        int mini_batch_size,
        PtrR32Wrapper X, int nch_X, int nr_X, int nc_X, int stride_X, int is_upsample, float k,
        PtrR32Wrapper Y);

void math21_vector_clip_wrapper(int n, float k, PtrR32Wrapper x, int stride);

void math21_vector_xy_wrapper(int n, PtrR32InWrapper x, int stride_x, PtrR32Wrapper y, int stride_y);

void
math21_vector_assign_by_mask_wrapper(int n, PtrR32Wrapper x, float mask_num, PtrR32InWrapper mask,
                                     float val);

void math21_vector_kx_by_mask_wrapper(int n, float k, PtrR32Wrapper x, PtrR32InWrapper mask,
                                      float mask_num);

NumB math21_vector_isEmpty_wrapper(PtrVoidInWrapper x);

PtrVoidWrapper math21_vector_getEmpty_wrapper();

PtrN8Wrapper math21_vector_getEmpty_N8_wrapper();

PtrR32Wrapper math21_vector_getEmpty_R32_wrapper();

void math21_vector_push_wrapper(PtrR32Wrapper x_gpu, const float *x, size_t n);

void math21_vector_push_N8_wrapper(PtrN8Wrapper x_gpu, const NumN8 *x, size_t n);

void math21_vector_pull_wrapper(PtrR32InWrapper x_gpu, float *x, size_t n);

void math21_vector_pull_N8_wrapper(PtrN8InWrapper x_gpu, NumN8 *x, size_t n);

void math21_vector_log_pointer_wrapper(PtrR32InWrapper v);

void math21_vector_pr_rand_uniform_01_wrapper(PtrR32Wrapper v, int size);

void math21_vector_loss_l1_wrapper(int n, PtrR32InWrapper x, PtrR32InWrapper t,
                                   PtrR32Wrapper dx, PtrR32Wrapper error);

void math21_vector_loss_l2_wrapper(int n, PtrR32InWrapper x, PtrR32InWrapper t,
                                   PtrR32Wrapper dx, PtrR32Wrapper error);

void math21_vector_loss_smooth_l1_wrapper(int n, PtrR32InWrapper x, PtrR32InWrapper t,
                                          PtrR32Wrapper dx, PtrR32Wrapper error);

void math21_vector_zero_by_thresh_wrapper(int n, PtrR32Wrapper x, int stride_x, float thresh);

#ifdef __cplusplus
}
#endif
