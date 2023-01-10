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

#ifdef MATH21_FLAG_USE_OPENCL

m21clvector math21_vector_deserialize_c_opencl(FILE *f, size_t *n0);

m21clvector math21_vector_deserialize_from_file_opencl(const char *name, size_t *n);

void math21_vector_serialize_c_opencl(FILE *f, m21clvector v, size_t n);

void math21_vector_serialize_to_file_opencl(const char *name, m21clvector v, size_t n);

// [from, to)
void math21_vector_save_opencl(const char *name, m21clvector v, size_t from, size_t to);

void math21_vector_log_opencl(const char *name, m21clvector v, size_t from, size_t to);

m21clvector math21_vector_create_with_default_value_opencl(size_t n, float value);

m21clvector math21_vector_create_with_default_value_int_opencl(size_t n, int value);

m21clvector math21_vector_create_buffer_opencl(size_t n, size_t elementSize);

PtrR32Wrapper math21_vector_resize_with_default_value_opencl(PtrR32Wrapper v, size_t n, float value);

PtrVoidWrapper math21_vector_resize_buffer_opencl(PtrVoidWrapper v, size_t n, size_t elementSize);

PtrIntWrapper math21_vector_resize_with_default_value_int_opencl(PtrIntWrapper v, size_t n, int value);

m21clvector math21_vector_create_from_cpuvector_opencl(size_t n, const float *x, int stride_x);

m21clvector math21_vector_create_from_cpuvector_int_opencl(size_t n, const int *x, int stride_x);

void math21_vector_free_opencl(m21clvector x_gpu);

void math21_vector_mean_fast_opencl(PtrR32InWrapper X, int mini_batch_size, int features_size, int in_class_size,
                                    PtrR32Wrapper mean);

void math21_vector_variance_fast_opencl(m21clvector x, m21clvector mean, int batch, int filters, int spatial, m21clvector variance);

void math21_vector_assign_from_vector_N8_opencl(int n, m21clvector X, m21clvector Y);

void math21_vector_assign_from_vector_opencl(int n, m21clvector X, int stride_x, m21clvector Y, int stride_y);

void math21_vector_kx_opencl(int n, float k, PtrR32Wrapper x, int stride_x);

void math21_vector_k_add_x_opencl(int n, float k, PtrR32Wrapper x, int stride_x);

void math21_vector_kx_add_y_opencl(int n, float k, PtrR32InWrapper x, int stride_x, PtrR32Wrapper y,
                                    int stride_y);

void math21_vector_normalize_opencl(PtrR32Wrapper x, PtrR32InWrapper mean, PtrR32InWrapper variance,
                                     int mini_batch_size, int features_size,
                                     int in_class_size);

void
math21_vector_kx_with_in_class_opencl(PtrR32Wrapper x, PtrR32InWrapper k, int mini_batch_size,
                                       int features_size,
                                       int in_class_size);

void math21_vector_x_add_b_with_in_class_opencl(m21clvector output, m21clvector biases, int batch, int n, int size);

void math21_vector_sum_with_in_class_opencl(PtrR32Wrapper db, PtrR32InWrapper dY, int mini_batch_size,
                                             int features_size,
                                             int in_class_size);
void math21_vector_sum_SchurProduct_with_in_class_opencl(PtrR32InWrapper X, PtrR32InWrapper dY,
                                                          int mini_batch_size, int features_size,
                                                          int in_class_size, PtrR32Wrapper dk);
void math21_vector_set_opencl(int N, float ALPHA, m21clvector X, int INCX);

void math21_vector_set_int_opencl(int N, int ALPHA, m21clvector X, int INCX);

void math21_vector_feature2d_add_2_opencl(
        int mini_batch_size,
        float kx, PtrR32InWrapper X, int nch_X, int nr_X, int nc_X,
        float ky, PtrR32Wrapper Y, int nch_Y, int nr_Y, int nc_Y);

void math21_vector_feature2d_add_3_opencl(
        int mini_batch_size,
        float kx, PtrR32InWrapper X, int nch_X, int nr_X, int nc_X,
        float kx2, PtrR32InWrapper X2, int nch_X2, int nr_X2, int nc_X2,
        float ky, PtrR32Wrapper Y, int nch_Y, int nr_Y, int nc_Y);

void math21_vector_feature2d_sample_opencl(
        int mini_batch_size,
        PtrR32Wrapper X, int nch_X, int nr_X, int nc_X, int stride_X, int is_upsample, float k,
        PtrR32Wrapper Y);

void math21_vector_clip_opencl(int n, float k, PtrR32Wrapper x, int stride);

void math21_vector_xy_opencl(int n, PtrR32InWrapper x, int stride_x, PtrR32Wrapper y, int stride_y);

void math21_vector_assign_by_mask_opencl(int n, PtrR32Wrapper x, float mask_num, PtrR32InWrapper mask,
                                          float val);

void math21_vector_kx_by_mask_opencl(int n, float k, PtrR32Wrapper x, PtrR32InWrapper mask,
                                          float mask_num);

void math21_vector_pr_rand_uniform_01_opencl(m21clvector x_gpu, size_t n);

void math21_vector_loss_l1_opencl(int n, PtrR32InWrapper x, PtrR32InWrapper t,
                                   PtrR32Wrapper dx, PtrR32Wrapper error);

void math21_vector_loss_l2_opencl(int n, PtrR32InWrapper x, PtrR32InWrapper t,
                                   PtrR32Wrapper dx, PtrR32Wrapper error);

void math21_vector_loss_smooth_l1_opencl(int n, PtrR32InWrapper x, PtrR32InWrapper t,
                                   PtrR32Wrapper dx, PtrR32Wrapper error);

void math21_vector_zero_by_thresh_opencl(int n, PtrR32Wrapper x, int stride_x, float thresh);

#endif