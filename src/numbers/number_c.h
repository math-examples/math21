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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include "_assert_c.h"
#include "number_types.h"
#include "config/clvector.h"
#include "export.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t NumN8;
typedef int8_t NumZ8; // 8 bit integer
typedef uint32_t NumN32;
typedef int32_t NumZ32; // 32 bit integer
typedef uint64_t NumN64;
typedef int64_t NumZ64;
typedef NumN64 NumSize;
typedef NumN64 NumPtr;

typedef double NumR64;

#ifndef MATH21_FLAG_USE_OPENCL
typedef const NumR *PtrRInWrapper;
typedef NumR *PtrRWrapper;
typedef const NumR *PtrRInGpu;
typedef NumR *PtrRGpu; // always is gpu vector
typedef const float *PtrR32InWrapper;
typedef float *PtrR32Wrapper;
typedef float *PtrR32Gpu;
typedef const int *PtrIntInWrapper;
typedef int *PtrIntWrapper;
typedef const NumN8 *PtrN8InWrapper;
typedef NumN8 *PtrN8Wrapper;
typedef void *PtrVoidWrapper;
typedef const void *PtrVoidInWrapper;
typedef NumN *PtrNWrapper;
typedef const NumN *PtrNInWrapper;
#else
typedef m21clvector PtrRInWrapper;
typedef m21clvector PtrRWrapper;
typedef m21clvector PtrRInGpu;
typedef m21clvector PtrRGpu; // always is gpu vector
typedef m21clvector PtrR32InWrapper;
typedef m21clvector PtrR32Wrapper;
typedef m21clvector PtrR32Gpu;
typedef m21clvector PtrIntInWrapper;
typedef m21clvector PtrIntWrapper;
typedef m21clvector PtrN8InWrapper;
typedef m21clvector PtrN8Wrapper;
typedef m21clvector PtrVoidWrapper;
typedef m21clvector PtrVoidInWrapper;
typedef m21clvector PtrNWrapper;
typedef m21clvector PtrNInWrapper;
#endif

// define specifier
#if defined(MATH21_WINDOWS)
#define NumSize_SPECIFIER    "%llu"
#elif defined(MATH21_LINUX)
#define NumSize_SPECIFIER    "%lu"
#else
#define NumSize_SPECIFIER    "%lu"
#endif

#if defined(_MSC_VER) || defined(__MINGW32__) //__MINGW32__ should goes before __GNUC__
#define SIZE_T_SPECIFIER    "%Iu"
#elif defined(__GNUC__)
#define SIZE_T_SPECIFIER    "%zu"
#else
#define SIZE_T_SPECIFIER    "%u"
#endif

#if defined(MATH21_FLAG_USE_CPU)
#define M21_EXPORT
#elif defined(MATH21_FLAG_USE_CUDA)
#define M21_EXPORT __host__ __device__
#elif defined(MATH21_FLAG_USE_OPENCL)
#define M21_EXPORT
#endif

#define MATH21_TIME_MAX (10000000)
#define MATH21_OPT_TIME_MAX (10000000)
#define XJ_TRUE 1
#define XJ_FALSE 0

#define MATH21_EPS2 (1e-10)
#define MATH21_EPS (1e-6)

#if defined(MATH21_USE_NUMR32)
#define MATH21_EPSILON MATH21_EPS
#else
#define MATH21_EPSILON MATH21_EPS2
#endif

#define MATH21_10NEG6 (0.000001)
#define MATH21_10NEG7 (0.0000001)
#define MATH21_EPS_NEG (-0.000001)
#define XJ_EPS (0.000001)
#define XJ_PI (3.14159265357989323)
#define XJ_TWO_PI 6.2831853071795864769252866f
#define XJ_MAX (10000000)
#define XJ_MIN (-10000000)
#define MATH21_GPU_BLOCK_SIZE 16
#define MATH21_CUDA_BLOCK_SIZE 512 // number of threads per dimension of block
#define MATH21_OPENCL_BLOCK_SIZE 512 // meaning different from that of cuda
#define MATH21_DIMS_RAW_TENSOR 3 // used by rawtensor
#define MATH21_DIMS_MAX_RAW_TENSOR 8 // used by rawtensor
#define MATH21_MASK_NUM 212121
#define math21_raise(exp) assert(exp)
#define math21_tool_assert_to_do_remove(exp) assert(exp)

#define math21_tool_assert(_exp)                                              \
    {if ( !(_exp) )                                                         \
    {                                                                       \
        math21_assert_breakpoint();                                           \
        xjassert();      \
    }}

#define math21_tool_container_size_assert(_exp) math21_tool_assert(_exp)


// debug
//#ifdef fprintf
//#undef fprintf
//#endif
//#define fprintf(X, ...) {}

enum {
    m21_device_type_default = 1,
    m21_device_type_gpu,
};

enum {
    m21_type_none = 0,
    m21_type_default,
    m21_type_NumN,
    m21_type_NumZ,
    m21_type_NumR,
    m21_type_Seqce,
    m21_type_Tensor,
    m21_type_Digraph,
    m21_type_vector_float_c,
    m21_type_vector_char_c,
    m21_type_NumR32,
    m21_type_NumR64,
    m21_type_TenN,
    m21_type_TenZ,
    m21_type_TenR, // 14
    m21_type_PointAd ,
    m21_type_NumSize,
    m21_type_NumN8,
    m21_type_TenN8,
    m21_type_string,
    m21_type_TenStr, // 20
    m21_type_SeqTenN,
    m21_type_SeqTenZ,
    m21_type_SeqTenR,
    m21_type_SeqSeqStr,
};

enum {
    m21_fname_none = 0,
    m21_fname_sum,
    m21_fname_norm1,
    m21_fname_norm2_square,
    m21_fname_mean,
    m21_fname_max,
    m21_fname_min,
    m21_fname_argmax,
    m21_fname_argmin,
    m21_fname_argmax_random,
    m21_fname_inner_product,
    m21_fname_distance_1,
    m21_fname_distance_2_square,
    m21_fname_add,
    m21_fname_subtract,
    m21_fname_multiply,
    m21_fname_divide,
    m21_fname_ele_is_equal,
    m21_fname_ele_is_less_than,
    m21_fname_ele_is_not_less_than, // greater than or equal to
    m21_fname_set_using_mask,
    m21_fname_sin,
    m21_fname_cos,
    m21_fname_tan,
    m21_fname_cot,
    m21_fname_exp,
    m21_fname_log,
    m21_fname_abs,
    m21_fname_kx_add, // k+x
    m21_fname_kx_subtract, // k-x
    m21_fname_xk_subtract, // x-k
    m21_fname_kx_mul, // k*x
    m21_fname_kx_divide, // k/x
    m21_fname_xk_divide, // x/k
    m21_fname_kx_pow, // k^x
    m21_fname_xk_pow, // x^k
    m21_fname_addto,
    m21_fname_multo,
};

enum {
    m21_flag_resize_method_default = 1,
    m21_flag_resize_method_sampling,
    m21_flag_resize_method_pooling,
    m21_flag_resize_method_interpolation_bilinear
};

enum {
    m21_flag_interpolation_none = 0,
    m21_flag_interpolation_bilinear,
};

enum {
    m21_flag_projection_none = 0,
    m21_flag_projection_euclidean, // n = 3 => dof = 3 and n = 4 => dof = 6 where m = n
    m21_flag_projection_similarity, // n = 3 => dof = 4 and n = 4 => dof = 7 where m = n
    m21_flag_projection_affine, // dof = (m-1)*n
    m21_flag_projection_projective, // dof = m*n-1
};

// object, not similar to std::shared_ptr
typedef struct m21point m21point;
struct m21point {
    NumN type; // type of data element in m21_type
    void *p; // pointer pointing to object.
    NumN *refCount;// when points to user-allocated data, the pointer is NULL
};

// todo: deprecate, use 'm21point + tensor' instead.
struct m21rawtensor {
    NumN type; // type of data element in math21_type
    NumN dims;
    NumN *d; // pointer to data address of Tensor shape, or sth created by user.
    void *data; // pointer to data address of Tensor, or sth created by user.
};
typedef struct m21rawtensor m21rawtensor;

struct m21image {
    NumN nr;
    NumN nc;
    NumN nch;
    NumR32 *data; // data shape: nch * nr * nc
};
typedef struct m21image m21image;

struct m21array2d {
    NumN type; // type of data element in math21_type
    NumN nr;
    NumN nc;
    void **data;
};
typedef struct m21array2d m21array2d;

struct m21data {
    m21array2d x;
    m21array2d y;
    NumB shallow;
};
typedef struct m21data m21data;

#define MATH21_STRINGIFY(x) #x
#define MATH21_CONCATENATOR(x, y) x##y

#ifdef __cplusplus
}
#endif
