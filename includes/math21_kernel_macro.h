//#define MATH21_IS_FROM_CPU

#if !defined(MATH21_IS_FROM_CPU)
#if defined(MATH21_FLAG_USE_CUDA)
#define MATH21_IS_FROM_CUDA
#elif defined(MATH21_FLAG_USE_OPENCL)
#define MATH21_IS_FROM_OPENCL
#endif
#endif

#if defined(MATH21_IS_FROM_CPU)
#define MATH21_MAKE_KERNEL_NAME_SUFFIX_1(X, Y1) MATH21_MACRO_CAT_2(X, cpu_kernel)
#define MATH21_MAKE_KERNEL_NAME_SUFFIX_2(X, Y1, Y2) MATH21_MACRO_CAT_2(X, cpu_kernel)
#define MATH21_MAKE_KERNEL_NAME_SUFFIX_3(X, Y1, Y2, Y3) MATH21_MACRO_CAT_2(X, cpu_kernel)
#define MATH21_KERNEL_GET_ID()
#define MATH21_KERNEL_TEMPLATE_HEADER_1(X) template<typename X>
#define MATH21_KERNEL_TEMPLATE_HEADER_2(X1, X2) template<typename X1, typename X2>
#define MATH21_KERNEL_EXPORT
#define MATH21_KERNEL_GLOBAL
#define MATH21_KERNEL_INPUT_ID , NumN id
#define MATH21_KERNEL_INPUT_OFFSETS_XY
#define MATH21_DEVICE_MAKE_F_LIKE_PTR(X, Y) X Y,

#elif defined(MATH21_IS_FROM_CUDA)
#define MATH21_MAKE_KERNEL_NAME_SUFFIX_1(X, Y1) MATH21_MACRO_CAT_2(X, cuda_kernel)
#define MATH21_MAKE_KERNEL_NAME_SUFFIX_2(X, Y1, Y2) MATH21_MACRO_CAT_2(X, cuda_kernel)
#define MATH21_MAKE_KERNEL_NAME_SUFFIX_3(X, Y1, Y2, Y3) MATH21_MACRO_CAT_2(X, cuda_kernel)
#define MATH21_KERNEL_GET_ID() int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x; id +=1;
#define MATH21_KERNEL_TEMPLATE_HEADER_1(X) template<typename X>
#define MATH21_KERNEL_TEMPLATE_HEADER_2(X1, X2) template<typename X1, typename X2>
#define MATH21_KERNEL_EXPORT __global__
#define MATH21_KERNEL_GLOBAL
#define MATH21_KERNEL_INPUT_ID
#define MATH21_KERNEL_INPUT_OFFSETS_XY
#define MATH21_DEVICE_MAKE_F_LIKE_PTR(X, Y) X Y,

#elif defined(MATH21_IS_FROM_OPENCL)
#define MATH21_MAKE_KERNEL_NAME_SUFFIX_1(X, Y1) MATH21_OPENCL_TEMPLATE_3(X, opencl_kernel, Y1)
#define MATH21_MAKE_KERNEL_NAME_SUFFIX_2(X, Y1, Y2) MATH21_OPENCL_TEMPLATE_4(X, opencl_kernel, Y1, Y2)
#define MATH21_MAKE_KERNEL_NAME_SUFFIX_3(X, Y1, Y2, Y3) MATH21_OPENCL_TEMPLATE_5(X, opencl_kernel, Y1, Y2, Y3)
#define MATH21_KERNEL_GET_ID() size_t global_x = get_global_id(0); size_t global_y = get_global_id(1); size_t global_z = get_global_id(2); size_t id = global_z * get_global_size(0) * get_global_size(1) + global_y * get_global_size(0) + global_x; id +=1;
#define MATH21_KERNEL_TEMPLATE_HEADER_1(X)
#define MATH21_KERNEL_TEMPLATE_HEADER_2(X1, X2)
#define MATH21_KERNEL_EXPORT __kernel
#define MATH21_KERNEL_GLOBAL __global
#define MATH21_KERNEL_INPUT_ID
#define MATH21_KERNEL_INPUT_OFFSETS_XY , NumN offset_x, NumN offset_y
#define MATH21_DEVICE_MAKE_F_LIKE_PTR(X, Y)

#else
#error MATH21_IS_FROM_NONE
#endif

#if defined(MATH21_IS_FROM_OPENCL)
#include <math21_opencl_device_code.h>
#endif