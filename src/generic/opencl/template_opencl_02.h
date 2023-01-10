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
#include "../../algebra/files.h"
#include "common.h"

#ifdef MATH21_FLAG_USE_OPENCL

namespace math21 {
    const static std::string kernel_file = "generic_02.kl";
// todo: use 'to string' method instead of splitting kl files.
//const static std::string kernel_file_f_bc_add_like = "generic_02_f_bc_add_like.kl";
// todo: reduce the number of programs.
    static Map_<std::string, std::shared_ptr<m21clprogram>> thePrograms;
    Map_<std::string, std::shared_ptr<m21clprogram>> programsForfshrink;
    Map_<std::string, std::shared_ptr<m21clprogram>> programs_tensor_f_inner_product_like_shrink;
    Map_<std::string, std::shared_ptr<m21clprogram>> programs_f_bc_add_like_ptr;
    Map_<std::string, std::shared_ptr<m21clprogram>> programs_f_bc_sin_like_ptr;
    Map_<std::string, std::shared_ptr<m21clprogram>> programs_f_kx_like_ptr;

// todo: make program file small
// one kernel one program
    template<typename T>
    void math21_template_tensor_f_shrink_opencl(NumN fname, NumN n, PtrVoidInWrapper x, PtrVoidWrapper y,
                                                NumN dims_x, PtrNInWrapper dx, NumN dims_y,
                                                PtrNInWrapper dy,
                                                NumN nb, PtrNInWrapper b,
                                                NumN nv, NumN dims_v, PtrNInWrapper dv) {
        std::string d_function_ptr = "f_shrink_min_like_ptr";
        std::string d_function_name;
        if (fname == m21_fname_sum) {
            d_function_name = "math21_device_f_sum";
        } else if (fname == m21_fname_mean) {
            d_function_name = "math21_device_f_mean";
        } else if (fname == m21_fname_max) {
            d_function_name = "math21_device_f_max";
        } else if (fname == m21_fname_min) {
            d_function_name = "math21_device_f_min";
        } else if (fname == m21_fname_argmax) {
            d_function_ptr = "f_shrink_argmin_like_ptr";
            d_function_name = "math21_device_f_argmax";
        } else if (fname == m21_fname_argmin) {
            d_function_ptr = "f_shrink_argmin_like_ptr";
            d_function_name = "math21_device_f_argmin";
        } else {
            MATH21_ASSERT(0, "not support calling function with name " << fname);
        }

        cl_kernel kernel = math21_opencl_kernel_get<T>(
                "math21_template_tensor_f_shrink_opencl_kernel", kernel_file,
                programsForfshrink, d_function_ptr, d_function_name);
        math21_opencl_kernel_arg_set(kernel, n, x, y, dims_x, dx, dims_y, dy, nb, b, nv, dims_v, dv);
        math21_opencl_kernel_run(kernel, n);
    }

    template<typename T>
    void math21_template_tensor_f_inner_product_like_shrink_opencl(NumN fname, NumN n,
                                                                   PtrVoidInWrapper x1, PtrVoidInWrapper x2,
                                                                   PtrVoidWrapper y,
                                                                   NumN dims_x, PtrNInWrapper dx, NumN dims_y,
                                                                   PtrNInWrapper dy,
                                                                   NumN nb, PtrNInWrapper b,
                                                                   NumN nv, NumN dims_v, PtrNInWrapper dv) {
//    x1 -= 1;
//    x2 -= 1;
//    y -= 1;
//    dx -= 1;
//    dy -= 1;
//    b -= 1;
//    dv -= 1;
        std::string d_function_ptr = "f_inner_product_like_ptr";
        std::string f;
        if (fname == m21_fname_inner_product) {
            f = "math21_device_f_inner_product";
        } else if (fname == m21_fname_distance_1) {
            f = "math21_device_f_distance_1";
        } else if (fname == m21_fname_distance_2_square) {
            f = "math21_device_f_distance_2_square";
        } else {
            MATH21_ASSERT(0, "not support calling function with name " << fname);
        }

        cl_kernel kernel = math21_opencl_kernel_get<T>(
                "math21_template_tensor_f_inner_product_like_shrink_opencl_kernel", kernel_file,
                programs_tensor_f_inner_product_like_shrink, d_function_ptr, f);
        math21_opencl_kernel_arg_set(
                kernel, n, x1, x2, y,
                dims_x, dx, dims_y, dy, nb, b, nv, dims_v, dv);
        math21_opencl_kernel_run(kernel, n);
    }

    template<typename T>
    void math21_template_tensor_f_inner_product_like_bcshrink_opencl(NumN fname, NumN n,
                                                                     PtrVoidInWrapper x1, PtrVoidInWrapper x2,
                                                                     PtrVoidWrapper y,
                                                                     NumN dims_x1, PtrNInWrapper dx1,
                                                                     NumN dims_x2, PtrNInWrapper dx2,
                                                                     NumN dims_x, PtrNInWrapper dx,
                                                                     NumN dims_y, PtrNInWrapper dy,
                                                                     NumN nb, PtrNInWrapper b,
                                                                     NumN nv, NumN dims_v, PtrNInWrapper dv) {
//    x1 -= 1;
//    x2 -= 1;
//    y -= 1;
//    dx1 -= 1;
//    dx2 -= 1;
//    dx -= 1;
//    dy -= 1;
//    b -= 1;
//    dv -= 1;
        std::string d_function_ptr = "f_inner_product_like_ptr";
        std::string f;
        if (fname == m21_fname_inner_product) {
            f = "math21_device_f_inner_product";
        } else if (fname == m21_fname_distance_1) {
            f = "math21_device_f_distance_1";
        } else if (fname == m21_fname_distance_2_square) {
            f = "math21_device_f_distance_2_square";
        } else {
            MATH21_ASSERT(0, "not support calling function with name " << fname);
        }

        cl_kernel kernel = math21_opencl_kernel_get<T>(
                "math21_template_tensor_f_inner_product_like_bcshrink_opencl_kernel", kernel_file,
                programs_tensor_f_inner_product_like_shrink, d_function_ptr, f);
        math21_opencl_kernel_arg_set(
                kernel, n, x1, x2, y,
                dims_x1, dx1, dims_x2, dx2,
                dims_x, dx, dims_y, dy, nb, b, nv, dims_v, dv);
        math21_opencl_kernel_run(kernel, n);
    }

// todo: use index 1 for x, y
// a special kind of sub
// x is sub-tensor of y
    template<typename T>
    void math21_template_tensor_f_with_broadcast_in_dn_opencl(NumN fname, NumN n,
                                                              PtrVoidInWrapper x1,
                                                              PtrVoidInWrapper x2,
                                                              PtrVoidWrapper y,
                                                              NumN dims_x1, PtrNInWrapper dx1,
                                                              NumN dims_x2, PtrNInWrapper dx2,
                                                              NumN dims_y, PtrNInWrapper dy) {
//    x1 -= 1;
//    x2 -= 1;
//    y -= 1;
//    dx1 -= 1;
//    dx2 -= 1;
//    dy -= 1;
        std::string d_function_ptr = "f_bc_add_like_ptr";
        std::string f;
        if (fname == m21_fname_add) {
            f = "math21_device_f_add";
        } else if (fname == m21_fname_subtract) {
            f = "math21_device_f_subtract";
        } else if (fname == m21_fname_multiply) {
            f = "math21_device_f_multiply";
        } else if (fname == m21_fname_divide) {
            f = "math21_device_f_divide";
        } else if (fname == m21_fname_ele_is_equal) {
            f = "math21_device_f_is_equal";
        } else if (fname == m21_fname_ele_is_less_than) {
            f = "math21_device_f_is_less_than";
        } else if (fname == m21_fname_ele_is_not_less_than) {
            f = "math21_device_f_is_not_less_than";
        } else if (fname == m21_fname_set_using_mask) {
        } else {
            MATH21_ASSERT(0, "not support calling function with name " << fname);
        }
        cl_kernel kernel;
        if (fname == m21_fname_set_using_mask) {
            kernel = math21_opencl_kernel_get<T>(
                    "math21_template_tensor_set_using_mask_in_dn_opencl_kernel", kernel_file, thePrograms);
        } else {
            kernel = math21_opencl_kernel_get<T>(
                    "math21_template_tensor_f_with_broadcast_in_dn_opencl_kernel", kernel_file,
                    programs_f_bc_add_like_ptr, d_function_ptr, f);
        }
        math21_opencl_kernel_arg_set(
                kernel, n, x1, x2, y, dims_x1, dx1, dims_x2, dx2, dims_y, dy);
        math21_opencl_kernel_run(kernel, n);
    }

// todo: use index 1 for x, y
    template<typename T>
    void math21_template_vector_f_add_like_opencl(NumN fname, NumN n,
                                                  PtrVoidInWrapper x1,
                                                  PtrVoidInWrapper x2,
                                                  PtrVoidWrapper y) {
//    x1 -= 1;
//    x2 -= 1;
//    y -= 1;

        std::string d_function_ptr = "f_bc_add_like_ptr";
        std::string f;
        if (fname == m21_fname_add) {
            f = "math21_device_f_add";
        } else if (fname == m21_fname_subtract) {
            f = "math21_device_f_subtract";
        } else if (fname == m21_fname_multiply) {
            f = "math21_device_f_multiply";
        } else if (fname == m21_fname_divide) {
            f = "math21_device_f_divide";
        } else if (fname == m21_fname_ele_is_equal) {
            f = "math21_device_f_is_equal";
        } else if (fname == m21_fname_ele_is_less_than) {
            f = "math21_device_f_is_less_than";
        } else if (fname == m21_fname_ele_is_not_less_than) {
            f = "math21_device_f_is_not_less_than";
        } else if (fname == m21_fname_set_using_mask) {
        } else {
            MATH21_ASSERT(0, "not support calling function with name " << fname);
        }
        cl_kernel kernel;
        if (fname == m21_fname_set_using_mask) {
            kernel = math21_opencl_kernel_get<T>(
                    "math21_template_vector_set_using_mask_opencl_kernel", kernel_file, thePrograms);
        } else {
            kernel = math21_opencl_kernel_get<T>(
                    "math21_template_vector_f_add_like_opencl_kernel", kernel_file,
                    programs_f_bc_add_like_ptr, d_function_ptr, f);
        }
        math21_opencl_kernel_arg_set(kernel, n, x1, x2, y);
        math21_opencl_kernel_run(kernel, n);
    }

    template<typename T>
    void math21_template_vector_f_sin_like_opencl(NumN fname, NumN n,
                                                  PtrVoidInWrapper x, PtrVoidWrapper y) {
//    x -= 1;
//    y -= 1;

        std::string d_function_ptr = "f_bc_sin_like_ptr";
        std::string f;
        if (fname == m21_fname_sin) {
            f = "math21_device_f_sin";
        } else if (fname == m21_fname_cos) {
            f = "math21_device_f_cos";
        } else if (fname == m21_fname_tan) {
            f = "math21_device_f_tan";
        } else if (fname == m21_fname_exp) {
            f = "math21_device_f_exp";
        } else if (fname == m21_fname_log) {
            f = "math21_device_f_log";
        } else if (fname == m21_fname_abs) {
            f = "math21_device_f_abs";
        } else {
            MATH21_ASSERT(0, "not support calling function with name " << fname);
        }

        cl_kernel kernel = math21_opencl_kernel_get<T>(
                "math21_template_vector_f_sin_like_opencl_kernel", kernel_file,
                programs_f_bc_sin_like_ptr, d_function_ptr, f);
        math21_opencl_kernel_arg_set(kernel, n, x, y);
        math21_opencl_kernel_run(kernel, n);
    }

    template<typename T>
    void math21_template_vector_f_kx_like_opencl(NumN fname, NumN n, T k,
                                                 PtrVoidInWrapper x, PtrVoidWrapper y) {
//    x -= 1;
//    y -= 1;
        std::string d_function_ptr = "f_kx_like_ptr";
        std::string f;
        if (fname == m21_fname_kx_add) {
            f = MATH21_STRINGIFY(math21_device_f_add);
        } else if (fname == m21_fname_kx_subtract) {
            f = MATH21_STRINGIFY(math21_device_f_subtract);
        } else if (fname == m21_fname_xk_subtract) {
            f = MATH21_STRINGIFY(math21_device_f_xk_subtract);
        } else if (fname == m21_fname_kx_mul) {
            f = MATH21_STRINGIFY(math21_device_f_multiply);
        } else if (fname == m21_fname_kx_divide) {
            f = MATH21_STRINGIFY(math21_device_f_divide);
        } else if (fname == m21_fname_xk_divide) {
            f = MATH21_STRINGIFY(math21_device_f_xk_divide);
        } else if (fname == m21_fname_kx_pow) {
            f = MATH21_STRINGIFY(math21_device_f_kx_pow);
        } else if (fname == m21_fname_xk_pow) {
            f = MATH21_STRINGIFY(math21_device_f_xk_pow);
        } else {
            MATH21_ASSERT(0, "not support calling function with name " << fname);
        }
        cl_kernel kernel = math21_opencl_kernel_get<T>(
                "math21_template_vector_f_kx_like_opencl_kernel", kernel_file,
                programs_f_kx_like_ptr, d_function_ptr, f);
        math21_opencl_kernel_arg_set(kernel, n, k, x, y);
        math21_opencl_kernel_run(kernel, n);
    }
}

#endif