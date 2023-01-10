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

#include "common.h"

#ifdef MATH21_FLAG_USE_OPENCL

namespace math21 {
    Map_<std::string, std::string> mapFLikeName;

    std::string math21_opencl_template_kernelName_using_suffix(const std::string &functionName, const std::string &suffix) {
        return functionName + "_" + suffix;
    }

    std::string math21_opencl_options_f_like(
            const std::string &d_function_ptr, const std::string &d_function_name) {
        if (mapFLikeName.isEmpty()) {
            mapFLikeName.add("f_shrink_min_like_ptr", "math21_device_f_min");
            mapFLikeName.add("f_shrink_argmin_like_ptr", "math21_device_f_argmin");
            mapFLikeName.add("f_bc_add_like_ptr", "math21_device_f_add");
            mapFLikeName.add("f_bc_sin_like_ptr", "math21_device_f_sin");
            mapFLikeName.add("f_kx_like_ptr", "math21_device_f_add");
            mapFLikeName.add("f_addto_like_ptr", "math21_device_f_addto");
            mapFLikeName.add("f_inner_product_like_ptr", "math21_device_f_inner_product");
            mapFLikeName.add("f_add_like_ptr", "math21_device_f_inner_product");
        }
//    else
        {
            if (!d_function_ptr.empty()) {
                MATH21_ASSERT(mapFLikeName.has(d_function_ptr));
                mapFLikeName.valueAt(d_function_ptr) = d_function_name;
            }
        }
        std::string options;
        auto &dataMap = mapFLikeName.getData();
        for (auto itr = dataMap.begin(); itr != dataMap.end(); ++itr) {
            options += "-D " + itr->first + "=" + itr->second + " ";
        }
        return options;
    }

    std::string math21_opencl_options_f_like2(
            const std::string &d_function_ptr, const std::string &d_function_name) {
        std::string options;
        if (!d_function_ptr.empty()) {
            options += "-D " + d_function_ptr + "=" + d_function_name + " ";
        }
        return options;
    }

    namespace detail {
        void math21_opencl_set_kernel_arg(
                cl_kernel &kernel, const SeqN &s, const Seqce<const void *> &v) {
            for (NumN i = 1; i <= s.size(); ++i) {
                math21_opencl_checkError(clSetKernelArg(kernel, i - 1, s(i), v(i)));
            }
        }

        void math21_opencl_kernel_arg_sv(SeqN &s, Seqce<const void *> &v) {
            MATH21_ASSERT(0);
        }

        // In a function template specialization, a template argument is optional if the compiler can deduce it from the type of the function arguments.
        template<>
        void imath21_opencl_kernel_arg_get_sv(
                SeqN &s, Seqce<const void *> &v, const m21clvector &x) {
            s.push(sizeof(x.buffer));
            v.push(&x.buffer);
        }
    }

    void math21_opencl_kernel_run(cl_kernel kernel, NumN n) {
        m21dim2 dim = math21_opencl_gridsize(n);
        size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

        cl_event e;
        cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                              NULL, &e);
        math21_opencl_checkError(error);
        math21_opencl_checkError(clWaitForEvents(1, &e));
        clReleaseEvent(e);
    }

    std::string math21_generic_fname_to_f_addto_like_ptr_opencl(NumN fname) {
        std::string d_function_ptr = "f_addto_like_ptr";
        return d_function_ptr;
    }

    std::string math21_generic_fname_to_f_addto_like_name_opencl(NumN fname) {
        std::string f;
        if (fname == m21_fname_addto) {
            f = "math21_device_f_addto";
        } else if (fname == m21_fname_multo) {
            f = "math21_device_f_multo";
        } else {
            MATH21_ASSERT(0, "not support calling function with name " << fname);
        }
        return f;
    }
}

#endif