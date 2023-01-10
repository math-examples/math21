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

#ifdef MATH21_FLAG_USE_OPENCL

namespace math21 {
    template<typename T>
    std::string math21_opencl_template_kernelNameSuffix(const std::string &d_functionName = "") {
        std::string typeName1 = math21_type_name<T>();
        std::string kernelName_suf = typeName1;
        if (!d_functionName.empty()) {
            kernelName_suf += "_" + d_functionName;
        }
        return kernelName_suf;
    }

    template<typename T1, typename T2>
    std::string math21_opencl_template_kernelNameSuffix(const std::string &d_functionName = "") {
        std::string typeName1 = math21_type_name<T1>();
        std::string typeName2 = math21_type_name<T2>();
        std::string kernelName_suf = typeName1 + "_" + typeName2;
        if (!d_functionName.empty()) {
            kernelName_suf += "_" + d_functionName;
        }
        return kernelName_suf;
    }

    std::string
    math21_opencl_template_kernelName_using_suffix(const std::string &functionName, const std::string &suffix);

    std::string math21_opencl_options_f_like(
            const std::string &d_function_ptr, const std::string &d_function_name);

    std::string math21_opencl_options_f_like2(
            const std::string &d_function_ptr, const std::string &d_function_name);

    template<typename T>
    std::string math21_opencl_options_for_program(
            const std::string &d_function_ptr = "", const std::string &d_function_name = "") {
        MATH21_ASSERT(sizeof(T) == 4 || sizeof(T) == 8, "Only NumR32 and NumR64 supported currently!");
        std::string options;
#ifdef MATH21_FLAG_NOT_EXTERNAL
        options += "-D MATH21_FLAG_NOT_EXTERNAL ";
#endif
        options += "-D NumReal=" + math21_type_name<T>() + " ";
        options += math21_opencl_options_f_like(d_function_ptr, d_function_name);
        options += "-I ";
        options += MATH21_INCLUDE_PATH;
        return options;
    }

// m21_type_NumN added, see math21_generic_tensor_f_with_broadcast_in_dn_opencl
    template<typename T1, typename T2>
    std::string math21_opencl_options_for_program(
            const std::string &d_function_ptr = "", const std::string &d_function_name = "") {
        MATH21_ASSERT(sizeof(T1) == 1 || sizeof(T1) == 4 || sizeof(T1) == 8,
                      "Only NumN8, NumR32 and NumR64 supported currently!");
        MATH21_ASSERT(sizeof(T2) == 1 || sizeof(T2) == 4 || sizeof(T2) == 8,
                      "Only NumN8, NumR32 and NumR64 supported currently!");
        std::string options;
#ifdef MATH21_FLAG_NOT_EXTERNAL
        options += "-D MATH21_FLAG_NOT_EXTERNAL ";
#endif
        options += "-D NumType1=" + math21_type_name<T1>() + " ";
        options += "-D NumType2=" + math21_type_name<T2>() + " ";
        options += math21_opencl_options_f_like2(d_function_ptr, d_function_name);
        options += "-I ";
        options += MATH21_INCLUDE_PATH;
        return options;
    }

    template<typename T>
    cl_kernel math21_opencl_kernel_get(
            const std::string &functionName,
            const std::string &kernel_file_name,
            Map_<std::string, std::shared_ptr<m21clprogram>> &map,
            const std::string d_function_ptr, const std::string d_function_name) {
        std::string kernelNameSuffix = math21_opencl_template_kernelNameSuffix<T>(d_function_name);
        if (!map.has(kernelNameSuffix)) {
            auto p = math21_opencl_build_program_from_file(
                    kernel_file_name,
                    math21_opencl_options_for_program<T>(
                            d_function_ptr, d_function_name));
            map.add(kernelNameSuffix, p);
        }
        std::string kernelName = math21_opencl_template_kernelName_using_suffix(functionName, kernelNameSuffix);
        cl_kernel kernel = math21_opencl_getKernel(map.valueAt(kernelNameSuffix), kernelName);
        return kernel;
    }

    template<typename T1, typename T2>
    cl_kernel math21_opencl_kernel_get(
            const std::string &functionName,
            const std::string &kernel_file_name,
            Map_<std::string, std::shared_ptr<m21clprogram>> &map,
            const std::string d_function_ptr, const std::string d_function_name) {
        std::string kernelNameSuffix = math21_opencl_template_kernelNameSuffix<T1, T2>(d_function_name);
        if (!map.has(kernelNameSuffix)) {
            auto p = math21_opencl_build_program_from_file(
                    kernel_file_name,
                    math21_opencl_options_for_program<T1, T2>(
                            d_function_ptr, d_function_name));
            map.add(kernelNameSuffix, p);
        }
        std::string kernelName = math21_opencl_template_kernelName_using_suffix(functionName, kernelNameSuffix);
        cl_kernel kernel = math21_opencl_getKernel(map.valueAt(kernelNameSuffix), kernelName);
        return kernel;
    }

    template<typename T1, typename T2>
    cl_kernel math21_opencl_kernel_get(
            const std::string &functionName,
            const std::string &kernel_file_name, Map_<std::string, std::shared_ptr<m21clprogram>> &map) {
        return math21_opencl_kernel_get<T1, T2>(functionName, kernel_file_name, map, "", "");
    }

    template<typename T>
    cl_kernel math21_opencl_kernel_get(
            const std::string &functionName,
            const std::string &kernel_file_name, Map_<std::string, std::shared_ptr<m21clprogram>> &map) {
        return math21_opencl_kernel_get<T>(functionName, kernel_file_name, map, "", "");
    }

    namespace detail {
        void math21_opencl_set_kernel_arg(
                cl_kernel &kernel, const SeqN &s, const Seqce<const void *> &v);

        void math21_opencl_kernel_arg_sv(SeqN &s, Seqce<const void *> &v);

        template<typename T>
        void imath21_opencl_kernel_arg_get_sv(
                SeqN &s, Seqce<const void *> &v, const T &x) {
            s.push(sizeof(x));
            v.push(&x);
        }

        template<>
        void imath21_opencl_kernel_arg_get_sv(SeqN &s, Seqce<const void *> &v, const m21clvector &x);

        template<typename T>
        void math21_opencl_kernel_arg_sv(SeqN &s, Seqce<const void *> &v, const T &x) {
            imath21_opencl_kernel_arg_get_sv(s, v, x);
        }

        // https://docs.microsoft.com/en-us/cpp/cpp/ellipses-and-variadic-templates
        template<typename T, typename... Ts>
        void math21_opencl_kernel_arg_sv(SeqN &s, Seqce<const void *> &v,
                                         const T &x, const Ts &... xs) {
            imath21_opencl_kernel_arg_get_sv(s, v, x);
            math21_opencl_kernel_arg_sv(s, v, xs...); // recursive call using pack expansion syntax
        }
    }

    template<typename... Ts>
    void math21_opencl_kernel_arg_set(cl_kernel &kernel, const Ts &... xs) {
        SeqN s;
        Seqce<const void *> v;
        detail::math21_opencl_kernel_arg_sv(s, v, xs...);
        detail::math21_opencl_set_kernel_arg(kernel, s, v);
    }

    void math21_opencl_kernel_run(cl_kernel kernel, NumN n);

    std::string math21_generic_fname_to_f_addto_like_ptr_opencl(NumN fname);

    std::string math21_generic_fname_to_f_addto_like_name_opencl(NumN fname);
}

#endif