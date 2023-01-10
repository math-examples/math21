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
#include "../../gpu/files.h"

namespace math21 {
    // not support overloaded kernel
    template<typename KernelType, typename... Ts>
    void math21_cuda_kernel_arg_set_and_run(
            NumN n, KernelType kernel, const Ts &... args) {
        kernel <<< math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >>>(args...);
        math21_cuda_check_error(cudaPeekAtLastError());
    }

#define math21_cuda_kernel_arg_set_and_run_overload(\
            n, kernel, arg, ...) \
        kernel <<< math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >>>(arg, __VA_ARGS__); \
        math21_cuda_check_error(cudaPeekAtLastError())

    // https://stackoverflow.com/questions/15644261/cuda-function-pointers
    template<typename T>
    struct m21cudaCallableFunctionPointer {
    public:
        m21cudaCallableFunctionPointer() {
            ptr = NULL;
        }

        void set(T *f_) {
            // copy the function pointers to their host equivalent
            cudaMemcpyFromSymbol(&ptr, *f_, sizeof(T));
        }

        T ptr;
    };

    m21cudaCallableFunctionPointer<math21_type_f_addto_like>
    math21_generic_fname_to_f_addto_like_name_cuda(NumN fname);

}