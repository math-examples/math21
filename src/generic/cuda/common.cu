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

namespace math21 {

    M21_EXPT_DEVICE math21_type_f_addto_like math21_device_f_addto_p = &math21_device_f_addto;
    M21_EXPT_DEVICE math21_type_f_addto_like math21_device_f_multo_p = &math21_device_f_multo;

    m21cudaCallableFunctionPointer<math21_type_f_addto_like>
    math21_generic_fname_to_f_addto_like_name_cuda(NumN fname) {
        m21cudaCallableFunctionPointer<math21_type_f_addto_like> f;
        if (fname == m21_fname_addto) {
            f.set(&math21_device_f_addto_p);
        } else if (fname == m21_fname_multo) {
            f.set(&math21_device_f_multo_p);
        } else {
            MATH21_ASSERT(0, "not support calling function with name " << fname);
        }
        return f;
    }
}