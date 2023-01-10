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
#include "basic/files.h"

namespace math21 {

    struct OptParasAdam {
    public:
        NumN num_iters;
        NumR step_size;
        NumR b1;
        NumR b2;
        NumR eps;

        OptParasAdam() {
            num_iters = 100;
            step_size = 0.001;
            b1 = 0.9;
            b2 = 0.999;
            eps = 1e-8;
        }
    };

    class OptCallbackAdam {
    public:
        virtual ~OptCallbackAdam() {}

        virtual NumB compute(const TenR &x, TenR &dy, NumN iter) const = 0;
    };

    class OptPrintCallbackAdam {
    public:
        virtual ~OptPrintCallbackAdam() {}

        virtual void log(const TenR &x, const TenR &dy, NumN iter) const = 0;
    };

    void math21_opt_adam(
            TenR &x,
            void (*grad)(const TenR &x, TenR &dy, NumN iter, void *grad_data),
            void *grad_data,
            void (*callback)(const TenR &x_cur, const TenR &gradient, NumN iter, void *callback_data),
            void *callback_data,
            const OptParasAdam &parasAdam);

    void math21_opt_adam(
            TenR &x, const OptParasAdam &parasAdam, const OptCallbackAdam &gradCb, const OptPrintCallbackAdam *printCb);

    OptDetail *math21_opt_create_adam(const OptParasAdam &parasAdam, const OptCallbackAdam &gradCb,
                                      const OptPrintCallbackAdam *printCb);

    void math21_opt_destroy_adam(OptDetail *detail) ;
}