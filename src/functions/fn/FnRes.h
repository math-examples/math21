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

#include "Fn.h"

namespace math21 {
    class FnRes {
    public:
        const char *name;
        int mini_batch_size;
        FnMatType y; // Y, with batch
        FnMatType dy; // dL/dY

        int h, w, c; // nr_X, nc_X, nch_X
        int out_h, out_w, out_c; // nr_Y, nc_Y, nch_Y
        int inputs; // x_size, no batch
        int outputs; // y_size, no batch
        int index;
        PtrR32Wrapper delta; // dL/dY
        PtrR32Wrapper output; // Y

        float k1, k2;
        MATH21_FUNCTION_ACTIVATION_TYPE activation;
    public:
        FnRes();

        virtual ~FnRes();

        void init();

        void create(int batch, int index,
                    int w, int h, int c, int w2, int h2,
                    int c2, NumN deviceType);

        void resize(int h, int w);

        void log(const char *varName) const;

        void forward(mlfunction_net *fnet, mlfunction_node *finput);

        void backward(mlfunction_net *fnet, mlfunction_node *finput);

        void saveState(FILE *file) const;
    };
}