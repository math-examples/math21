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
    class FnMaxPooling {
    public:
        const char *name;
        int batch; // mini_batch_size
        int h;
        int w;
        int c;
        int out_h;
        int out_w;
        int out_c;
        int padding;
        int size;
        int stride;
        PtrIntWrapper indexes;
        int outputs; // l.out_h*l.out_w*l.out_c;
        int inputs; // l.out_h*l.out_w*l.out_c;
        PtrR32Wrapper output; // Y
        PtrR32Wrapper delta; // dL/dY

        FnMatType y; // Y, with batch
        FnMatType dy; // dL/dY
    public:
        FnMaxPooling();

        virtual ~FnMaxPooling();

        void init();

        void create(int batch, int c, int h, int w,
                    int size, int stride, int padding);

        void resize(int h, int w);

        void log(const char *varName) const;

        void forward(const mlfunction_node *finput, int is_train);

        void backward(mlfunction_node *finput);

        void saveState(FILE *file) const;
    };

}