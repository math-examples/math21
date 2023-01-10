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
    class FnSoftmax {
    public:
        const char* name;
        int batch; // mini_batch_size
        int groups; // num_group. Different groups share same W.
        int inputs; // x_size, no batch
        int outputs; // y_size, no batch
        PtrR32Wrapper loss;
        float *loss_cpu;
        PtrR32Wrapper delta; // dL/dY
        PtrR32Wrapper output; // Y
        float *cost;

        m21tree *softmax_tree;
        float temperature;
        int spatial;
        int noloss;
        int h, w, c; // nr_X, nc_X, nch_X

        FnMatType y; // Y, with batch
        FnMatType dy; // dL/dY
    public:
        FnSoftmax();

        virtual ~FnSoftmax();

        void init();

        void create(const mlfunction_node *finput, int groups);

        void resize(const mlfunction_net *net);

        void log(const char *varName) const;

        void forward(mlfunction_net *net, mlfunction_node *finput);

        void backward(mlfunction_net *net, mlfunction_node *finput);

        void saveState(FILE *file) const;

        static void netSetTemperature(mlfunction_net *fnet, float t);
    };
}