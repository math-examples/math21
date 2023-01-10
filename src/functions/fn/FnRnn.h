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
#include "FnFullyConnected.h"

namespace math21 {
    class FnRnn {
    public:
        const char *name;
        int inputs; // x_size, no batch
        int outputs; // y_size, no batch
        int batch; // mini_batch_size

        int steps;

        FnFullyConnected *input_layer;
        FnFullyConnected *self_layer;
        FnFullyConnected *output_layer;

        PtrR32Wrapper state;
        PtrR32Wrapper prev_state; // only used by backward
        PtrR32Wrapper delta; // dL/dY
        PtrR32Wrapper output; // Y

        FnMatType y; // Y, with batch
        FnMatType dy; // dL/dY
    public:
        FnRnn();

        virtual ~FnRnn();

        void init();

        void create(int batch_size, int input_size, int output_size,
                    int n_time_step, MATH21_FUNCTION_ACTIVATION_TYPE activation, int is_use_bias,
                    int is_batch_normalize,
                    int is_adam);

        void resize(const mlfunction_net *net);

        void log(const char *varName) const;

        void reset();

        void forward(mlfunction_node *finput, int is_train);

        void backward(mlfunction_node *finput, int is_train);

        void update(OptUpdate *optUpdate);

        void saveState(FILE *file) const;

        void resetState(int b);

        void setMbs(int mini_batch_size);

        void saveThetaOrderBwsmv(FILE *fp);

        void loadThetaOrderBwsmvFlipped(FILE *fp, int flipped);

        void saveTheta(FILE *fp);

        void loadTheta(FILE *fp);
    };
}