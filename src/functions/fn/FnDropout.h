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
    class FnDropout {
    public:
        const char *name;
        int batch; // mini_batch_size
        int y_dim[MATH21_DIMS_RAW_TENSOR];
        int outputs; // l.out_h*l.out_w*l.out_c;
        int inputs; // l.out_h*l.out_w*l.out_c;

        PtrR32Wrapper y; // Y
        PtrR32Wrapper dy; // dL/dY

        // With probability `rate`, drops elements of `x`. Input that are kept are
        //  scaled up by `scale = 1 / (1 - rate)`, otherwise outputs `0`.  The scaling is so that
        //  the expected sum is unchanged.
        float rate; // The probability that each element is dropped. For example, setting rate=0.1 would drop 10% of input elements.
        float scale; // tf.nn.dropout
        PtrR32Wrapper rand; // rand(i) in [0, 1], (Note: it is independent of t when in rnn,
        // and in this case its size is 1/T of usual)

        int total_mbs; // n_time_step * mini_batch_size, created in memory
        int n_time_step;
        int i_time_step; // time in rnn.

//        FnMatType y; // Y, with batch
//        FnMatType dy; // dL/dY
    public:
        FnDropout();

        FnDropout(mlfunction_node *finput, float rate, int n_time_step, const char *name);

        virtual ~FnDropout();

        void init();

        void create(mlfunction_node *finput, float rate, int n_time_step, const char *name);

        void resize(const mlfunction_net *net);

        void log(const char *varName) const;

        void forward(mlfunction_node *finput, int is_train);

        void backward(mlfunction_node *finput);

        void saveState(FILE *file) const;

        void increaseByTime(int time_steps);

        void reset();
    };
}