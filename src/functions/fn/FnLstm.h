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
#include "FnDropout.h"

namespace math21 {
    // lstm
    // h(t) = lstm(x(t), h(t-1)), t = 1, ..., T.
    // h(t), c(t) = f_lstm(x(t), h(t-1), c(t-1)), t = 1, ..., T.
    class FnLstm {
    public:
        const char* name;
        int implementationMode; // 1, 2, 3
        int inputs; // x_size, no batch
        int outputs; // y_size, no batch
        int batch; // rnn_batch_size, mini_batch_size

        // for input x
        FnFullyConnected *fcWi;
        FnFullyConnected *fcWf;
        FnFullyConnected *fcWo;
        FnFullyConnected *fcWg;

        FnFullyConnected *fcWx; // implementationMode = 2

        // for hidden h
        FnFullyConnected *fcUi;
        FnFullyConnected *fcUf;
        FnFullyConnected *fcUo;
        FnFullyConnected *fcUg;

        FnFullyConnected *fcUh; // implementationMode = 2

        FnFullyConnected *fcW; // implementationMode = 3
        PtrR32Wrapper xh_interleaved; // implementationMode = 3
        PtrR32Wrapper dxh_interleaved; // implementationMode = 3

        PtrR32Wrapper delta; // dL/dY
        PtrR32Wrapper output; // Y, here is h(t), t = 1, ..., T.
        PtrR32Wrapper last_output; // Y, here is h(T)

        PtrR32Wrapper h_0;
        PtrR32Wrapper c_0;

        // temp
        PtrR32Wrapper temp;
        PtrR32Wrapper dc_t;

        PtrR32Wrapper i;
        PtrR32Wrapper f;
        PtrR32Wrapper o;
        PtrR32Wrapper g;

        PtrR32Wrapper d_i; // implementationMode = 2
        PtrR32Wrapper d_f; // implementationMode = 2
        PtrR32Wrapper d_o; // implementationMode = 2
        PtrR32Wrapper d_g; // implementationMode = 2
        PtrR32Wrapper ifog_interleaved; // implementationMode = 2
        PtrR32Wrapper difog_interleaved; // implementationMode = 2
        PtrR32Wrapper ifog_noninterleaved; // implementationMode = 2
        PtrR32Wrapper difog_noninterleaved; // implementationMode = 2

        PtrR32Wrapper c; // current c
        PtrR32Wrapper dc_tm1_at_t;
        PtrR32Wrapper cell; // here is c(t), t = 1, ..., T.
        PtrR32Wrapper h; // current hidden state h

        int steps; // n_time_step
        int i_time_step; // time in rnn. todo: check relationship with set_mbs

        NumB is_dropout_x;
        NumB is_dropout_h;
        FnDropout *dropout_x;
        FnDropout *dropout_h;
        int is_return_sequences;

        FnMatType y; // Y, with batch
        FnMatType dy; // dL/dY
    public:
        FnLstm();

        virtual ~FnLstm();

        void init();

        void create(
                int batch, int input_size, int output_size, int n_time_step, int is_use_bias,
                int is_batch_normalize, int is_unit_forget_bias, float dropout_rate_x, float dropout_rate_h, int is_adam,
                int is_return_sequences, int implementationMode);

        void log(const char *varName) const;

        void forward(mlfunction_node *finput, int is_train);

        void backward(mlfunction_node *finput0, int is_train);

        void update(OptUpdate *optUpdate);

        void saveState(FILE *file) const;

        void resetState(int b);

        void increaseByTime(int time_steps);

        void reset();

        void setMbs(int mini_batch_size);

        void loadThetaOrderBwsmvFlipped(FILE *fp, int flipped);

        void saveThetaOrderBwsmv(FILE *fp);

        void saveTheta(FILE *fp);

        void loadTheta(FILE *fp);
    };
}
