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
#include "FnBatchnorm.h"

namespace math21 {
    class FnFullyConnected {
    public:
        const char *name;
        float learning_rate_scale;
        int inputs; // x_size, no batch
        int outputs; // y_size, no batch
        int batch; // mini_batch_size
        int h, w, c; // nr_X, nc_X, nch_X
        int out_h, out_w, out_c; // nr_Y, nc_Y, nch_Y
        PtrR32Wrapper delta; // dL/dY
        PtrR32Wrapper output; // Y, shape: n_time_step * mbs * nr_Y * nc_Y * nch_Y
        PtrR32Wrapper weight_updates; // dL/dW
        PtrR32Wrapper bias_updates; // dL/db
        PtrR32Wrapper weights; // W
        float *weights_cpu; // W
        PtrR32Wrapper biases; // b
        float *biases_cpu; // b
        int is_use_bias; // applied only to bias when not using batch normalization.

        PtrR32Wrapper m;
        PtrR32Wrapper v;
        PtrR32Wrapper bias_m;
        PtrR32Wrapper bias_v; // todo: remove when is_use_bias is 0

        int nweights;
        MATH21_FUNCTION_ACTIVATION_TYPE activation;
        FnBatchnorm *bn;
        int flipped; // now just kept for loading dk paras.

        int total_mbs; // n_time_step * mini_batch_size, created in memory
        int n_time_step;
        int i_time_step; // time in rnn. todo: check relationship with set_mbs

        FnMatType y; // Y, with batch
        FnMatType dy; // dL/dY
    public:
        FnFullyConnected();

        FnFullyConnected(
                int rnn_batch_size, int n_time_step, int input_size, int output_size,
                MATH21_FUNCTION_ACTIVATION_TYPE activation, int is_use_bias, int is_batch_normalize, int is_adam,
                const char *name);

        virtual ~FnFullyConnected();

        void init();

        void create(
                int batch_size, int input_size, int output_size,
                MATH21_FUNCTION_ACTIVATION_TYPE activation, int is_use_bias, int is_batch_normalize, int is_adam,
                const char *name);

        void createWithNTimeStep(
                int rnn_batch_size, int n_time_step, int input_size, int output_size,
                MATH21_FUNCTION_ACTIVATION_TYPE activation, int is_use_bias, int is_batch_normalize, int is_adam,
                const char *name);

        void resize(const mlfunction_net *net);

        void log(const char *varName) const;

        void forward(mlfunction_node *finput, int is_train);

        void backward(mlfunction_node *finput, int is_train);

        void update(OptUpdate *optUpdate);

        void saveState(FILE *file) const;

        void increaseByTime(int time_steps);

        void reset();

        void setMbs(int mini_batch_size);

        void mergeTo(FnFullyConnected *fb);

        void scale(float s);

        void pullWrapper(NumB useRolling);

        void pushWrapper(NumB useRolling);

        void pushByWrapper(FnFullyConnected *fb, NumB useRolling);

        void saveThetaOrderBwsmv(FILE *fp);

        void loadThetaOrderBwsmv(FILE *fp);

        void loadThetaOrderBwsmvFlipped(FILE *fp, int flipped);

        void saveTheta(FILE *fp);

        void loadTheta(FILE *fp);
    };
}