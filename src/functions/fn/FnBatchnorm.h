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

    // use _cpu only when not cpu.
    class FnBatchnorm {
    public:
        const char *name;
        int is_this_type;
        int is_train; // is train
        int outputs; // y_size, no batch
        int mini_batch_size; // mini_batch_size
        PtrR32Wrapper output; // Y
        PtrR32Wrapper x; // ?
//    int features_size;
        int out_c; // features_size
        int in_class_size;
        PtrR32Wrapper mean;
        PtrR32Wrapper variance;
        PtrR32Wrapper mean_delta;
        PtrR32Wrapper variance_delta;
        PtrR32Wrapper rolling_mean;
        float *rolling_mean_cpu;
        PtrR32Wrapper rolling_variance;
        float *rolling_variance_cpu;
        PtrR32Wrapper x_norm;
        PtrR32Wrapper biases; // b
        float *biases_cpu; // when not cpu
        PtrR32Wrapper bias_updates; // dL/db
        PtrR32Wrapper scales;
        float *scales_cpu;
        PtrR32Wrapper scale_updates;

        PtrR32Wrapper delta; // dL/dY. If not owned, it will not be reset when forward in train.
        int out_h, out_w; // nr_Y, nc_Y, nch_Y
        int h, w, c; // nr_X, nc_X, nch_X
        int inputs; // x_size, no batch

        // adam
        PtrR32Wrapper bias_m;
        PtrR32Wrapper bias_v;
        PtrR32Wrapper scale_m;
        PtrR32Wrapper scale_v;

        float learning_rate_scale;

        int total_mbs; // n_time_step * mini_batch_size, created in memory
        int n_time_step;
        int i_time_step; // time in rnn

        FnMatType y; // Y, with batch
        FnMatType dy; // dL/dY
    public:
        FnBatchnorm();

        virtual ~FnBatchnorm();

        void init();

        void create(
                int is_this_type, mlfunction_node *finput,
                int mini_batch_size, int nc_Y, int nr_Y, int nch_Y, int adam);

        void resize(const mlfunction_node *fnode, int nc_Y, int nr_Y);

        void log(const char *varName) const;

        void forward(mlfunction_node *finput);

        void backward(mlfunction_node *finput);

        void update(OptUpdate *optUpdate);

        void saveTheta(FILE *fp, int isPull);

        void loadTheta(FILE *fp, int isPush);

        void saveState(FILE *file) const;

        void mergeTo(FnBatchnorm *fb);

        void scale(float s);

        void pullWrapper(NumB useRolling);

        void pushWrapper(NumB useRolling);

        void pushByWrapper(FnBatchnorm *fb, NumB useRolling);

        void increaseByTime(int time_steps);

        void reset();
    };

}