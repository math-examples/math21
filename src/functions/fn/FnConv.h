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

    class FnConv {
    public:
        const char *name;
        int nr_X, nc_X, nch_X; // nr_X, nc_X, nch_X
        int nr_Y, nc_Y, nch_Y; // nr_Y, nc_Y, nch_Y
        int n_group; // num_group. Different groups share same W.
        int batch; // mini_batch_size
        int stride; // stride
        int k_size; // k_size
        int pad; // padding
        int x_size; // x_size, no batch
        int y_size; // y_size, no batch
        int n_W; // n_W
        PtrR32Wrapper W; // W
        float *W_cpu; // W
        PtrR32Wrapper dW; // dL/dW
        PtrR32Wrapper b; // b
        float *b_cpu; // b
        PtrR32Wrapper db; // dL/db
        PtrR32Wrapper dY; // dL/dY
        PtrR32Wrapper Y; // Y
        size_t workspace_size; // X_prime_size in convolution. shape: (k1_size*k2_size*nch_X)*(nr_Y*nc_Y)
        MATH21_FUNCTION_ACTIVATION_TYPE activation;
        FnBatchnorm *bn;

        // adam
        PtrR32Wrapper m;
        PtrR32Wrapper v;
        PtrR32Wrapper bias_m;
        PtrR32Wrapper bias_v;

        float clip; // constant k, so -k<= x <= k, used only clip != 0. default k = 0
        float learning_rate_scale;

        int flipped; // transpose, used in load weight.
        void *detail;
        mlfunction_node *fnode; // for debug only

        FnMatType y; // Y, with batch
        FnMatType dy; // dL/dY
    public:
        FnConv();

        virtual ~FnConv();

        void init();

        void create(
                int mini_batch_size, int nr_X, int nc_X, int nch_X, int nch_Y,
                int num_group, int k_size, int stride, int padding, MATH21_FUNCTION_ACTIVATION_TYPE activation,
                int is_batch_normalize, int adam);

        void resize(int nr_X, int nc_X);

        void forward(const mlfunction_node *finput0,
                     int is_train, PtrR32Wrapper workspace);

        void backward(mlfunction_node *finput,
                      int is_train, PtrR32Wrapper workspace);

        void update(OptUpdate *optUpdate);

        void saveState(FILE *file) const;

        void mergeTo(FnConv *fb);

        void scale(float s);

        void pullWrapper(NumB useRolling);

        void pushWrapper(NumB useRolling);

        void pushByWrapper(FnConv *fb, NumB useRolling);

        void set_mbs(int mini_batch_size);

        void saveTheta(FILE *fp);

        void loadTheta(FILE *fp);

        void log(const char *varName) const;

        const void *getDataToCpu(const char *varName);

        m21rawtensor getRawTensorToCpu(const char *varName);
    private:
        static int cal_nr_or_nc_Y(int nr_X, int pad, int size, int stride);

        static size_t get_X_prime_size(int nr_Y, int nc_Y,
                                       int k_size, int nch_X, int num_group);
    };
}