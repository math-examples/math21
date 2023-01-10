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

typedef enum {
    mlfnode_cost_type_L2,
    mlfnode_cost_type_masked,
    mlfnode_cost_type_L1,
    mlfnode_cost_type_seg,
    mlfnode_cost_type_smooth,
    mlfnode_cost_type_wgan,
} mlfnode_cost_type;

namespace math21 {
    class FnCost {
    public:
        const char *name;
        int batch; // mini_batch_size
        int groups; // num_group. Different groups share same W.
        int inputs; // x_size, no batch
        int outputs; // y_size, no batch
        int y_dim[MATH21_DIMS_RAW_TENSOR];
        PtrR32Wrapper delta; // dL/dY
        PtrR32Wrapper output; // Y
        float *tmp_cpu; // Y
        float *cost;
        mlfnode_cost_type cost_type;
        float scale;
        float ratio;
        float noobject_scale;
        float thresh;
        float smooth;

        FnMatType y; // Y, with batch
        FnMatType dy; // dL/dY
    public:
        FnCost();

        virtual ~FnCost();

        void init();

        void create(const mlfunction_node *finput,
                    mlfnode_cost_type cost_type, float scale);

        void resize(int h, int w);

        void log(const char *varName) const;

        void forward(mlfunction_net *net, mlfunction_node *finput);

        void backward(mlfunction_net *net, mlfunction_node *finput);

        void saveState(FILE *file) const;

        static mlfnode_cost_type getType(const char *s);

        static const char *getName(mlfnode_cost_type a);
    };
}
