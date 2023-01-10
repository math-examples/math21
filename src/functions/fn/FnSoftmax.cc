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

#include "inner_cc.h"
#include "src/softmax/softmax_wrapper.h"
#include "FnSoftmax.h"

namespace math21 {

    FnSoftmax::FnSoftmax() {
        init();
    }

    FnSoftmax::~FnSoftmax() {}

    void FnSoftmax::init() {
        name = 0;
        batch = 0;
        groups = 0;
        inputs = 0;
        outputs = 0;
        loss = 0;
        loss_cpu = 0;
        delta = 0;
        output = 0;
        cost = 0;
        softmax_tree = 0;
        temperature = 0;
        spatial = 0;
        noloss = 0;
        h = 0;
        w = 0;
        c = 0;
    }

    void FnSoftmax::create(const mlfunction_node *finput, int groups) {
        auto f = this;
        int batch = finput->mini_batch_size;
        int inputs = finput->y_size;
        assert(inputs % groups == 0);
        f->c = finput->y_dim[0];
        f->h = finput->y_dim[1];
        f->w = finput->y_dim[2];
        f->batch = batch;
        f->groups = groups;
        f->inputs = inputs;
        f->outputs = inputs;
        f->cost = math21_vector_create_with_default_value_cpu(1, 0);
        f->loss = math21_vector_create_with_default_value_wrapper(inputs * batch, 0);
#ifndef MATH21_FLAG_USE_CPU
        f->loss_cpu = math21_vector_create_with_default_value_cpu(inputs * batch, 0);
#endif
        f->output = math21_vector_create_with_default_value_wrapper(inputs * batch, 0);
        f->delta = math21_vector_create_with_default_value_wrapper(inputs * batch, 0);
        f->name = math21_string_create_from_string("softmax");
    }

    void FnSoftmax::resize(const mlfunction_net *fnet) {

    }

    void FnSoftmax::log(const char *varName) const {
        auto f = this;
        fprintf(stdout, "softmax: (%d, %d, %d, %d) -> (%d, %d, %d, %d)\n",
                f->h, f->w, f->c, f->batch, f->h, f->w, f->c, f->batch);
    }

    void FnSoftmax::forward(mlfunction_net *net, mlfunction_node *finput) {
        auto f = this;
        if (net->is_train) {
            math21_vector_set_wrapper(f->batch * f->outputs, 0, f->delta, 1);
        }
        if (f->softmax_tree) {
            math21_ml_function_softmax_tree_wrapper(finput->y, 1, f->batch, f->inputs, f->temperature, f->output,
                                                    *f->softmax_tree);
            /*
            int i;
            int count = 0;
            for (i = 0; i < f->softmax_tree->groups; ++i) {
                int group_size = f->softmax_tree->group_size[i];
                softmax(finput->y + count, group_size, f->batch, f->inputs, 1, 0, 1, f->temperature, f->output + count);
                count += group_size;
            }
            */
        } else {
            if (f->spatial) {
                math21_ml_function_softmax_wrapper(finput->y, f->c, f->batch * f->c, f->inputs / f->c, f->w * f->h, 1,
                                                   f->w * f->h, 1, f->output);
            } else {
                math21_ml_function_softmax_wrapper(finput->y, f->inputs / f->groups, f->batch, f->inputs, f->groups,
                                                   f->inputs / f->groups, 1,
                                                   f->temperature, f->output);
            }
        }
        if (!math21_vector_isEmpty_wrapper(net->data_y_wrapper) && !f->noloss) {
            math21_ml_function_softmax_x_ent_wrapper(f->batch * f->inputs, f->output, net->data_y_wrapper, f->delta,
                                                     f->loss);
            if (f->softmax_tree) {
                math21_vector_assign_by_mask_wrapper(f->batch * f->inputs, f->delta, MATH21_MASK_NUM, net->data_y_wrapper,
                                                     0);
                math21_vector_assign_by_mask_wrapper(f->batch * f->inputs, f->loss, MATH21_MASK_NUM, net->data_y_wrapper,
                                                     0);
            }
            float *loss;
#if defined(MATH21_FLAG_USE_CPU)
            loss = f->loss;
#else
            loss = f->loss_cpu;
            math21_vector_pull_wrapper(f->loss, f->loss_cpu, f->batch * f->inputs);
#endif
            // todo: add sum wrapper, remove loss_cpu. See FnCost
            f->cost[0] = math21_vector_sum_cpu(loss, f->batch * f->inputs);
        }
    }

    void FnSoftmax::backward(mlfunction_net *net, mlfunction_node *finput) {
        auto f = this;
        math21_vector_kx_add_y_wrapper(f->batch * f->inputs, 1, f->delta, 1, finput->dy, 1);
    }

    void FnSoftmax::saveState(FILE *file) const {
        auto f = this;
//    if (math21_time_index_get() == math21_time_index_get_debug_time()) {
//        math21_tool_assert(0);
//    }
        math21_vector_serialize_c_wrapper(file, f->output, f->batch * f->outputs);
        math21_vector_serialize_c_wrapper(file, f->delta, f->batch * f->outputs);
        math21_vector_serialize_c_wrapper(file, f->loss, f->batch * f->outputs);
        math21_vector_serialize_c_cpu(file, f->cost, 1);
    }

    void FnSoftmax::netSetTemperature(mlfunction_net *fnet, float t) {
        int i;
        for (i = 0; i < fnet->n_node; ++i) {
            mlfunction_node *fnode = fnet->nodes[i];
            if (fnode->type == mlfnode_type_softmax) {
                auto *f = (FnSoftmax *) fnode->function;
                f->temperature = t;
            }
        }
    }

}
