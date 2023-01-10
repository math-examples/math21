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
#include "dropout.h"

using namespace math21;

FnDropout *
math21_ml_function_dropout_create(mlfunction_node *fnode, mlfunction_node *finput, float rate, int n_time_step,
                                  const char *name);

void math21_ml_function_dropout_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                      mlfunction_node *finput, m21list *options) {

    float probability = math21_function_option_find_float(options, "probability", .5);
    const char *name = math21_function_option_find_str_quiet(options, "name", 0);
    int n_time_step = math21_function_option_find_int_quiet(options, "n_time_step", 1);
    math21_ml_function_dropout_create(fnode, finput, probability, n_time_step, name);
}

void math21_ml_function_dropout_node_saveState(const mlfunction_node *fnode, FILE *file) {
    auto f = (FnDropout *) fnode->function;
    f->saveState(file);
}

void math21_ml_function_dropout_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    auto *f = (FnDropout *) fnode->function;
    f->batch = mini_batch_size;
}

void math21_ml_function_dropout_node_forward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto f = (FnDropout *) fnode->function;
    f->forward(finput, net->is_train);
}

void math21_ml_function_dropout_node_backward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto f = (FnDropout *) fnode->function;
    f->backward(finput);
}

void math21_ml_function_dropout_node_log(const mlfunction_node *fnode, const char *varName) {
    auto f = (const FnDropout *) fnode->function;
    f->log(varName);
}

const char *math21_ml_function_dropout_node_getName(const mlfunction_node *fnode) {
    auto *f = (const FnDropout *) fnode->function;
    return f->name;
}

void math21_ml_function_dropout_node_reset(mlfunction_node *fnode) {
    auto *f = (FnDropout *) fnode->function;
    fnode->mini_batch_size = f->total_mbs;
    math21_rawtensor_shape_assign(fnode->y_dim, f->y_dim);
    fnode->y_size = f->outputs;
    fnode->y = f->y;
    fnode->dy = f->dy;
}

// finput can have empty vector, but must have its shape.
FnDropout *
math21_ml_function_dropout_create(mlfunction_node *fnode, mlfunction_node *finput, float rate, int n_time_step,
                                  const char *name) {
    auto f = new FnDropout(finput, rate, n_time_step, name);
    if (fnode) {
        fnode->type = mlfnode_type_dropout;
        fnode->function = f;
        fnode->saveState = math21_ml_function_dropout_node_saveState;
        fnode->set_mbs = math21_ml_function_dropout_node_set_mbs;
        fnode->forward = math21_ml_function_dropout_node_forward;
        fnode->backward = math21_ml_function_dropout_node_backward;
        fnode->log = math21_ml_function_dropout_node_log;
        fnode->getName = math21_ml_function_dropout_node_getName;
        math21_ml_function_dropout_node_reset(fnode);
    }
    return f;
}