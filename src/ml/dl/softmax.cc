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
#include "softmax.h"

using namespace math21;

FnSoftmax *math21_ml_function_softmax_create(mlfunction_node *fnode, const mlfunction_node *finput, int groups);

void math21_ml_function_softmax_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                      const mlfunction_node *finput, m21list *options) {
    int groups = math21_function_option_find_int_quiet(options, "groups", 1);
    FnSoftmax *f = math21_ml_function_softmax_create(fnode, finput, groups);

    f->temperature = math21_function_option_find_float_quiet(options, "temperature", 1);
    const char *tree_file = math21_function_option_find_str(options, "tree", 0);
    if (tree_file) f->softmax_tree = math21_data_struncture_tree_read(tree_file);
    f->spatial = math21_function_option_find_int_quiet(options, "spatial", 0);
    f->noloss = math21_function_option_find_int_quiet(options, "noloss", 0);

    math21_tool_assert(f->softmax_tree == 0);
}

void math21_ml_function_softmax_node_saveState(const mlfunction_node *fnode, FILE *file) {
    auto f = (FnSoftmax *) fnode->function;
    f->saveState(file);
}

float math21_ml_function_softmax_node_getCost(mlfunction_node *fnode) {
    auto f = (FnSoftmax *) fnode->function;
    return *f->cost;
}

void math21_ml_function_softmax_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    auto f = (FnSoftmax *) fnode->function;
    f->batch = mini_batch_size;
}

void math21_ml_function_softmax_node_forward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto f = (FnSoftmax *) fnode->function;
    f->forward(net, finput);
}

void math21_ml_function_softmax_node_backward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto f = (FnSoftmax *) fnode->function;
    f->backward(net, finput);
}

void math21_ml_function_softmax_node_log(const mlfunction_node *fnode, const char *varName) {
    auto f = (const FnSoftmax *) fnode->function;
    f->log(varName);
}

const char *math21_ml_function_softmax_node_getName(const mlfunction_node *fnode) {
    auto f = (const FnSoftmax *) fnode->function;
    return f->name;
}

void math21_ml_function_softmax_node_reset(mlfunction_node *fnode) {
    auto f = (FnSoftmax *) fnode->function;
    fnode->mini_batch_size = f->batch;
    fnode->y_dim[0] = f->c;
    fnode->y_dim[1] = f->h;
    fnode->y_dim[2] = f->w;
    fnode->y_size = fnode->y_dim[0] * fnode->y_dim[1] * fnode->y_dim[2];
    fnode->y = f->output;
    fnode->dy = f->delta;
}

FnSoftmax *
math21_ml_function_softmax_create(mlfunction_node *fnode, const mlfunction_node *finput, int groups) {
    auto f = new FnSoftmax();
    f->create(finput, groups);
    if (fnode) {
        fnode->type = mlfnode_type_softmax;
        fnode->function = f;
        fnode->saveState = math21_ml_function_softmax_node_saveState;
        fnode->getCost = math21_ml_function_softmax_node_getCost;
        fnode->set_mbs = math21_ml_function_softmax_node_set_mbs;
        fnode->forward = math21_ml_function_softmax_node_forward;
        fnode->backward = math21_ml_function_softmax_node_backward;
        fnode->log = math21_ml_function_softmax_node_log;
        fnode->getName = math21_ml_function_softmax_node_getName;
        math21_ml_function_softmax_node_reset(fnode);
    }
    return f;
}