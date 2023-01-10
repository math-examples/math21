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
#include "cost.h"

using namespace math21;

void math21_ml_function_cost_node_reset(mlfunction_node *fnode);

FnCost *math21_ml_function_cost_create(
        mlfunction_node *fnode, const mlfunction_node *finput,
        mlfnode_cost_type cost_type, float scale);

void math21_ml_function_cost_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                   const mlfunction_node *finput, m21list *options) {
    const char *type_s = math21_function_option_find_str(options, "type", "sse");
    auto type = FnCost::getType(type_s);
    float scale = math21_function_option_find_float_quiet(options, "scale", 1);
    FnCost *f = math21_ml_function_cost_create(fnode, finput, type, scale);
    f->ratio = math21_function_option_find_float_quiet(options, "ratio", 0);
    f->noobject_scale = math21_function_option_find_float_quiet(options, "noobj", 1);
    f->thresh = math21_function_option_find_float_quiet(options, "thresh", 0);
    f->smooth = math21_function_option_find_float_quiet(options, "smooth", 0);
}

void math21_ml_function_cost_node_saveState(const mlfunction_node *fnode, FILE *file) {
    auto f = (FnCost *) fnode->function;
    f->saveState(file);
}

float math21_ml_function_cost_node_getCost(mlfunction_node *fnode) {
    auto *f = (FnCost *) fnode->function;
    return *f->cost;
}

void math21_ml_function_cost_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    auto *f = (FnCost *) fnode->function;
    f->batch = mini_batch_size;
}

void math21_ml_function_cost_node_resize(mlfunction_node *fnode, const mlfunction_net *fnet, int nr_X, int nc_X) {
    auto f = (FnCost *) fnode->function;
    f->resize(nr_X, nc_X);
    math21_ml_function_cost_node_reset(fnode);
}

void math21_ml_function_cost_node_forward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto f = (FnCost *) fnode->function;
    f->forward(net, finput);
}

void math21_ml_function_cost_node_backward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto f = (FnCost *) fnode->function;
    f->backward(net, finput);
}

void math21_ml_function_cost_node_log(const mlfunction_node *fnode, const char *varName) {
    auto f = (const FnCost *) fnode->function;
    f->log(varName);
}

const char *math21_ml_function_cost_node_getName(const mlfunction_node *fnode) {
    auto *f = (const FnCost *) fnode->function;
    return f->name;
}

void math21_ml_function_cost_node_reset(mlfunction_node *fnode) {
    auto *f = (FnCost *) fnode->function;
    fnode->mini_batch_size = f->batch;
    math21_rawtensor_shape_assign(fnode->y_dim, f->y_dim);
    fnode->y_size = f->outputs;
    fnode->y = f->output;
    fnode->dy = f->delta;
}
FnCost *math21_ml_function_cost_create(
        mlfunction_node *fnode, const mlfunction_node *finput,
        mlfnode_cost_type cost_type, float scale) {
    auto f = new FnCost();
    f->create(finput, cost_type, scale);
    if (fnode) {
        fnode->type = mlfnode_type_cost;
        fnode->function = f;
        fnode->saveState = math21_ml_function_cost_node_saveState;
        fnode->getCost = math21_ml_function_cost_node_getCost;
        fnode->set_mbs = math21_ml_function_cost_node_set_mbs;
        fnode->resize = math21_ml_function_cost_node_resize;
        fnode->forward = math21_ml_function_cost_node_forward;
        fnode->backward = math21_ml_function_cost_node_backward;
        fnode->log = math21_ml_function_cost_node_log;
        fnode->getName = math21_ml_function_cost_node_getName;
        math21_ml_function_cost_node_reset(fnode);
    }
    return f;
}