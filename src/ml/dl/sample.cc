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
#include "sample.h"

using namespace math21;

void math21_ml_function_sample_node_reset(mlfunction_node *fnode);

FnSample *
math21_ml_function_sample_create(mlfunction_node *fnode, int mini_batch_size, int nc, int nr, int nch, int stride);

void math21_ml_function_sample_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                     const mlfunction_node *finput, m21list *options) {

    int stride = math21_function_option_find_int(options, "stride", 2);
    FnSample *f = math21_ml_function_sample_create(fnode, finput->mini_batch_size, finput->y_dim[2],
                                                   finput->y_dim[1], finput->y_dim[0], stride);
    f->scale = math21_function_option_find_float_quiet(options, "scale", 1);
}

void math21_ml_function_sample_node_saveState(const mlfunction_node *fnode, FILE *file) {
    auto *f = (FnSample *) fnode->function;
    f->saveState(file);
}

void math21_ml_function_sample_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    auto *f = (FnSample *) fnode->function;
    f->batch = mini_batch_size;
}

void math21_ml_function_sample_node_resize(mlfunction_node *fnode, const mlfunction_net *fnet, int nr_X, int nc_X) {
    auto *f = (FnSample *) fnode->function;
    f->resize(nr_X, nc_X);
    math21_ml_function_sample_node_reset(fnode);
}

void math21_ml_function_sample_node_forward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto *f = (FnSample *) fnode->function;
    f->forward(finput, net->is_train);
}

void math21_ml_function_sample_node_backward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto *f = (FnSample *) fnode->function;
    f->backward(finput);
}

void math21_ml_function_sample_node_log(const mlfunction_node *fnode, const char *varName) {
    auto *f = (const FnSample *) fnode->function;
    f->log(varName);
}

const char *math21_ml_function_sample_node_getName(const mlfunction_node *fnode) {
    auto *f = (const FnSample *) fnode->function;
    return f->name;
}

void math21_ml_function_sample_node_reset(mlfunction_node *fnode) {
    auto *f = (FnSample *) fnode->function;
    fnode->mini_batch_size = f->batch;
    fnode->x_dim[0] = f->c;
    fnode->x_dim[1] = f->h;
    fnode->x_dim[2] = f->w;
    fnode->y_dim[0] = f->out_c;
    fnode->y_dim[1] = f->out_h;
    fnode->y_dim[2] = f->out_w;
    fnode->x_size = fnode->x_dim[0] * fnode->x_dim[1] * fnode->x_dim[2];
    fnode->y_size = fnode->y_dim[0] * fnode->y_dim[1] * fnode->y_dim[2];
    fnode->y = f->output;
    fnode->dy = f->delta;
}

FnSample *
math21_ml_function_sample_create(mlfunction_node *fnode, int mini_batch_size, int nc, int nr, int nch, int stride) {
    auto *f = new FnSample();
    f->create(mini_batch_size, nc, nr, nch, stride);
    if (fnode) {
        fnode->type = mlfnode_type_sample;
        fnode->function = f;
        fnode->saveState = math21_ml_function_sample_node_saveState;
        fnode->set_mbs = math21_ml_function_sample_node_set_mbs;
        fnode->resize = math21_ml_function_sample_node_resize;
        fnode->forward = math21_ml_function_sample_node_forward;
        fnode->backward = math21_ml_function_sample_node_backward;
        fnode->log = math21_ml_function_sample_node_log;
        fnode->getName = math21_ml_function_sample_node_getName;
        math21_ml_function_sample_node_reset(fnode);
    }
    return f;
}