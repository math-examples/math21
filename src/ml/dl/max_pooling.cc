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
#include "max_pooling.h"

using namespace math21;

FnMaxPooling *math21_ml_function_max_pooling_create(
        mlfunction_node *fnode, int batch, int c, int h, int w, int size, int stride, int padding);

void math21_ml_function_max_pooling_node_reset(mlfunction_node *fnode);

void math21_ml_function_max_pooling_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                          const mlfunction_node *finput, m21list *options) {
    int stride = math21_function_option_find_int(options, "stride", 1);
    int size = math21_function_option_find_int(options, "size", stride);
    int padding = math21_function_option_find_int_quiet(options, "padding", size - 1);

    int mini_batch_size, nr, nc, nch;
    nch = finput->y_dim[0];
    nr = finput->y_dim[1];
    nc = finput->y_dim[2];
    mini_batch_size = finput->mini_batch_size;

    if (!(nch && nr && nc)) math21_error("Layer before maxpool layer must output image.");

    math21_ml_function_max_pooling_create(fnode, mini_batch_size, nch, nr, nc, size, stride, padding);
}

void math21_ml_function_max_pooling_node_saveState(const mlfunction_node *fnode, FILE *file) {
    auto f = (FnMaxPooling *) fnode->function;
    f->saveState(file);
}

void math21_ml_function_max_pooling_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    auto f = (FnMaxPooling *) fnode->function;
    f->batch = mini_batch_size;
}

void
math21_ml_function_max_pooling_node_resize(mlfunction_node *fnode, const mlfunction_net *fnet, int nr_X, int nc_X) {
    auto f = (FnMaxPooling *) fnode->function;
    f->resize(nr_X, nc_X);
    math21_ml_function_max_pooling_node_reset(fnode);
}

void math21_ml_function_max_pooling_node_forward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto f = (FnMaxPooling *) fnode->function;
    f->forward(finput, net->is_train);
}

void math21_ml_function_max_pooling_node_backward(
        mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto f = (FnMaxPooling *) fnode->function;
    f->backward(finput);
}

void math21_ml_function_max_pooling_node_log(const mlfunction_node *fnode, const char *varName) {
    auto f = (const FnMaxPooling *) fnode->function;
    f->log(varName);
}

const char *math21_ml_function_max_pooling_node_getName(const mlfunction_node *fnode) {
    auto *f = (const FnMaxPooling *) fnode->function;
    return f->name;
}

void math21_ml_function_max_pooling_node_reset(mlfunction_node *fnode) {
    FnMaxPooling *f = (FnMaxPooling *) fnode->function;
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

FnMaxPooling *math21_ml_function_max_pooling_create(
        mlfunction_node *fnode, int batch, int c, int h, int w, int size, int stride, int padding) {
    auto f = new FnMaxPooling();
    f->create(batch, c, h, w, size, stride, padding);
    if (fnode) {
        fnode->type = mlfnode_type_max_pooling;
        fnode->function = f;
        fnode->saveState = math21_ml_function_max_pooling_node_saveState;
        fnode->set_mbs = math21_ml_function_max_pooling_node_set_mbs;
        fnode->resize = math21_ml_function_max_pooling_node_resize;
        fnode->forward = math21_ml_function_max_pooling_node_forward;
        fnode->backward = math21_ml_function_max_pooling_node_backward;
        fnode->log = math21_ml_function_max_pooling_node_log;
        fnode->getName = math21_ml_function_max_pooling_node_getName;
        math21_ml_function_max_pooling_node_reset(fnode);
    }
    return f;
}
