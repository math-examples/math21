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
#include "res.h"

using namespace math21;

void math21_ml_function_res_node_reset(mlfunction_node *fnode);

// shortcut
FnRes *math21_ml_function_res_create(
        mlfunction_node *fnode, int batch, int index, int w, int h, int c, int w2, int h2, int c2);

void math21_ml_function_res_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                  const mlfunction_node *finput, m21list *options) {
    char *l = math21_function_option_find(options, "from");
    int index = atoi(l);
    int index_fnode = fnode->id - 1;
    if (index < 0) index = index_fnode + index;

    mlfunction_node *from = fnet->nodes[index];

    FnRes *f = math21_ml_function_res_create(fnode, finput->mini_batch_size, index, finput->y_dim[2],
                                             finput->y_dim[1], finput->y_dim[0], from->y_dim[2],
                                             from->y_dim[1],
                                             from->y_dim[0]);

    f->k1 = math21_function_option_find_float_quiet(options, "beta", 1);
    f->k2 = math21_function_option_find_float_quiet(options, "alpha", 1);
    const char *activation_s = math21_function_option_find_str(options, "activation", "linear");
    f->activation = math21_function_activation_get_type(activation_s);
}

void math21_ml_function_res_node_saveState(const mlfunction_node *fnode, FILE *file) {
    auto *f = (FnRes *) fnode->function;
    f->saveState(file);
}

void math21_ml_function_res_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    auto *f = (FnRes *) fnode->function;
    f->mini_batch_size = mini_batch_size;
}

void math21_ml_function_res_node_resize(mlfunction_node *fnode, const mlfunction_net *fnet, int nr_X, int nc_X) {
    auto *f = (FnRes *) fnode->function;
    f->resize(nr_X, nc_X);
    math21_ml_function_res_node_reset(fnode);
}

void math21_ml_function_res_node_forward(mlfunction_node *fnode, mlfunction_net *fnet, mlfunction_node *finput) {
    auto *f = (FnRes *) fnode->function;
    f->forward(fnet, finput);
}

void math21_ml_function_res_node_backward(mlfunction_node *fnode, mlfunction_net *fnet, mlfunction_node *finput) {
    auto *f = (FnRes *) fnode->function;
    f->backward(fnet, finput);
}

void math21_ml_function_res_node_log(const mlfunction_node *fnode, const char *varName) {
    auto *f = (const FnRes *) fnode->function;
    f->log(varName);
}

const char *math21_ml_function_res_node_getName(const mlfunction_node *fnode) {
    auto *f = (const FnRes *) fnode->function;
    return f->name;
}

void math21_ml_function_res_node_reset(mlfunction_node *fnode) {
    auto *f = (FnRes *) fnode->function;
    fnode->mini_batch_size = f->mini_batch_size;
    fnode->y_dim[0] = f->out_c;
    fnode->y_dim[1] = f->out_h;
    fnode->y_dim[2] = f->out_w;
    fnode->y_size = fnode->y_dim[0] * fnode->y_dim[1] * fnode->y_dim[2];
    fnode->y = f->output;
    fnode->dy = f->delta;
}

// shortcut
FnRes *math21_ml_function_res_create(
        mlfunction_node *fnode, int batch, int index,
        int w, int h, int c, int w2, int h2,
        int c2) {
    auto *f = new FnRes();
    NumN deviceType = math21_gpu_is_available() ? m21_device_type_gpu : m21_device_type_default;
    f->create(batch, index, w, h, c, w2, h2, c2, deviceType);
    if (fnode) {
        fnode->type = mlfnode_type_res;
        fnode->function = f;
        fnode->saveState = math21_ml_function_res_node_saveState;
        fnode->set_mbs = math21_ml_function_res_node_set_mbs;
        fnode->resize = math21_ml_function_res_node_resize;
        fnode->forward = math21_ml_function_res_node_forward;
        fnode->backward = math21_ml_function_res_node_backward;
        fnode->log = math21_ml_function_res_node_log;
        fnode->getName = math21_ml_function_res_node_getName;
        math21_ml_function_res_node_reset(fnode);
    }
    return f;
}