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
#include "fully_connected.h"

using namespace math21;

FnFullyConnected *math21_ml_function_fully_connected_create(
        mlfunction_node *fnode, int batch_size, int input_size, int output_size,
        MATH21_FUNCTION_ACTIVATION_TYPE activation, int is_use_bias, int is_batch_normalize, int is_adam,
        const char *name);

void math21_ml_function_fully_connected_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                              const mlfunction_node *finput, m21list *options) {
    int output = math21_function_option_find_int(options, "output", 1);
    const char *activation_s = math21_function_option_find_str(options, "activation", "logistic");
    const char *name = math21_function_option_find_str_quiet(options, "name", 0);
    MATH21_FUNCTION_ACTIVATION_TYPE activation = math21_function_activation_get_type(activation_s);
    int is_use_bias = math21_function_option_find_int_quiet(options, "is_use_bias", 1);
    int is_batch_normalize = math21_function_option_find_int_quiet(options, "batch_normalize", 0);

    FnFullyConnected *f = math21_ml_function_fully_connected_create(fnode, finput->mini_batch_size,
                                                                    finput->y_size, output,
                                                                    activation, is_use_bias,
                                                                    is_batch_normalize, fnet->adam, name);
    f->learning_rate_scale = math21_function_option_find_float_quiet(options, "learning_rate", 1);
}

void math21_ml_function_fully_connected_node_saveState(const mlfunction_node *fnode, FILE *file) {
    auto *f = (FnFullyConnected *) fnode->function;
    f->saveState(file);
}

// todo: check relation with n_time_step;
void math21_ml_function_fully_connected_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    auto *f = (FnFullyConnected *) fnode->function;
    f->setMbs(mini_batch_size);
}

void
math21_ml_function_fully_connected_node_forward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto *f = (FnFullyConnected *) fnode->function;
    f->forward(finput, net->is_train);
}

void
math21_ml_function_fully_connected_node_backward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto *f = (FnFullyConnected *) fnode->function;
    f->backward(finput, net->is_train);
}

void math21_ml_function_fully_connected_node_update(mlfunction_node *fnode, OptUpdate *optUpdate) {
    auto *f = (FnFullyConnected *) fnode->function;
    f->update(optUpdate);
}

void math21_ml_function_fully_connected_node_log(const mlfunction_node *fnode, const char *varName) {
    auto *f = (const FnFullyConnected *) fnode->function;
    f->log(varName);
}

void math21_ml_function_fully_connected_node_merge_to(mlfunction_node *fnode, mlfunction_node *fbase) {
    auto *f = (FnFullyConnected *) fnode->function;
    auto *fb = (FnFullyConnected *) fbase->function;
    f->mergeTo(fb);
}

void math21_ml_function_fully_connected_node_scale(mlfunction_node *fnode, float s) {
    auto *f = (FnFullyConnected *) fnode->function;
    f->scale(s);
}

void math21_ml_function_fully_connected_node_pull_wrapper(mlfunction_node *fnode, NumB useRolling) {
    auto *f = (FnFullyConnected *) fnode->function;
    f->pullWrapper(useRolling);
}

void math21_ml_function_fully_connected_node_push_by_wrapper(mlfunction_node *fnode, mlfunction_node *fbase,
                                                             NumB useRolling) {
    auto *f = (FnFullyConnected *) fnode->function;
    auto *fb = (FnFullyConnected *) fbase->function;
    f->pushByWrapper(fb, useRolling);
}

void math21_ml_function_fully_connected_node_load_theta(mlfunction_node *fnode, FILE *fp) {
    auto *f = (FnFullyConnected *) fnode->function;
    f->loadTheta(fp);
}

void math21_ml_function_fully_connected_node_save_theta(mlfunction_node *fnode, FILE *fp) {
    auto *f = (FnFullyConnected *) fnode->function;
    f->saveTheta(fp);
}

void math21_ml_function_fully_connected_node_save_theta_order_bwsmv(mlfunction_node *fnode, FILE *fp) {
    auto *f = (FnFullyConnected *) fnode->function;
    f->saveThetaOrderBwsmv(fp);
}

void
math21_ml_function_fully_connected_node_load_theta_order_bwsmv_flipped(mlfunction_node *fnode, FILE *fp, int flipped) {
    auto *f = (FnFullyConnected *) fnode->function;
    f->loadThetaOrderBwsmvFlipped(fp, flipped);
}

const char *math21_ml_function_fully_connected_node_getName(const mlfunction_node *fnode) {
    auto *f = (const FnFullyConnected *) fnode->function;
    return f->name;
}

void math21_ml_function_fully_connected_node_reset(mlfunction_node *fnode) {
    auto *f = (FnFullyConnected *) fnode->function;
    fnode->mini_batch_size = f->total_mbs;
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

void math21_ml_function_fully_connected_node_set_it(
        mlfunction_node *fnode, FnFullyConnected* f) {
    if (fnode) {
        fnode->type = mlfnode_type_fully_connected;
        fnode->function = f;
        fnode->saveState = math21_ml_function_fully_connected_node_saveState;
        fnode->set_mbs = math21_ml_function_fully_connected_node_set_mbs;
        fnode->forward = math21_ml_function_fully_connected_node_forward;
        fnode->backward = math21_ml_function_fully_connected_node_backward;
        fnode->update = math21_ml_function_fully_connected_node_update;
        fnode->log = math21_ml_function_fully_connected_node_log;
        fnode->mergeTo = math21_ml_function_fully_connected_node_merge_to;
        fnode->scale = math21_ml_function_fully_connected_node_scale;
        fnode->pull = math21_ml_function_fully_connected_node_pull_wrapper;
        fnode->pushBy = math21_ml_function_fully_connected_node_push_by_wrapper;
        fnode->loadTheta = math21_ml_function_fully_connected_node_load_theta;
        fnode->saveTheta = math21_ml_function_fully_connected_node_save_theta;
        fnode->loadThetaOrderBwsmvFlipped = math21_ml_function_fully_connected_node_load_theta_order_bwsmv_flipped;
        fnode->saveThetaOrderBwsmv = math21_ml_function_fully_connected_node_save_theta_order_bwsmv;
        fnode->getName = math21_ml_function_fully_connected_node_getName;
        math21_ml_function_fully_connected_node_reset(fnode);
    }
}

// Z = h(Y), Y = X*W.t + b
FnFullyConnected *math21_ml_function_fully_connected_create(
        mlfunction_node *fnode, int batch_size, int input_size, int output_size,
        MATH21_FUNCTION_ACTIVATION_TYPE activation, int is_use_bias, int is_batch_normalize, int is_adam,
        const char *name) {
    auto *f = new FnFullyConnected();
    f->create(batch_size, input_size, output_size,
              activation, is_use_bias, is_batch_normalize, is_adam,
              name);
    math21_ml_function_fully_connected_node_set_it(fnode, f);
    return f;
}