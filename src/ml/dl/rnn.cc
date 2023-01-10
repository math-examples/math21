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
#include "rnn.h"

using namespace math21;

FnRnn *math21_ml_function_rnn_create(
        mlfunction_node *fnode, int batch_size, int input_size, int output_size,
        int n_time_step, MATH21_FUNCTION_ACTIVATION_TYPE activation, int is_use_bias, int is_batch_normalize,
        int is_adam);

void math21_ml_function_rnn_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                  const mlfunction_node *finput, m21list *options) {
    int output = math21_function_option_find_int(options, "output", 1);
    const char *activation_s = math21_function_option_find_str(options, "activation", "logistic");
    MATH21_FUNCTION_ACTIVATION_TYPE activation = math21_function_activation_get_type(activation_s);
    int is_use_bias = math21_function_option_find_int_quiet(options, "is_use_bias", 1);
    int is_batch_normalize = math21_function_option_find_int_quiet(options, "batch_normalize", 0);

    math21_ml_function_rnn_create(fnode, fnet->mini_batch_size, finput->y_size, output,
                                  fnet->n_time_step_in_rnn, activation,
                                  is_use_bias, is_batch_normalize, fnet->adam);
}

void math21_ml_function_rnn_node_saveState(const mlfunction_node *fnode, FILE *file) {
    auto f = (FnRnn *) fnode->function;
    f->saveState(file);
}

// todo: may deprecate this.
void math21_ml_function_rnn_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    auto f = (FnRnn *) fnode->function;
    f->setMbs(mini_batch_size);
}

void math21_ml_function_rnn_node_forward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto *f = (FnRnn *) fnode->function;
    f->forward(finput, net->is_train);
}

void math21_ml_function_rnn_node_backward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto f = (FnRnn *) fnode->function;
    f->backward(finput, net->is_train);
}

void math21_ml_function_rnn_node_update(mlfunction_node *fnode, OptUpdate *optUpdate) {
    auto f = (FnRnn *) fnode->function;
    f->update(optUpdate);
}

void math21_ml_function_rnn_node_reset_state(mlfunction_node *fnode, int b) {
    auto f = (FnRnn *) fnode->function;
    f->resetState(b);
}

void math21_ml_function_rnn_node_save_theta_order_bwsmv(mlfunction_node *fnode, FILE *fp) {
    auto f = (FnRnn *) fnode->function;
    f->saveThetaOrderBwsmv(fp);
}

void math21_ml_function_rnn_node_save_theta(mlfunction_node *fnode, FILE *fp) {
    auto f = (FnRnn *) fnode->function;
    f->saveTheta(fp);
}

void math21_ml_function_rnn_node_load_theta(mlfunction_node *fnode, FILE *fp) {
    auto f = (FnRnn *) fnode->function;
    f->loadTheta(fp);
}

void math21_ml_function_rnn_node_load_theta_order_bwsmv_flipped(mlfunction_node *fnode, FILE *fp, int flipped) {
    auto f = (FnRnn *) fnode->function;
    f->loadThetaOrderBwsmvFlipped(fp, flipped);
}

void math21_ml_function_rnn_node_reset(mlfunction_node *fnode) {
    auto *f = (FnRnn *) fnode->function;
    fnode->mini_batch_size = f->steps * f->batch;
    fnode->x_dim[0] = f->inputs;
    fnode->x_dim[1] = 1;
    fnode->x_dim[2] = 1;
    fnode->y_dim[0] = f->outputs;
    fnode->y_dim[1] = 1;
    fnode->y_dim[2] = 1;
    fnode->x_size = fnode->x_dim[0] * fnode->x_dim[1] * fnode->x_dim[2];
    fnode->y_size = fnode->y_dim[0] * fnode->y_dim[1] * fnode->y_dim[2];
    fnode->y = f->output;
    fnode->dy = f->delta;
}

FnRnn *math21_ml_function_rnn_create(
        mlfunction_node *fnode, int batch_size, int input_size, int output_size,
        int n_time_step, MATH21_FUNCTION_ACTIVATION_TYPE activation, int is_use_bias, int is_batch_normalize,
        int is_adam) {
    FnRnn *f = new FnRnn();
    f->create(batch_size, input_size, output_size,
              n_time_step, activation, is_use_bias, is_batch_normalize,
              is_adam);
    if (fnode) {
        fnode->type = mlfnode_type_rnn;
        fnode->function = f;
        fnode->saveState = math21_ml_function_rnn_node_saveState;
        fnode->set_mbs = math21_ml_function_rnn_node_set_mbs;
        fnode->forward = math21_ml_function_rnn_node_forward;
        fnode->backward = math21_ml_function_rnn_node_backward;
        fnode->update = math21_ml_function_rnn_node_update;
        fnode->resetState = math21_ml_function_rnn_node_reset_state;
        fnode->saveTheta = math21_ml_function_rnn_node_save_theta;
        fnode->loadTheta = math21_ml_function_rnn_node_load_theta;
        fnode->loadThetaOrderBwsmvFlipped = math21_ml_function_rnn_node_load_theta_order_bwsmv_flipped;
        fnode->saveThetaOrderBwsmv = math21_ml_function_rnn_node_save_theta_order_bwsmv;
        math21_ml_function_rnn_node_reset(fnode);
    }
    return f;
}