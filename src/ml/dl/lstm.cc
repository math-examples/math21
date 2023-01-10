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
#include "lstm.h"

using namespace math21;

FnLstm *math21_ml_function_lstm_create(
        mlfunction_node *fnode, int batch, int input_size, int output_size, int n_time_step, int is_use_bias,
        int is_batch_normalize, int is_unit_forget_bias, float dropout_rate_x, float dropout_rate_h, int is_adam,
        int is_return_sequences, int implementationMode);

void math21_ml_function_lstm_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                   const mlfunction_node *finput, m21list *options) {
    int output = math21_function_option_find_int(options, "output", 1);
    int is_use_bias = math21_function_option_find_int_quiet(options, "is_use_bias", 1);
    int is_batch_normalize = math21_function_option_find_int_quiet(options, "batch_normalize", 0);
    int is_unit_forget_bias = math21_function_option_find_int_quiet(options, "is_unit_forget_bias", 1);
    float dropout_rate_x = math21_function_option_find_float_quiet(options, "dropout_rate_x", 0);
    float dropout_rate_h = math21_function_option_find_float_quiet(options, "dropout_rate_h", 0);
    int is_return_sequences = math21_function_option_find_int_quiet(options, "is_return_sequences", 0);
    int implementationMode = math21_function_option_find_int_quiet(options, "implementationMode", 1);
    math21_ml_function_lstm_create(fnode, finput->mini_batch_size, finput->y_size, output, fnet->n_time_step_in_rnn,
                                   is_use_bias,
                                   is_batch_normalize, is_unit_forget_bias, dropout_rate_x, dropout_rate_h,
                                   fnet->adam, is_return_sequences, implementationMode);
}

void math21_ml_function_lstm_node_saveState(const mlfunction_node *fnode, FILE *file) {
    auto f = (FnLstm *) fnode->function;
    f->saveState(file);
}

// todo: may deprecate this.
void math21_ml_function_lstm_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    auto f = (FnLstm *) fnode->function;
    f->setMbs(mini_batch_size);
}

void math21_ml_function_lstm_node_forward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto f = (FnLstm *) fnode->function;
    f->forward(finput, net->is_train);
}

void math21_ml_function_lstm_node_backward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto f = (FnLstm *) fnode->function;
    f->backward(finput, net->is_train);
}

void math21_ml_function_lstm_node_update(mlfunction_node *fnode, OptUpdate *optUpdate) {
    auto f = (FnLstm *) fnode->function;
    f->update(optUpdate);
}

void math21_ml_function_lstm_node_log(const mlfunction_node *fnode, const char *varName) {
    auto f = (const FnLstm *) fnode->function;
    f->log(varName);
}

void math21_ml_function_lstm_node_reset_state(mlfunction_node *fnode, int b) {
    auto f = (FnLstm *) fnode->function;
    f->resetState(b);
}

void math21_ml_function_lstm_node_save_theta(mlfunction_node *fnode, FILE *fp) {
    auto f = (FnLstm *) fnode->function;
    f->saveTheta(fp);
}

void math21_ml_function_lstm_node_load_theta(mlfunction_node *fnode, FILE *fp) {
    auto f = (FnLstm *) fnode->function;
    f->loadTheta(fp);
}

void math21_ml_function_lstm_node_load_theta_order_bwsmv_flipped(mlfunction_node *fnode, FILE *fp, int flipped) {
    auto f = (FnLstm *) fnode->function;
    f->loadThetaOrderBwsmvFlipped(fp, flipped);
}

void math21_ml_function_lstm_node_save_theta_order_bwsmv(mlfunction_node *fnode, FILE *fp) {
    auto f = (FnLstm *) fnode->function;
    f->saveThetaOrderBwsmv(fp);
}

const char *math21_ml_function_lstm_node_getName(const mlfunction_node *fnode) {
    auto *f = (const FnLstm *) fnode->function;
    return f->name;
}

void math21_ml_function_lstm_node_reset(mlfunction_node *fnode) {
    auto *f = (FnLstm *) fnode->function;
    fnode->x_dim[0] = f->inputs;
    fnode->x_dim[1] = 1;
    fnode->x_dim[2] = 1;
    fnode->y_dim[0] = f->outputs;
    fnode->y_dim[1] = 1;
    fnode->y_dim[2] = 1;
    fnode->x_size = fnode->x_dim[0] * fnode->x_dim[1] * fnode->x_dim[2];
    fnode->y_size = fnode->y_dim[0] * fnode->y_dim[1] * fnode->y_dim[2];
    if (f->is_return_sequences) {
        fnode->mini_batch_size = f->steps * f->batch;
        fnode->y = f->output;
        fnode->dy = f->delta;
    } else {
        fnode->mini_batch_size = f->batch;
        fnode->y = f->last_output;
        fnode->dy = f->delta + (f->steps - 1) * f->batch * f->outputs;
    }
}

FnLstm *math21_ml_function_lstm_create(
        mlfunction_node *fnode, int batch, int input_size, int output_size, int n_time_step, int is_use_bias,
        int is_batch_normalize, int is_unit_forget_bias, float dropout_rate_x, float dropout_rate_h, int is_adam,
        int is_return_sequences, int implementationMode) {
    auto *f = new FnLstm();
    f->create(batch, input_size, output_size, n_time_step, is_use_bias,
              is_batch_normalize, is_unit_forget_bias, dropout_rate_x, dropout_rate_h, is_adam,
              is_return_sequences, implementationMode);
    if (fnode) {
        fnode->type = mlfnode_type_lstm;
        fnode->function = f;
        fnode->saveState = math21_ml_function_lstm_node_saveState;
        fnode->set_mbs = math21_ml_function_lstm_node_set_mbs;
        fnode->forward = math21_ml_function_lstm_node_forward;
        fnode->backward = math21_ml_function_lstm_node_backward;
        fnode->update = math21_ml_function_lstm_node_update;
        fnode->log = math21_ml_function_lstm_node_log;
        fnode->resetState = math21_ml_function_lstm_node_reset_state;
        fnode->saveTheta = math21_ml_function_lstm_node_save_theta;
        fnode->loadTheta = math21_ml_function_lstm_node_load_theta;
        fnode->loadThetaOrderBwsmvFlipped = math21_ml_function_lstm_node_load_theta_order_bwsmv_flipped;
        fnode->saveThetaOrderBwsmv = math21_ml_function_lstm_node_save_theta_order_bwsmv;
        fnode->getName = math21_ml_function_lstm_node_getName;
        math21_ml_function_lstm_node_reset(fnode);
    }
    return f;
}