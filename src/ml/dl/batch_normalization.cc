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

#include <stdio.h>
#include <assert.h>
#include "batch_normalization.h"
#include "inner_cc.h"

using namespace math21;

FnBatchnorm *math21_ml_function_batchnorm_create(
        mlfunction_node *fnode, int is_this_type, mlfunction_node *finput,
        int mini_batch_size, int nc_Y, int nr_Y, int nch_Y, int adam);

void math21_ml_function_batchnorm_resize(FnBatchnorm *l, mlfunction_node *fnode, int nc_Y, int nr_Y);

void math21_ml_function_batchnorm_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                        const mlfunction_node *finput, m21list *options) {
    math21_ml_function_batchnorm_create(fnode, 1, 0, finput->mini_batch_size, finput->y_dim[2],
                                        finput->y_dim[1], finput->y_dim[0], 0);
}

void math21_ml_function_batchnorm_node_saveState(const mlfunction_node *fnode, FILE *file) {
    FnBatchnorm *f = (FnBatchnorm *) fnode->function;
    f->saveState(file);
}

void math21_ml_function_batchnorm_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    FnBatchnorm *f = (FnBatchnorm *) fnode->function;
    f->mini_batch_size = mini_batch_size;
}

void
math21_ml_function_batchnorm_node_forward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto *f = (FnBatchnorm *) fnode->function;
    f->is_train = net->is_train;
    f->forward(finput);
}

void
math21_ml_function_batchnorm_node_backward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    FnBatchnorm *f = (FnBatchnorm *) fnode->function;
    f->is_train = net->is_train;
    f->backward(finput);
}

void math21_ml_function_batchnorm_node_update(mlfunction_node *fnode, OptUpdate *optUpdate) {
    FnBatchnorm *f = (FnBatchnorm *) fnode->function;
    f->update(optUpdate);
}

void math21_ml_function_batchnorm_node_log(const mlfunction_node *fnode, const char *varName) {
    auto *f = (const FnBatchnorm *) fnode->function;
    f->log(varName);
}

const char *math21_ml_function_batchnorm_node_getName(const mlfunction_node *fnode) {
    auto *f = (const FnBatchnorm *) fnode->function;
    return f->name;
}


void math21_ml_function_batchnorm_node_reset(mlfunction_node *fnode) {
    auto *f = (FnBatchnorm *) fnode->function;
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

FnBatchnorm *math21_ml_function_batchnorm_create(
        mlfunction_node *fnode, int is_this_type, mlfunction_node *finput,
        int mini_batch_size, int nc_Y, int nr_Y, int nch_Y, int adam) {
    auto *f = new FnBatchnorm();
    f->create(is_this_type, finput, mini_batch_size, nc_Y, nr_Y, nch_Y, adam);
    if (fnode) {
        fnode->type = mlfnode_type_batchnorm;
        fnode->function = f;
        fnode->saveState = math21_ml_function_batchnorm_node_saveState;
        fnode->set_mbs = math21_ml_function_batchnorm_node_set_mbs;
        fnode->forward = math21_ml_function_batchnorm_node_forward;
        fnode->backward = math21_ml_function_batchnorm_node_backward;
        fnode->update = math21_ml_function_batchnorm_node_update;
        fnode->log = math21_ml_function_batchnorm_node_log;
        fnode->getName = math21_ml_function_batchnorm_node_getName;
        math21_ml_function_batchnorm_node_reset(fnode);
    }
    return f;
}

void math21_ml_function_batchnorm_resize(FnBatchnorm *f, mlfunction_node *fnode, int nc_Y, int nr_Y) {
    f->resize(fnode, nc_Y, nr_Y);
    if (fnode) {
        if (!fnode->function) {
            fnode->function = f;
        }
        math21_ml_function_batchnorm_node_reset(fnode);
    }
}