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
#include "conv.h"

using namespace math21;

void math21_ml_function_conv_node_reset(mlfunction_node *fnode);

FnConv *
math21_ml_function_conv_create(mlfunction_node *fnode, int mini_batch_size, int nr_X, int nc_X, int nch_X, int nch_Y,
                               int num_group, int k_size, int stride, int padding,
                               MATH21_FUNCTION_ACTIVATION_TYPE activation,
                               int is_batch_normalize, int adam);

// todo: consider add channels_last and channels_first
// todo: rename to cross-correlated
void math21_ml_function_conv_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                   const mlfunction_node *finput, m21list *options) {
    int nch_out = math21_function_option_find_int(options, "filters", 1);
    int k_size = math21_function_option_find_int(options, "size", 1);
    int stride = math21_function_option_find_int(options, "stride", 1);
    // todo: padding type: valid or same
    int pad = math21_function_option_find_int_quiet(options, "pad", 0);
    int padding = math21_function_option_find_int_quiet(options, "padding", 0);
    int groups = math21_function_option_find_int_quiet(options, "groups", 1);
    if (pad) padding = k_size / 2;

    const char *activation_s = math21_function_option_find_str(options, "activation", "logistic");
    MATH21_FUNCTION_ACTIVATION_TYPE activation = math21_function_activation_get_type(activation_s);

    int mini_batch_size, nr, nc, nch;
    nch = finput->y_dim[0];
    nr = finput->y_dim[1];
    nc = finput->y_dim[2];
    mini_batch_size = finput->mini_batch_size;
    if (!(nr && nc && nch)) math21_error("Layer before convolutional layer must output image.");
    int is_batch_normalize = math21_function_option_find_int_quiet(options, "batch_normalize", 0);

    FnConv *f = math21_ml_function_conv_create(fnode, mini_batch_size, nr, nc, nch, nch_out, groups,
                                               k_size,
                                               stride, padding, activation, is_batch_normalize, fnet->adam);

    f->flipped = math21_function_option_find_int_quiet(options, "flipped", 0);
    f->learning_rate_scale = math21_function_option_find_float_quiet(options, "learning_rate", 1);
}

void math21_ml_function_conv_node_create(mlfunction_node *fnode) {
}

void math21_ml_function_conv_node_saveState(const mlfunction_node *fnode, FILE *file) {
    auto *f = (FnConv *) fnode->function;
    f->saveState(file);
}

size_t math21_ml_function_conv_node_getGlobalSpaceSize(mlfunction_node *fnode) {
    auto *f = (FnConv *) fnode->function;
    return f->workspace_size;
}

// todo: check relation with n_time_step;
void math21_ml_function_conv_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    auto *f = (FnConv *) fnode->function;
    f->set_mbs(mini_batch_size);
}

void math21_ml_function_conv_node_resize(mlfunction_node *fnode, const mlfunction_net *fnet, int nr_X, int nc_X) {
    auto *f = (FnConv *) fnode->function;
    f->resize(nr_X, nc_X);
    math21_ml_function_conv_node_reset(fnode);
}

void math21_ml_function_conv_node_forward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto *f = (FnConv *) fnode->function;
    f->forward(finput, net->is_train, net->workspace);
}

void math21_ml_function_conv_node_backward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto *f = (FnConv *) fnode->function;
    f->backward(finput, net->is_train, net->workspace);
}

void math21_ml_function_conv_node_update(mlfunction_node *fnode, OptUpdate *optUpdate) {
    auto *f = (FnConv *) fnode->function;
    f->update(optUpdate);
}

const void *math21_ml_function_conv_node_getDataToCpu(mlfunction_node *fnode, const char *varName) {
    auto *f = (FnConv *) fnode->function;
    return f->getDataToCpu(varName);
}

m21rawtensor math21_ml_function_conv_node_getRawTensorToCpu(mlfunction_node *fnode, const char *varName) {
    auto *f = (FnConv *) fnode->function;
    return f->getRawTensorToCpu(varName);
}

void math21_ml_function_conv_node_log(const mlfunction_node *fnode, const char *varName) {
    auto *f = (const FnConv *) fnode->function;
    f->log(varName);
}

void math21_ml_function_conv_node_merge_to(mlfunction_node *fnode, mlfunction_node *fbase) {
    auto *f = (FnConv *) fnode->function;
    auto *fb = (FnConv *) fbase->function;
    f->mergeTo(fb);
}

void math21_ml_function_conv_node_scale(mlfunction_node *fnode, float s) {
    auto *f = (FnConv *) fnode->function;
    f->scale(s);
}

void math21_ml_function_conv_node_pull_wrapper(mlfunction_node *fnode, NumB useRolling) {
    auto *f = (FnConv *) fnode->function;
    f->pullWrapper(useRolling);
}

void math21_ml_function_conv_node_push_by_wrapper(mlfunction_node *fnode, mlfunction_node *fbase, NumB useRolling) {
    auto *f = (FnConv *) fnode->function;
    auto *fb = (FnConv *) fbase->function;
    f->pushByWrapper(fb, useRolling);
}

void math21_ml_function_conv_node_load_theta(mlfunction_node *fnode, FILE *fp) {
    auto *f = (FnConv *) fnode->function;
    f->loadTheta(fp);
}

void math21_ml_function_conv_node_save_theta(mlfunction_node *fnode, FILE *fp) {
    auto *f = (FnConv *) fnode->function;
    f->saveTheta(fp);
}

const char *math21_ml_function_conv_node_getName(const mlfunction_node *fnode) {
    auto *f = (const FnConv *) fnode->function;
    return f->name;
}

void math21_ml_function_conv_node_reset(mlfunction_node *fnode) {
    auto *f = (FnConv *) fnode->function;
    fnode->mini_batch_size = f->batch;
    fnode->x_dim[0] = f->nch_X;
    fnode->x_dim[1] = f->nr_X;
    fnode->x_dim[2] = f->nc_X;
    fnode->y_dim[0] = f->nch_Y;
    fnode->y_dim[1] = f->nr_Y;
    fnode->y_dim[2] = f->nc_Y;
    fnode->x_size = fnode->x_dim[0] * fnode->x_dim[1] * fnode->x_dim[2];
    fnode->y_size = fnode->y_dim[0] * fnode->y_dim[1] * fnode->y_dim[2];
    fnode->y = f->Y;
    fnode->dy = f->dY;
}

FnConv *math21_ml_function_conv_create(
        mlfunction_node *fnode, int mini_batch_size, int nr_X, int nc_X, int nch_X, int nch_Y,
        int num_group, int k_size, int stride, int padding, MATH21_FUNCTION_ACTIVATION_TYPE activation,
        int is_batch_normalize, int adam) {
    auto *f = new FnConv();
    f->create(mini_batch_size, nr_X, nc_X, nch_X, nch_Y,
              num_group, k_size, stride, padding, activation, is_batch_normalize, adam);
    if (fnode) {
        fnode->type = mlfnode_type_conv;
        fnode->function = f;
        fnode->saveState = math21_ml_function_conv_node_saveState;
        fnode->getGlobalSpaceSize = math21_ml_function_conv_node_getGlobalSpaceSize;
        fnode->set_mbs = math21_ml_function_conv_node_set_mbs;
        fnode->resize = math21_ml_function_conv_node_resize;
        fnode->forward = math21_ml_function_conv_node_forward;
        fnode->backward = math21_ml_function_conv_node_backward;
        fnode->update = math21_ml_function_conv_node_update;
        fnode->log = math21_ml_function_conv_node_log;
        fnode->getDataToCpu = math21_ml_function_conv_node_getDataToCpu;
        fnode->getRawTensorToCpu = math21_ml_function_conv_node_getRawTensorToCpu;
        fnode->mergeTo = math21_ml_function_conv_node_merge_to;
        fnode->scale = math21_ml_function_conv_node_scale;
        fnode->pull = math21_ml_function_conv_node_pull_wrapper;
        fnode->pushBy = math21_ml_function_conv_node_push_by_wrapper;
        fnode->loadTheta = math21_ml_function_conv_node_load_theta;
        fnode->saveTheta = math21_ml_function_conv_node_save_theta;
        fnode->getName = math21_ml_function_conv_node_getName;
        math21_ml_function_conv_node_reset(fnode);
        f->fnode = fnode;
    }
    return f;
}
