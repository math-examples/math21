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
#include "route.h"

using namespace math21;

void math21_ml_function_route_node_reset(mlfunction_node *fnode);

FnRoute *math21_ml_function_route_create(
        mlfunction_node *fnode, const mlfunction_net *fnet, int mini_batch_size,
        const VecN &inputLayers);

// route specified convolutional layers
void math21_ml_function_route_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                    const mlfunction_node *finput, m21list *options) {

//    NumB debug = 1;
    NumB debug = 0;
    char *l = math21_function_option_find(options, "layers");
    int len = strlen(l);
    if (!l) math21_error("Route Layer must specify input layers");
    int num_layer = 1;
    int i;
    for (i = 0; i < len; ++i) {
        if (l[i] == ',') ++num_layer;
    }

    VecN inputLayers(num_layer);
    for (i = 0; i < num_layer; ++i) {
        int index = atoi(l);
        l = strchr(l, ',') + 1;
        int index_fnode = fnode->id - 1;
        if (debug) {
            m21log("index_fnode", index_fnode);
            m21log("index", index);
        }
        if (index < 0) index = index_fnode + index;
        inputLayers(i + 1) = index;
    }
    math21_ml_function_route_create(fnode, fnet, finput->mini_batch_size, inputLayers);
}

void math21_ml_function_route_node_saveState(const mlfunction_node *fnode, FILE *file) {
    auto &f = *(FnRoute *) fnode->function;
    f.saveState(file);
}

void math21_ml_function_route_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    auto *f = (FnRoute *) fnode->function;
    f->mini_batch_size = mini_batch_size;
}

void math21_ml_function_route_node_resize(mlfunction_node *fnode, const mlfunction_net *fnet, int nr_X, int nc_X) {
    auto *f = (FnRoute *) fnode->function;
    f->resize(fnet);
    math21_ml_function_route_node_reset(fnode);
}

void math21_ml_function_route_node_forward(mlfunction_node *fnode, mlfunction_net *fnet, mlfunction_node *finput) {
    auto &f = *(FnRoute *) fnode->function;
    f.forward(fnet);
}

void math21_ml_function_route_node_backward(mlfunction_node *fnode, mlfunction_net *fnet, mlfunction_node *finput) {
    auto &f = *(FnRoute *) fnode->function;
    f.backward(fnet);
}

void math21_ml_function_route_node_log(const mlfunction_node *fnode, const char *varName) {
    auto &f = *(const FnRoute *) fnode->function;
    f.log(varName);
}

const char *math21_ml_function_route_node_getName(const mlfunction_node *fnode) {
    auto *f = (const FnRoute *) fnode->function;
    return f->name;
}

void math21_ml_function_route_node_reset(mlfunction_node *fnode) {
    auto *f = (const FnRoute *) fnode->function;
    fnode->mini_batch_size = f->mini_batch_size;
    fnode->y_dim[0] = (int) f->y.dim(2);
    fnode->y_dim[1] = (int) f->y.dim(3);
    fnode->y_dim[2] = (int) f->y.dim(4);
    fnode->x_size = (int) (f->y.size() / f->y.dim(1));
    fnode->y_size = (int) (f->y.size() / f->y.dim(1));
    fnode->y = (PtrR32Wrapper) f->y.getDataAddressWrapper();
    fnode->dy = (PtrR32Wrapper) f->dy.getDataAddressWrapper();
}

FnRoute *math21_ml_function_route_create(
        mlfunction_node *fnode, const mlfunction_net *fnet,
        int mini_batch_size, const VecN &inputLayers) {
    auto *f = new FnRoute();
    NumN deviceType = math21_gpu_is_available() ? m21_device_type_gpu : m21_device_type_default;
    f->create(fnet, mini_batch_size, inputLayers, deviceType);
    if (fnode) {
        fnode->type = mlfnode_type_route;
        fnode->function = f;
        fnode->saveState = math21_ml_function_route_node_saveState;
        fnode->set_mbs = math21_ml_function_route_node_set_mbs;
        fnode->resize = math21_ml_function_route_node_resize;
        fnode->forward = math21_ml_function_route_node_forward;
        fnode->backward = math21_ml_function_route_node_backward;
        fnode->log = math21_ml_function_route_node_log;
        fnode->getName = math21_ml_function_route_node_getName;
        math21_ml_function_route_node_reset(fnode);
    }
    return f;
}