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
#include "yolo.h"

using namespace math21;

void math21_ml_function_yolo_node_reset(mlfunction_node *fnode);

FnYolo *
math21_ml_function_yolo_create(mlfunction_node *fnode, int mini_batch_size, int nc_grids, int nr_grids, int num_box,
                               int total_prior, int *prior_mask, int num_class, int max_boxes);

int *math21_ml_function_yolo_read_map(const char *filename);

int *math21_ml_function_yolo_parse_mask(const char *a, int *num) {
    int *mask = 0;
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == ',') ++n;
        }
        mask = (int *) math21_vector_calloc_cpu(n, sizeof(int));
        for (i = 0; i < n; ++i) {
            int val = atoi(a);
            mask[i] = val;
            a = strchr(a, ',') + 1;
        }
        *num = n;
    }
    return mask;
}

void math21_ml_function_yolo_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                   const mlfunction_node *finput, m21list *options) {
    int classes = math21_function_option_find_int(options, "classes", 20);
    int total = math21_function_option_find_int(options, "num", 1);
    int num = total;

    const char *a = math21_function_option_find_str(options, "mask", 0);
    int *mask = math21_ml_function_yolo_parse_mask(a, &num);
    int max_boxes = math21_function_option_find_int_quiet(options, "max", 90);
    FnYolo *f = math21_ml_function_yolo_create(fnode, finput->mini_batch_size, finput->y_dim[2],
                                               finput->y_dim[1], num, total, mask, classes, max_boxes);

    f->ignore_thresh = math21_function_option_find_float(options, "ignore_thresh", .5);
    f->truth_thresh = math21_function_option_find_float(options, "truth_thresh", 1);
    f->onlyforward = math21_function_option_find_int_quiet(options, "onlyforward", 0);
    const char *map_file = math21_function_option_find_str(options, "map", 0);
    if (map_file) f->map = math21_ml_function_yolo_read_map(map_file);
    int index_fnode = fnode->id - 1;
    f->net_index = index_fnode;

    f->jitter = math21_function_option_find_float(options, "jitter", .2);
    f->random = math21_function_option_find_int_quiet(options, "random", 0);

    a = math21_function_option_find_str(options, "anchors", 0);
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == ',') ++n;
        }
        for (i = 0; i < n; ++i) {
            float bias = atof(a);
            f->biases[i] = bias;
            a = strchr(a, ',') + 1;
        }
    }
}

void math21_ml_function_yolo_node_saveState(const mlfunction_node *fnode, FILE *file) {
    auto f = (FnYolo *) fnode->function;
    f->saveState(file);
}

float math21_ml_function_yolo_node_getCost(mlfunction_node *fnode) {
    FnYolo *f = (FnYolo *) fnode->function;
    return *f->cost;
}

void math21_ml_function_yolo_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    FnYolo *f = (FnYolo *) fnode->function;
    f->batch = mini_batch_size;
}

void math21_ml_function_yolo_node_resize(mlfunction_node *fnode, const mlfunction_net *fnet, int nr_X, int nc_X) {
    auto *f = (FnYolo *) fnode->function;
    f->resize(nr_X, nc_X);
    math21_ml_function_yolo_node_reset(fnode);
}

void math21_ml_function_yolo_node_forward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto *f = (FnYolo *) fnode->function;
    f->net_train = net->is_train;
#if defined(MATH21_FLAG_USE_CPU)
    f->net_truth = net->data_y_wrapper;
#else
    f->net_truth = net->data_y_cpu;
#endif
    f->net_h = net->data_x_dim[1];
    f->net_w = net->data_x_dim[2];
    f->forward(finput);
}

void math21_ml_function_yolo_node_backward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    FnYolo *f = (FnYolo *) fnode->function;
    f->net_train = net->is_train;
#if defined(MATH21_FLAG_USE_CPU)
    f->net_truth = net->data_y_wrapper;
#else
    f->net_truth = net->data_y_cpu;
#endif
    f->net_h = net->data_x_dim[1];
    f->net_w = net->data_x_dim[2];
    f->backward(finput);
}

void math21_ml_function_yolo_node_log(const mlfunction_node *fnode, const char *varName) {
    auto *f = (const FnYolo *) fnode->function;
    f->log(varName);
}

const char *math21_ml_function_yolo_node_getName(const mlfunction_node *fnode) {
    auto *f = (const FnYolo *) fnode->function;
    return f->name;
}

void math21_ml_function_yolo_node_reset(mlfunction_node *fnode) {
    FnYolo *f = (FnYolo *) fnode->function;
    fnode->mini_batch_size = f->batch;
    fnode->y_dim[0] = f->out_c;
    fnode->y_dim[1] = f->out_h;
    fnode->y_dim[2] = f->out_w;
    fnode->y_size = fnode->y_dim[0] * fnode->y_dim[1] * fnode->y_dim[2];
#if defined(MATH21_FLAG_USE_CPU)
    fnode->y = f->output;
    fnode->dy = f->delta;
#else
    fnode->y = f->output_gpu;
    fnode->dy = f->delta_gpu;
#endif
}

int *math21_ml_function_yolo_read_map(const char *filename) {
    int n = 0;
    int *map = 0;
    char *str;
    FILE *file = fopen(filename, "r");
    if (!file) math21_file_error(filename);
    while ((str = math21_file_get_line_c(file))) {
        ++n;
        map = (int *) math21_vector_realloc_cpu(map, n * sizeof(int));
        map[n - 1] = atoi(str);
    }
    return map;
}

FnYolo *
math21_ml_function_yolo_create(mlfunction_node *fnode, int mini_batch_size, int nc_grids, int nr_grids, int num_box,
                               int total_prior, int *prior_mask, int num_class, int max_boxes) {
    FnYolo *f = new FnYolo();
    f->create(mini_batch_size, nc_grids, nr_grids, num_box,
              total_prior, prior_mask, num_class, max_boxes);
    if (fnode) {
        fnode->type = mlfnode_type_yolo;
        fnode->function = f;
        fnode->saveState = math21_ml_function_yolo_node_saveState;
        fnode->getCost = math21_ml_function_yolo_node_getCost;
        fnode->set_mbs = math21_ml_function_yolo_node_set_mbs;
        fnode->resize = math21_ml_function_yolo_node_resize;
        fnode->forward = math21_ml_function_yolo_node_forward;
        fnode->backward = math21_ml_function_yolo_node_backward;
        fnode->log = math21_ml_function_yolo_node_log;
        fnode->getName = math21_ml_function_yolo_node_getName;
        math21_ml_function_yolo_node_reset(fnode);
    }
    return f;
}