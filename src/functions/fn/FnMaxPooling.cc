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
#include "src/max_pooling/max_pooling_cpu.h"
#include "src/max_pooling/max_pooling_cuda.h"
#include "src/max_pooling/max_pooling_opencl.h"
#include "FnMaxPooling.h"

namespace math21 {

    FnMaxPooling::FnMaxPooling() {
        init();
    }

    FnMaxPooling::~FnMaxPooling() {}

    void FnMaxPooling::init() {
        name = 0;
        batch = 0;
        h = 0;
        w = 0;
        c = 0;
        out_h = 0;
        out_w = 0;
        out_c = 0;
        padding = 0;
        size = 0;
        stride = 0;
        indexes = 0;
        outputs = 0;
        inputs = 0;
        output = 0;
        delta = 0;
    }

    void FnMaxPooling::create(int batch, int c, int h, int w, int size, int stride,
                              int padding) {
        auto f = this;
        f->batch = batch;
        f->c = c;
        f->h = h;
        f->w = w;
        f->padding = padding;
        f->out_w = (w + padding - size) / stride + 1;
        f->out_h = (h + padding - size) / stride + 1;
        f->out_c = c;
        f->outputs = f->out_h * f->out_w * f->out_c;
        f->inputs = h * w * c;
        f->size = size;
        f->stride = stride;
        f->indexes = math21_vector_create_from_cpuvector_int_wrapper(f->batch * f->outputs, 0, 1);
        f->output = math21_vector_create_with_default_value_wrapper(f->batch * f->outputs, 0);
        f->delta = math21_vector_create_with_default_value_wrapper(f->batch * f->outputs, 0);
        f->name = math21_string_create_from_string("max pooling");
    }

    void FnMaxPooling::resize(int h, int w) {
        auto l = this;
        l->h = h;
        l->w = w;
        l->inputs = h * w * l->c;

        l->out_w = (w + l->padding - l->size) / l->stride + 1;
        l->out_h = (h + l->padding - l->size) / l->stride + 1;
        l->outputs = l->out_w * l->out_h * l->c;

        l->indexes = math21_vector_resize_with_default_value_int_wrapper(l->indexes, l->outputs * l->batch, 0);
        l->output = math21_vector_resize_with_default_value_wrapper(l->output, l->outputs * l->batch, 0);
        l->delta = math21_vector_resize_with_default_value_wrapper(l->delta, l->outputs * l->batch, 0);
    }

    void FnMaxPooling::log(const char *varName) const {
        auto f = this;
        fprintf(stdout, "%s         %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", f->name, f->size, f->size,
                f->stride, f->w, f->h,
                f->c, f->out_w, f->out_h, f->out_c);
    }


    void FnMaxPooling::forward(const mlfunction_node *finput, int is_train) {
        auto f = this;
        if (is_train) {
            math21_vector_set_wrapper(f->batch * f->outputs, 0, f->delta, 1);
        }
#if defined(MATH21_FLAG_USE_CPU)
        math21_ml_function_max_pooling_forward_cpu(f, finput);
#elif defined(MATH21_FLAG_USE_CUDA)
        math21_ml_function_max_pooling_forward_cuda(f, finput);
#elif defined(MATH21_FLAG_USE_OPENCL)
        math21_ml_function_max_pooling_forward_opencl(f, finput);
#endif
    }

    void FnMaxPooling::backward(mlfunction_node *finput) {
        auto f = this;
#if defined(MATH21_FLAG_USE_CPU)
        math21_ml_function_max_pooling_backward_cpu(f, finput);
#elif defined(MATH21_FLAG_USE_CUDA)
        math21_ml_function_max_pooling_backward_cuda(f, finput);
#elif defined(MATH21_FLAG_USE_OPENCL)
        math21_ml_function_max_pooling_backward_opencl(f, finput);
#endif
    }

    void FnMaxPooling::saveState(FILE *file) const {
        auto f = this;
        math21_vector_serialize_c_wrapper(file, f->output, f->batch * f->outputs);
        math21_vector_serialize_c_wrapper(file, f->delta, f->batch * f->outputs);
//    math21_vector_serialize_int_c_wrapper(file, f->indexes, f->batch * f->outputs);
    }

}
