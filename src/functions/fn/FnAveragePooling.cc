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
#include "src/average_pooling/average_pooling_cpu.h"
#include "src/average_pooling/average_pooling_cuda.h"
#include "src/average_pooling/average_pooling_opencl.h"
#include "FnAveragePooling.h"

namespace math21 {

    FnAveragePooling::FnAveragePooling() {
        init();
    }

    FnAveragePooling::~FnAveragePooling() {}

    void FnAveragePooling::init() {
        name = 0;
        batch = 0;
        h = 0;
        w = 0;
        c = 0;
        out_h = 0;
        out_w = 0;
        out_c = 0;
        outputs = 0;
        inputs = 0;
        output = 0;
        delta = 0;
    }

    void FnAveragePooling::create(int batch, int c, int h, int w) {
        auto f = this;
        f->batch = batch;
        f->h = h;
        f->w = w;
        f->c = c;
        f->out_w = 1;
        f->out_h = 1;
        f->out_c = c;
        f->outputs = f->out_c;
        f->inputs = h * w * c;
        f->output = math21_vector_create_with_default_value_wrapper(f->batch * f->outputs, 0);
        f->delta = math21_vector_create_with_default_value_wrapper(f->batch * f->outputs, 0);
        f->name = math21_string_create_from_string("average pooling");
    }

    void FnAveragePooling::resize(int h, int w) {
        auto l = this;
        l->w = w;
        l->h = h;
        l->inputs = h * w * l->c;
    }

    void FnAveragePooling::log(const char *varName) const {
        auto f = this;
        fprintf(stdout, "%s                %4d x%4d x%4d   ->  %4d\n", f->name, f->w, f->h, f->c, f->c);
    }

    void FnAveragePooling::forward(const mlfunction_net *net, const mlfunction_node *finput) {
        auto f = this;
        if (net->is_train) {
            math21_vector_set_wrapper(f->batch * f->outputs, 0, f->delta, 1);
        }
#if defined(MATH21_FLAG_USE_CPU)
        math21_ml_function_average_pooling_forward_cpu(f, finput);
#elif defined(MATH21_FLAG_USE_CUDA)
        math21_ml_function_average_pooling_forward_cuda(f, finput);
#elif defined(MATH21_FLAG_USE_OPENCL)
        math21_ml_function_average_pooling_forward_opencl(f, finput);
#endif
    }

    void FnAveragePooling::backward(mlfunction_node *finput) {
        auto f = this;
#if defined(MATH21_FLAG_USE_CPU)
        math21_ml_function_average_pooling_backward_cpu(f, finput);
#elif defined(MATH21_FLAG_USE_CUDA)
        math21_ml_function_average_pooling_backward_cuda(f, finput);
#elif defined(MATH21_FLAG_USE_OPENCL)
        math21_ml_function_average_pooling_backward_opencl(f, finput);
#endif
    }

    void FnAveragePooling::saveState(FILE *file) const {
        auto f = this;
        math21_vector_serialize_c_wrapper(file, f->output, f->batch * f->outputs);
        math21_vector_serialize_c_wrapper(file, f->delta, f->batch * f->outputs);
    }

}
