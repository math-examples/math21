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
#include "src/dropout/dropout_cpu.h"
#include "src/dropout/dropout_cuda.h"
#include "src/dropout/dropout_opencl.h"
#include "FnDropout.h"

namespace math21 {

    FnDropout::FnDropout() {
        init();
    }

    FnDropout::FnDropout(mlfunction_node *finput, float rate, int n_time_step, const char *name) {
        init();
        create(finput, rate, n_time_step, name);
    }

    FnDropout::~FnDropout() {}

    void FnDropout::init() {
        name = 0;
        batch = 0;
//        y_dim = 0;
        outputs = 0;
        inputs = 0;
        y = 0;
        dy = 0;
        rate = 0;
        scale = 0;
        rand = 0;
        total_mbs = 0;
        n_time_step = 0;
        i_time_step = 0;
    }

    // finput can have empty vector, but must have its shape.
    void FnDropout::create(mlfunction_node *finput, float rate, int n_time_step, const char *name) {
        auto f = this;
        int inputs = finput->y_size;
        f->rate = rate;
        f->inputs = inputs;
        f->outputs = inputs;
        f->total_mbs = finput->mini_batch_size;
        f->scale = 1. / (1. - rate);

        math21_rawtensor_shape_assign(f->y_dim, finput->y_dim);
        if (math21_rawtensor_size(f->y_dim) == 0) {
            math21_rawtensor_shape_set(f->outputs, f->y_dim);
        }

        math21_tool_assert(n_time_step > 0);
        f->n_time_step = n_time_step;
        f->i_time_step = 0;
        math21_tool_assert(f->total_mbs % n_time_step == 0);
        f->batch = f->total_mbs / n_time_step;

        f->rand = math21_vector_create_with_default_value_wrapper(f->batch * f->inputs, 0);
        f->y = math21_vector_create_with_default_value_wrapper(f->total_mbs * f->outputs, 0);
        f->dy = math21_vector_create_with_default_value_wrapper(f->total_mbs * f->outputs, 0);
        if (!name) {
            name = "dropout";
        }
        f->name = math21_string_create_from_string(name);
    }


    void FnDropout::resize(const mlfunction_net *fnet) {

    }

    void FnDropout::log(const char *varName) const {
        auto f = this;
        if (f->n_time_step > 1) {
            fprintf(stdout, "%s with time: (%d, %d, %d, %d, %d) -> (%d, %d, %d, %d, %d), rate = %.2f\n",
                    f->name,
                    f->y_dim[1], f->y_dim[2], f->y_dim[0], f->n_time_step, f->batch,
                    f->y_dim[1], f->y_dim[2], f->y_dim[0], f->n_time_step, f->batch,
                    f->rate);
        } else {
            fprintf(stdout, "%s: (%d, %d, %d, %d) -> (%d, %d, %d, %d), rate = %.2f\n",
                    f->name,
                    f->y_dim[1], f->y_dim[2], f->y_dim[0], f->total_mbs,
                    f->y_dim[1], f->y_dim[2], f->y_dim[0], f->total_mbs, f->rate);
        }
    }

    void FnDropout::forward(mlfunction_node *finput, int is_train) {
        auto f = this;
        if (is_train) {
            if (f->i_time_step == 0) {
                math21_vector_set_wrapper(f->total_mbs * f->outputs, 0, f->dy, 1);
            }
        }
#if defined(MATH21_FLAG_USE_CPU)
        math21_ml_function_dropout_forward_cpu(f, finput, is_train);
#elif defined(MATH21_FLAG_USE_CUDA)
        math21_ml_function_dropout_forward_cuda(f, finput, is_train);
#elif defined(MATH21_FLAG_USE_OPENCL)
        math21_ml_function_dropout_forward_opencl(f, finput, is_train);
#endif
    }

    void FnDropout::backward(mlfunction_node *finput) {
        auto f = this;
#if defined(MATH21_FLAG_USE_CPU)
        math21_ml_function_dropout_backward_cpu(f, finput);
#elif defined(MATH21_FLAG_USE_CUDA)
        math21_ml_function_dropout_backward_cuda(f, finput);
#elif defined(MATH21_FLAG_USE_OPENCL)
        math21_ml_function_dropout_backward_opencl(f, finput);
#endif
    }

    void FnDropout::saveState(FILE *file) const {
        auto f = this;
        math21_vector_serialize_c_wrapper(file, f->y, f->total_mbs * f->outputs);
        math21_vector_serialize_c_wrapper(file, f->dy, f->total_mbs * f->outputs);
        math21_vector_serialize_c_wrapper(file, f->rand, f->total_mbs * f->outputs);
    }

    void FnDropout::increaseByTime(int time_steps) {
        auto f = this;
        f->i_time_step += time_steps;
        int num = time_steps * f->batch * f->outputs;
        f->y += num;
        f->dy += num;
    }

    void FnDropout::reset() {
        auto f = this;
        int num = f->i_time_step * f->batch * f->outputs;
        f->y -= num;
        f->dy -= num;
        f->i_time_step = 0;
    }

}
