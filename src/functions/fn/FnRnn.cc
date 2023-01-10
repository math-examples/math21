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
#include "FnRnn.h"

namespace math21 {

    FnRnn::FnRnn() {
        init();
    }

    FnRnn::~FnRnn() {}

    void FnRnn::init() {
        name = 0;
        inputs = 0;
        outputs = 0;
        batch = 0;
        steps = 0;

        input_layer = 0;
        self_layer = 0;
        output_layer = 0;

        state = 0;
        prev_state = 0;
        delta = 0;
        output = 0;
    }

    void FnRnn::create(
            int batch_size, int input_size, int output_size,
            int n_time_step, MATH21_FUNCTION_ACTIVATION_TYPE activation, int is_use_bias, int is_batch_normalize,
            int is_adam) {
        fprintf(stdout, "rnn layer: input_size %d, output_size %d\n", input_size, output_size);
        int rnn_batch_size = batch_size / n_time_step;
        FnRnn *f = (FnRnn *) math21_vector_calloc_cpu(1, sizeof(FnRnn));
        f->batch = rnn_batch_size;
        f->steps = n_time_step;
        f->inputs = input_size;

        // added by YE
        f->state = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);
        f->prev_state = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);

        fprintf(stdout, "\t\t");

        f->input_layer = new FnFullyConnected(rnn_batch_size, n_time_step,
                                              input_size,
                                              output_size, activation,
                                              is_use_bias,
                                              is_batch_normalize,
                                              is_adam, "fc_input");
        fprintf(stdout, "\t\t");
        f->self_layer = new FnFullyConnected(rnn_batch_size, n_time_step,
                                             output_size,
                                             output_size, activation,
                                             is_use_bias,
                                             is_batch_normalize, is_adam, "fc_self");
        fprintf(stdout, "\t\t");
        f->output_layer = new FnFullyConnected(rnn_batch_size, n_time_step,
                                               output_size,
                                               output_size, activation,
                                               is_use_bias,
                                               is_batch_normalize,
                                               is_adam, "fc_output");
        f->outputs = output_size;
        f->output = f->output_layer->output;
        f->delta = f->output_layer->delta;
    }


    void FnRnn::resize(const mlfunction_net *fnet) {

    }

    void FnRnn::log(const char *varName) const {

    }

    void FnRnn::reset() {
        auto f = this;
        f->input_layer->reset();
        f->self_layer->reset();
        f->output_layer->reset();
    }

    // h(t) = tanh( Whh * h(t-1) + Wxh * x(t) + bh)
    // y(t) = Why*h(t) + by
    // y_x = W_xh * x + b_x
    // y_h = W_hh * h(t-1) + b_h
    // h(t) <- y_h + y_x
    // y = W_hy * h(t) + b_y
    // h_pre, h(1), ..., h(t), ..., h(T)
    void FnRnn::forward(mlfunction_node *finput, int is_train) {
        auto f = this;
        int i;

        FnFullyConnected input_layer = *f->input_layer;
        FnFullyConnected self_layer = *f->self_layer;
        FnFullyConnected output_layer = *f->output_layer;

        if (is_train) {
            math21_vector_set_wrapper(f->outputs * f->batch * f->steps, 0, f->delta, 1);
            // h_pre <- h
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->state, 1, f->prev_state, 1);
        }

        mlfunction_node finput_fc0 = {0};
        mlfunction_node *finput_fc = &finput_fc0;
        for (i = 0; i < f->steps; ++i) {

            // y_x = W_xh * x + b_x
            finput_fc->y = finput->y + i * f->inputs * f->batch;;
            input_layer.forward(finput_fc, is_train);


            // y_h = W_hh * h(t-1) + b_h
            finput_fc->y = f->state;
            self_layer.forward(finput_fc, is_train);

            // h <- y_h + y_x
            // h(t) <- y_h(t) + y_x(t)
            math21_vector_set_wrapper(f->outputs * f->batch, 0, f->state, 1);
            math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, input_layer.output, 1, f->state, 1);
            math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, self_layer.output, 1, f->state, 1);


            // y = W_hy * h(t) + b_y
            finput_fc->y = f->state;
            output_layer.forward(finput_fc, is_train);

            input_layer.increaseByTime(1);
            self_layer.increaseByTime(1);
            output_layer.increaseByTime(1);
        }
        f->reset();
    }

    // cpu not check
    // => dL/dWhy, dL/dby, dL/dWhh, dL/dWxh, dL/dbh
    void FnRnn::backward(mlfunction_node *finput, int is_train) {
        auto f = this;
        int i;
        FnFullyConnected input_layer = *f->input_layer;
        FnFullyConnected self_layer = *f->self_layer;
        FnFullyConnected output_layer = *f->output_layer;

        input_layer.increaseByTime(f->steps - 1);
        self_layer.increaseByTime(f->steps - 1);
        output_layer.increaseByTime(f->steps - 1);

        PtrR32Wrapper last_input = input_layer.output;
        PtrR32Wrapper last_self = self_layer.output;
        mlfunction_node finput_fc0 = {0};
        mlfunction_node *finput_fc = &finput_fc0;
        for (i = f->steps - 1; i >= 0; --i) {
            // todo: check and remove the following lines because f->state is computed already elsewhere.
            // checked once
            // h = y_h + y_x
            // h(t) = y_h(t) + y_x(t)
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, input_layer.output, 1, f->state, 1);
            math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, self_layer.output, 1, f->state, 1);

            finput_fc->y = f->state;
            finput_fc->dy = self_layer.delta; // dL/dh = dL/dy_h
            // dL/dy => dL/dh(t) => dL/dy_h
            // dL/dh(t) = (dL/dy)*(dy/dh(t)) + (dL/dh(t) at t+1)
            output_layer.backward(finput_fc, is_train);

            if (i != 0) {
                // h = y_h + y_x
                // h(t-1) = y_h(t-1) + y_x(t-1)
                math21_vector_set_wrapper(f->outputs * f->batch, 0, f->state, 1);
                math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, input_layer.output - f->outputs * f->batch, 1,
                                               f->state, 1);
                math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, self_layer.output - f->outputs * f->batch, 1,
                                               f->state, 1);
            } else {
                // h <- h(0) <- h_pre
                math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->prev_state, 1, f->state, 1);
            }

            // dL/dh = dL/dy_h = dL/dy_x => dL/dy_x
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, self_layer.delta, 1, input_layer.delta,
                                                     1);

            // y_h = W_hh * h(t-1) + b_h
            // dL/y_h => (dL/dh(t-1) at t)
            finput_fc->y = f->state;
            finput_fc->dy = (i > 0) ? self_layer.delta - f->outputs * f->batch : math21_vector_getEmpty_R32_wrapper();
            self_layer.backward(finput_fc, is_train);

            // y_x = W_xh * x + b_x
            // dL/y_x => dL/dx(t)
            finput_fc->y = finput->y + i * f->inputs * f->batch;
            if (!math21_vector_isEmpty_wrapper(finput->dy)) finput_fc->dy = finput->dy + i * f->inputs * f->batch;
            else finput_fc->dy = math21_vector_getEmpty_R32_wrapper();
            input_layer.backward(finput_fc, is_train);

            input_layer.increaseByTime(-1);
            self_layer.increaseByTime(-1);
            output_layer.increaseByTime(-1);
        }

        // restore h for next forward or just restore it.
        // h <- h(T) = y_h(T) + y_x(T)
        math21_vector_set_wrapper(f->outputs * f->batch, 0, f->state, 1);
        math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, last_input, 1, f->state, 1);
        math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, last_self, 1, f->state, 1);
        f->reset();
    }

    void FnRnn::update(OptUpdate *optUpdate) {
        auto f = this;
        f->input_layer->update(optUpdate);
        f->self_layer->update(optUpdate);
        f->output_layer->update(optUpdate);
    }

    void FnRnn::saveState(FILE *file) const {
        auto f = this;
        math21_vector_serialize_c_wrapper(file, f->state, f->batch * f->outputs);
//    f->input_layer->saveState(file);
//    f->self_layer->saveState(file);
//    f->output_layer->saveState(file);
    }

    void FnRnn::resetState(int b) {
        auto f = this;
        math21_vector_set_wrapper(f->outputs, 0, f->state + f->outputs * b, 1);
    }

    void FnRnn::setMbs(int mini_batch_size) {
        auto f = this;
        f->batch = mini_batch_size;
        f->input_layer->setMbs(mini_batch_size);
        f->self_layer->setMbs(mini_batch_size);
        f->output_layer->setMbs(mini_batch_size);
    }

    void FnRnn::saveThetaOrderBwsmv(FILE *fp) {
        auto f = this;
        f->input_layer->saveThetaOrderBwsmv(fp);
        f->self_layer->saveThetaOrderBwsmv(fp);
        f->output_layer->saveThetaOrderBwsmv(fp);
    }

    void FnRnn::loadThetaOrderBwsmvFlipped(FILE *fp, int flipped) {
        auto f = this;
        f->input_layer->loadThetaOrderBwsmvFlipped(fp, flipped);
        f->self_layer->loadThetaOrderBwsmvFlipped(fp, flipped);
        f->output_layer->loadThetaOrderBwsmvFlipped(fp, flipped);
    }

    void FnRnn::saveTheta(FILE *fp) {
        auto f = this;
        f->input_layer->saveTheta(fp);
        f->self_layer->saveTheta(fp);
        f->output_layer->saveTheta(fp);
    }

    void FnRnn::loadTheta(FILE *fp) {
        auto f = this;
        f->input_layer->loadTheta(fp);
        f->self_layer->loadTheta(fp);
        f->output_layer->loadTheta(fp);
    }
}
