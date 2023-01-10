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
#include "FnFullyConnected.h"

namespace math21 {

    FnFullyConnected::FnFullyConnected() {
        init();
    }

    FnFullyConnected::FnFullyConnected(
            int rnn_batch_size, int n_time_step, int input_size, int output_size,
            MATH21_FUNCTION_ACTIVATION_TYPE activation, int is_use_bias, int is_batch_normalize, int is_adam,
            const char *name) {
        init();
        createWithNTimeStep(rnn_batch_size, n_time_step, input_size, output_size,
                            activation, is_use_bias, is_batch_normalize, is_adam,
                            name);
    }

    FnFullyConnected::~FnFullyConnected() {}

    void FnFullyConnected::init() {
        name = 0;
        learning_rate_scale = 0;
        inputs = 0;
        outputs = 0;
        batch = 0;
        h = 0;
        w = 0;
        c = 0;
        out_h = 0;
        out_w = 0;
        out_c = 0;
        delta = 0;
        output = 0;
        weight_updates = 0;
        bias_updates = 0;
        weights = 0;
        weights_cpu = 0;
        biases = 0;
        biases_cpu = 0;
        is_use_bias = 0;
        m = 0;
        v = 0;
        bias_m = 0;
        bias_v = 0;
        nweights = 0;
//        activation = 0;
        bn = 0;
        flipped = 0;
        total_mbs = 0;
        n_time_step = 0;
        i_time_step = 0;
    }

    // Z = h(Y), Y = X*W.t + b
    void FnFullyConnected::create(
            int batch_size, int input_size, int output_size,
            MATH21_FUNCTION_ACTIVATION_TYPE activation, int is_use_bias, int is_batch_normalize, int is_adam,
            const char *name) {
        auto *f = this;
        f->learning_rate_scale = 1;

        f->inputs = input_size;
        f->outputs = output_size;
        f->batch = batch_size;
        f->h = 1;
        f->w = 1;
        f->c = input_size;
        f->out_h = 1;
        f->out_w = 1;
        f->out_c = output_size;

        f->output = math21_vector_create_with_default_value_wrapper(f->batch * f->outputs, 0);
        f->delta = math21_vector_create_with_default_value_wrapper(f->batch * f->outputs, 0);

        int nweights = output_size * input_size;
        f->nweights = nweights;
        float scale = sqrt(2. / input_size);
        int i;
#if defined(MATH21_FLAG_USE_CPU)
        f->weights = math21_vector_create_with_default_value_wrapper(nweights, 0);
    for (i = 0; i < nweights; ++i) f->weights[i] = scale * math21_pr_rand_uniform(-1, 1);
#else
        f->weights_cpu = math21_vector_create_with_default_value_cpu(nweights, 0);
//    for (i = 0; i < nweights; ++i) f->weights_cpu[i] = scale * math21_pr_rand_uniform(-1, 1);
        if (math21_ml_function_tool_is_debug()) {
            for (i = 0; i < nweights; ++i) f->weights_cpu[i] = 1;
        } else {
            for (i = 0; i < nweights; ++i) f->weights_cpu[i] = scale * math21_pr_rand_uniform(-1, 1);
        }
        f->weights = math21_vector_create_from_cpuvector_wrapper(nweights, f->weights_cpu, 1);
#endif
        f->weight_updates = math21_vector_create_with_default_value_wrapper(nweights, 0);

        if (is_batch_normalize) {
            mlfunction_node finput = {0};
            finput.y = f->output;
            finput.dy = f->delta;
            f->bn = new FnBatchnorm();
            f->bn->create(0, &finput, batch_size, f->out_w, f->out_h, f->out_c, is_adam);
        } else {
            if (is_use_bias) {
                f->is_use_bias = is_use_bias;
#ifndef MATH21_FLAG_USE_CPU
                f->biases_cpu = math21_vector_create_with_default_value_cpu(output_size, 0);
#endif
                f->biases = math21_vector_create_with_default_value_wrapper(output_size, 0);
                f->bias_updates = math21_vector_create_with_default_value_wrapper(output_size, 0);
            } else {
                // f->biases is empty already.
            }
        }

        if (is_adam) {
            f->m = math21_vector_create_with_default_value_wrapper(nweights, 0);
            f->v = math21_vector_create_with_default_value_wrapper(nweights, 0);
            if (!f->bn) {
                if (f->is_use_bias) {
                    f->bias_m = math21_vector_create_with_default_value_wrapper(f->outputs, 0);
                    f->bias_v = math21_vector_create_with_default_value_wrapper(f->outputs, 0);
                }
            }
        }

        f->activation = activation;
        f->total_mbs = f->batch;
        f->n_time_step = 1;
        f->i_time_step = 0;
        if (!name) {
            name = "fully connected";
        }
        f->name = math21_string_create_from_string(name);
    }

    void FnFullyConnected::createWithNTimeStep(
            int rnn_batch_size, int n_time_step, int input_size, int output_size,
            MATH21_FUNCTION_ACTIVATION_TYPE activation, int is_use_bias, int is_batch_normalize, int is_adam,
            const char *name) {
        auto *f = this;
        math21_tool_assert(n_time_step > 0);
        create(rnn_batch_size * n_time_step,
               input_size,
               output_size, activation,
               is_use_bias,
               is_batch_normalize,
               is_adam, name);
        f->n_time_step = n_time_step;
        f->batch = rnn_batch_size;
        if (f->bn) {
            f->bn->n_time_step = n_time_step;
            f->bn->mini_batch_size = rnn_batch_size;
        }
    }

    void FnFullyConnected::resize(const mlfunction_net *fnet) {

    }

    void FnFullyConnected::log(const char *varName) const {
        auto *f = this;
        std::string _varNameNew;
        if (!math21_ml_function_tool_varName_check(f->name, varName, _varNameNew)) {
            return;
        }
        varName = _varNameNew.c_str();

        if (math21_string_is_equal(varName, "summary")) {
            if (f->n_time_step > 1) {
                fprintf(stdout, "%s with time: (%d, %d, %d, %d, %d) -> (%d, %d, %d, %d, %d)\n", f->name,
                        f->h, f->w, f->c, f->n_time_step, f->batch, f->out_h, f->out_w, f->out_c, f->n_time_step,
                        f->batch);
            } else {
                fprintf(stdout, "%s: (%d, %d, %d, %d) -> (%d, %d, %d, %d)\n", f->name,
                        f->h, f->w, f->c, f->total_mbs, f->out_h, f->out_w, f->out_c, f->total_mbs);
            }
            return;
        }
        fprintf(stdout, "%s:\n", f->name);
        std::string name = varName;
        if (name == "y") {
            if (f->out_h == 1 && f->out_w == 1) {
                math21_tensor_2d_float_log_wrapper(varName, f->output, f->total_mbs, f->out_c);
            } else {
                math21_tensor_4d_float_log_wrapper(varName, f->output, f->total_mbs, f->out_c, f->out_h, f->out_w);
            }
        } else if (name == "dy") {
            if (f->out_h == 1 && f->out_w == 1) {
                math21_tensor_2d_float_log_wrapper(varName, f->delta, f->total_mbs, f->out_c);
            } else {
                math21_tensor_4d_float_log_wrapper(varName, f->delta, f->total_mbs, f->out_c, f->out_h, f->out_w);
            }
        } else if (name == "W") {
            math21_tensor_2d_float_log_wrapper(varName, f->weights, f->outputs, f->inputs);
        } else if (name == "dW") {
            math21_tensor_2d_float_log_wrapper(varName, f->weight_updates, f->outputs, f->inputs);
        } else if (name == "b") {
            if (f->bn) {

            } else {
                if (f->is_use_bias) {
                    math21_tensor_1d_float_log_wrapper(varName, f->biases, f->outputs);
                }
            }
        } else if (name == "db") {
            if (f->bn) {

            } else {
                if (f->is_use_bias) {
                    math21_tensor_1d_float_log_wrapper(varName, f->bias_updates, f->outputs);
                }
            }
        } else {
            m21log("no variable name ", varName);
        }
    }


    // Z = h(Y), Y = W*X + b, or Y = X*W.t + b
    void FnFullyConnected::forward(mlfunction_node *finput, int is_train) {
        auto *f = this;
        if (is_train) {
            if (f->i_time_step == 0) {
                math21_vector_set_wrapper(f->total_mbs * f->outputs, 0, f->delta, 1);
            }
        }
        math21_vector_set_wrapper(f->outputs * f->batch, 0, f->output, 1);
        int sb = f->batch;
        int sx = f->inputs;
        int sy = f->outputs;
        PtrR32Wrapper x = finput->y;
        PtrR32Wrapper w = f->weights;
        PtrR32Wrapper y = f->output;

        // Y += X*W.t
        math21_matrix_multiply_k1AB_add_k2C_similar_wrapper(0, 1, sb, sy, sx, 1, x, sx, w, sx, 1, y, sy);

        if (f->bn) {
            // Y = BN(Y)
            FnBatchnorm *fbn = f->bn;
            fbn->is_train = is_train;
            fbn->forward(0);
        } else {
            // Y += b
            if (f->is_use_bias) {
                math21_function_conv2d_bias_forward_wrapper(f->output, f->biases, f->batch, f->outputs, 1);
            }
        }

        // Z = h(Y)
        math21_function_activation_vector_wrapper(f->output, f->outputs * f->batch, f->activation);
    }

    // Z = h(Y), Y = W*X + b, or Y = X*W.t + b
    // dL/dZ => dL/dW, dL/dX
    void FnFullyConnected::backward(mlfunction_node *finput, int is_train) {
        auto *f = this;
        if (f->bn) {
            math21_vector_clip_wrapper(f->outputs * f->batch, 1, f->delta, 1);
        }

        // dL/dY = dL/dZ *.ele h.d(Y)
        math21_function_activation_gradient_vector_wrapper(f->output, f->outputs * f->batch, f->activation, f->delta);

        if (f->bn) {
            FnBatchnorm *fbn = f->bn;
            fbn->is_train = is_train;
            fbn->backward(0);
        } else {
            if (f->is_use_bias) {
                // dL/db += sum(dL/dY(i))
                math21_function_conv2d_bias_backward_wrapper(f->bias_updates, f->delta, f->batch, f->outputs, 1);
            }
        }

        int m = f->outputs;
        int k = f->batch;
        int n = f->inputs;
        PtrR32Wrapper a = f->delta;
        PtrR32Wrapper b = finput->y;
        PtrR32Wrapper c = f->weight_updates;
        // dL/dW += dL/dY * X.t
        math21_matrix_multiply_k1AB_add_k2C_similar_wrapper(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);

        m = f->batch;
        k = f->outputs;
        n = f->inputs;

        a = f->delta;
        b = f->weights;
        c = finput->dy;

        // dL/dX = W.t * dL/dY
        // but here is addTo dX ...
        if (!math21_vector_isEmpty_wrapper(finput->dy)) {
            math21_matrix_multiply_k1AB_add_k2C_similar_wrapper(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
        }
    }

    void FnFullyConnected::update(OptUpdate *optUpdate) {
        auto *f = this;
        OptUpdate_Adam *a = 0;
        if (optUpdate->type == OptUpdateType_Adam) {
            a = (OptUpdate_Adam *) optUpdate->detail;
        }
        float learning_rate = optUpdate->alpha * f->learning_rate_scale;
        float momentum = optUpdate->momentum;
        float decay = optUpdate->decay;
        int batch = optUpdate->mini_batch_size;
        if (a) {
            math21_optimization_adam_update_wrapper(f->weights, f->weight_updates, f->m, f->v, a->beta1, a->beta2,
                                                    a->eps, decay, learning_rate, f->inputs * f->outputs, batch, a->t);
            if (f->bn) {
                f->bn->learning_rate_scale = f->learning_rate_scale;
                f->bn->update(optUpdate);
            } else {
                if (f->is_use_bias) {
                    math21_optimization_adam_update_wrapper(f->biases, f->bias_updates, f->bias_m, f->bias_v, a->beta1,
                                                            a->beta2, a->eps, decay, learning_rate, f->outputs, batch,
                                                            a->t);
                }
            }
        } else {
            if (f->bn) {
                f->bn->learning_rate_scale = f->learning_rate_scale;
                f->bn->update(optUpdate);
            } else {
                if (f->is_use_bias) {
                    // b = b - alpha * dL/db
                    // f->bias_updates = -dL/db because of loss function L is -L.
                    math21_vector_kx_add_y_wrapper(f->outputs, learning_rate / batch, f->bias_updates, 1, f->biases, 1);
                    // dL/db = momentum * dL/db
                    math21_vector_kx_wrapper(f->outputs, momentum, f->bias_updates, 1);
                }
            }

            math21_vector_kx_add_y_wrapper(f->inputs * f->outputs, -decay * batch, f->weights, 1, f->weight_updates, 1);
            math21_vector_kx_add_y_wrapper(f->inputs * f->outputs, learning_rate / batch, f->weight_updates, 1,
                                           f->weights, 1);
            math21_vector_kx_wrapper(f->inputs * f->outputs, momentum, f->weight_updates, 1);
        }
    }

    void FnFullyConnected::saveState(FILE *file) const {
        auto *f = this;
        math21_vector_serialize_c_wrapper(file, f->output, f->total_mbs * f->outputs);
        math21_vector_serialize_c_wrapper(file, f->delta, f->total_mbs * f->outputs);
        math21_vector_serialize_c_wrapper(file, f->weights, f->nweights);
        math21_vector_serialize_c_wrapper(file, f->weight_updates, f->nweights);
        if (f->bn) {
            f->bn->saveState(file);
        } else {
            if (f->is_use_bias) {
                math21_vector_serialize_c_wrapper(file, f->biases, f->outputs);
                math21_vector_serialize_c_wrapper(file, f->bias_updates, f->outputs);
            }
        }
    }

    void FnFullyConnected::increaseByTime(int time_steps) {
        auto *f = this;
        f->i_time_step += time_steps;

        int num = f->outputs * f->batch * time_steps;
        f->output += num;
        f->delta += num;
        if (f->bn) {
            f->bn->increaseByTime(time_steps);
        }
    }

    void FnFullyConnected::reset() {
        auto f = this;
        int num = f->outputs * f->batch * f->i_time_step;
        f->output -= num;
        f->delta -= num;
        f->i_time_step = 0;

        if (f->bn) {
            f->bn->reset();
        }
    }

    // todo: may deprecate
    void FnFullyConnected::setMbs(int mini_batch_size) {
        auto f = this;
        f->batch = mini_batch_size;
        if (f->bn) {
            f->bn->mini_batch_size = mini_batch_size;
        }
    }

    // merge f to fb
    void FnFullyConnected::mergeTo(FnFullyConnected *fb) {
        auto f = this;
        math21_vector_kx_add_y_cpu(f->nweights, 1, f->weights_cpu, 1, fb->weights_cpu, 1);
        if (f->bn) {
            f->bn->mergeTo(fb->bn);
        } else {
            if (f->is_use_bias) {
                math21_vector_kx_add_y_cpu(f->outputs, 1, f->biases_cpu, 1, fb->biases_cpu, 1);
            }
        }
    }

    void FnFullyConnected::scale(float s) {
        auto f = this;
        math21_vector_kx_cpu(f->nweights, s, f->weights_cpu, 1);
        if (f->bn) {
            f->bn->scale(s);
        } else {
            if (f->is_use_bias) {
                math21_vector_kx_cpu(f->outputs, s, f->biases_cpu, 1);
            }
        }
    }

    void FnFullyConnected::pullWrapper(NumB useRolling) {
        auto f = this;
        math21_vector_pull_wrapper(f->weights, f->weights_cpu, f->nweights);
        if (f->bn) {
            f->bn->pullWrapper(useRolling);
        } else {
            if (f->is_use_bias) {
                math21_vector_pull_wrapper(f->biases, f->biases_cpu, f->outputs);
            }
        }
    }

    void FnFullyConnected::pushWrapper(NumB useRolling) {
        auto f = this;
        f->pushByWrapper(f, useRolling);
    }

    // f is pushed by fb
    void FnFullyConnected::pushByWrapper(FnFullyConnected *fb, NumB useRolling) {
        auto f = this;
        math21_vector_push_wrapper(f->weights, fb->weights_cpu, f->nweights);
        if (f->bn) {
            f->bn->pushByWrapper(fb->bn, useRolling);
        } else {
            if (f->is_use_bias) {
                math21_vector_push_wrapper(f->biases, fb->biases_cpu, f->outputs);
            }
        }
    }

    void FnFullyConnected::saveThetaOrderBwsmv(FILE *fp) {
        auto f = this;
#ifndef MATH21_FLAG_USE_CPU
        f->pullWrapper(1);
#endif

        float *weights = 0;
        float *biases = 0;
        float *scales = 0;
        float *rolling_mean = 0;
        float *rolling_variance = 0;
#if defined(MATH21_FLAG_USE_CPU)
        weights = f->weights;
    if (f->bn) {
        biases = f->bn->biases;
        scales = f->bn->scales;
        rolling_mean = f->bn->rolling_mean;
        rolling_variance = f->bn->rolling_variance;
    } else {
        biases = f->biases;
    }
#else
        weights = f->weights_cpu;
        if (f->bn) {
            biases = f->bn->biases_cpu;
            scales = f->bn->scales_cpu;
            rolling_mean = f->bn->rolling_mean_cpu;
            rolling_variance = f->bn->rolling_variance_cpu;
        } else {
            biases = f->biases_cpu;
        }
#endif

        int num = f->nweights;
        if (biases) {
            fwrite(biases, sizeof(float), f->outputs, fp);
        }
        fwrite(weights, sizeof(float), num, fp);
        if (scales)fwrite(scales, sizeof(float), f->outputs, fp);
        if (rolling_mean)fwrite(rolling_mean, sizeof(float), f->outputs, fp);
        if (rolling_variance)fwrite(rolling_variance, sizeof(float), f->outputs, fp);
    }

    void FnFullyConnected::loadThetaOrderBwsmv(FILE *fp) {
        auto f = this;
        float *weights = 0;
        float *biases = 0;
        float *scales = 0;
        float *rolling_mean = 0;
        float *rolling_variance = 0;
#if defined(MATH21_FLAG_USE_CPU)
        weights = f->weights;
    if (f->bn) {
        biases = f->bn->biases;
        scales = f->bn->scales;
        rolling_mean = f->bn->rolling_mean;
        rolling_variance = f->bn->rolling_variance;
    } else {
        biases = f->biases;
    }
#else
        weights = f->weights_cpu;
        if (f->bn) {
            biases = f->bn->biases_cpu;
            scales = f->bn->scales_cpu;
            rolling_mean = f->bn->rolling_mean_cpu;
            rolling_variance = f->bn->rolling_variance_cpu;
        } else {
            biases = f->biases_cpu;
        }
#endif

        int num = f->nweights;
        if (biases) {
            fread(biases, sizeof(float), f->outputs, fp);
        }
        fread(weights, sizeof(float), num, fp);
        if (scales)fread(scales, sizeof(float), f->outputs, fp);
        if (rolling_mean)fread(rolling_mean, sizeof(float), f->outputs, fp);
        if (rolling_variance)fread(rolling_variance, sizeof(float), f->outputs, fp);
        if (f->flipped) {
            math21_matrix_transpose(weights, f->inputs, f->outputs);
        }
#ifndef MATH21_FLAG_USE_CPU
        f->pushWrapper(1);
#endif
    }

    void FnFullyConnected::loadThetaOrderBwsmvFlipped(FILE *fp, int flipped) {
        auto f = this;
        f->flipped = flipped;
        f->loadThetaOrderBwsmv(fp);
    }

    void FnFullyConnected::saveTheta(FILE *fp) {
        auto f = this;
#ifndef MATH21_FLAG_USE_CPU
        f->pullWrapper(1);
#endif

#if defined(MATH21_FLAG_USE_CPU)
        float * weights = f->weights;
    float * biases = f->biases;
#else
        float *weights = f->weights_cpu;
        float *biases = f->biases_cpu;
#endif

        int num = f->nweights;
        if (f->bn) {
            f->bn->saveTheta(fp, 0);
        } else {
            if (biases) {
                fwrite(biases, sizeof(float), f->outputs, fp);
            }
        }
        fwrite(weights, sizeof(float), num, fp);
    }

    void FnFullyConnected::loadTheta(FILE *fp) {
        auto f = this;
#if defined(MATH21_FLAG_USE_CPU)
        float * weights = f->weights;
    float * biases = f->biases;
#else
        float *weights = f->weights_cpu;
        float *biases = f->biases_cpu;
#endif

        int num = f->nweights;
        if (f->bn) {
            f->bn->loadTheta(fp, 0);
        } else {
            if (biases) {
                fread(biases, sizeof(float), f->outputs, fp);
            }
        }
        fread(weights, sizeof(float), num, fp);
        if (f->flipped) {
            math21_matrix_transpose(weights, f->inputs, f->outputs);
        }
#ifndef MATH21_FLAG_USE_CPU
        f->pushWrapper(1);
#endif
    }
}
