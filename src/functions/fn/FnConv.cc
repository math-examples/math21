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
#include "FnConvDetail.h"
#include "FnConv.h"

namespace math21 {

    int FnConv::cal_nr_or_nc_Y(int nr_X, int pad, int k_size, int stride) {
        return (nr_X + 2 * pad - k_size) / stride + 1;
    }

    size_t FnConv::get_X_prime_size(int nr_Y, int nc_Y,
                                    int k_size, int nch_X, int num_group) {
        return (size_t) (nch_X / num_group * k_size * k_size) * (nr_Y * nc_Y) * sizeof(float);
    }

    FnConv::FnConv() {
        init();
    }

    FnConv::~FnConv() {}

    void FnConv::init() {
        name = 0;
        nr_X = 0, nc_X = 0, nch_X = 0;
        nr_Y = 0, nc_Y = 0, nch_Y = 0;
        n_group = 0;
        batch = 0;
        stride = 0;
        k_size = 0;
        pad = 0;
        x_size = 0;
        y_size = 0;
        n_W = 0;
        W = 0;
        W_cpu = 0;
        dW = 0;
        b = 0;
        b_cpu = 0;
        db = 0;
        dY = 0;
        Y = 0;
        workspace_size = 0;
//        activation = 0;
        bn = 0;
        m = 0;
        v = 0;
        bias_m = 0;
        bias_v = 0;
        clip = 0;
        learning_rate_scale = 0;
        flipped = 0;
        detail = 0;
        fnode = 0;
    }

    void FnConv::create(
            int mini_batch_size, int nr_X, int nc_X, int nch_X, int nch_Y,
            int num_group, int k_size, int stride, int padding, MATH21_FUNCTION_ACTIVATION_TYPE activation,
            int is_batch_normalize, int adam) {
        auto f = this;
        auto *f_detail = new FnConv_detail();
        f->detail = f_detail;
        f->nch_X = nch_X;
        f->nr_X = nr_X;
        f->nc_X = nc_X;
        f->n_group = num_group;
        f->nch_Y = nch_Y;
        MATH21_ASSERT(f->nch_X % f->n_group == 0 && f->nch_Y % f->n_group == 0,
                      "group_size_X and group_size_Y must be integers.");
        f->batch = mini_batch_size;
        f->stride = stride;
        f->k_size = k_size;
        f->pad = padding;
        // K shape: nch_Y * group_size_X * nr_K * nc_K;
        f->n_W = nch_Y * nch_X / num_group * k_size * k_size;

        int nr_Y = cal_nr_or_nc_Y(f->nr_X, f->pad, f->k_size, f->stride);
        int nc_Y = cal_nr_or_nc_Y(f->nc_X, f->pad, f->k_size, f->stride);
        f->nr_Y = nr_Y;
        f->nc_Y = nc_Y;
        f->y_size = f->nr_Y * f->nc_Y * f->nch_Y;
        // X shape: mbs * num_group * group_size_X * nr_X * nc_X
        f->x_size = f->nc_X * f->nr_X * f->nch_X;

        float scale = sqrt(2. / (k_size * k_size * nch_X / f->n_group));
        int i;
#if defined(MATH21_FLAG_USE_CPU)
        f->W = math21_vector_create_with_default_value_wrapper(f->n_W, 0);
    for (i = 0; i < f->n_W; ++i) f->W[i] = scale * math21_pr_rand_normal();
#else
        f->W_cpu = math21_vector_create_with_default_value_cpu(f->n_W, 0);
        for (i = 0; i < f->n_W; ++i) f->W_cpu[i] = scale * math21_pr_rand_normal();
        f->W = math21_vector_create_from_cpuvector_wrapper(f->n_W, f->W_cpu, 1); // by cl
#endif
        f->dW = math21_vector_create_with_default_value_wrapper(f->n_W, 0);
        f->Y = math21_vector_create_with_default_value_wrapper(f->batch * f->y_size, 0);
        f->dY = math21_vector_create_with_default_value_wrapper(f->batch * f->y_size, 0);

        f_detail->K_wrapper.setWrapper(f->W, f->nch_Y, f->nch_X / f->n_group, f->k_size, f->k_size);
        f_detail->dK_wrapper.setWrapper(f->dW, f->nch_Y, f->nch_X / f->n_group, f->k_size, f->k_size);
        f_detail->y_wrapper.setWrapper(f->Y, f->batch, f->nch_Y, f->nr_Y, f->nc_Y);
        f_detail->dy_wrapper.setWrapper(f->dY, f->batch, f->nch_Y, f->nr_Y, f->nc_Y);

        if (is_batch_normalize) {
            mlfunction_node finput = {0};
            finput.y = f->Y;
            finput.dy = f->dY;
            f->bn = new FnBatchnorm();
            f->bn->create(0, &finput, mini_batch_size, nc_Y, nr_Y, nch_Y, adam);
        } else {
#if !defined(MATH21_FLAG_USE_CPU)
            f->b_cpu = math21_vector_create_with_default_value_cpu(nch_Y, 0);
#endif
            f->b = math21_vector_create_with_default_value_wrapper(nch_Y, 0);
            f->db = math21_vector_create_with_default_value_wrapper(nch_Y, 0);

            f_detail->b_wrapper.setWrapper(f->b, f->nch_Y);
            f_detail->db_wrapper.setWrapper(f->db, f->nch_Y);
        }

        if (adam) {
            f->m = math21_vector_create_with_default_value_wrapper(f->n_W, 0);
            f->v = math21_vector_create_with_default_value_wrapper(f->n_W, 0);
            if (!f->bn) {
                f->bias_m = math21_vector_create_with_default_value_wrapper(nch_Y, 0);
                f->bias_v = math21_vector_create_with_default_value_wrapper(nch_Y, 0);
            }
        }

        // X_prime shape (group_size_X * k_size * k_size ) * (nr_Y * nc_Y)
        f->workspace_size = get_X_prime_size(f->nr_Y, f->nc_Y,
                                             f->k_size, f->nch_X, f->n_group);
        f->activation = activation;

        f->name = math21_string_create_from_string("conv2d");
    }

    void FnConv::resize(int nr_X, int nc_X) {
        FnConv *f = this;
        auto *f_detail = (FnConv_detail *) f->detail;
        f->nc_X = nc_X;
        f->nr_X = nr_X;
        int nr_Y = cal_nr_or_nc_Y(f->nr_X, f->pad, f->k_size, f->stride);
        int nc_Y = cal_nr_or_nc_Y(f->nc_X, f->pad, f->k_size, f->stride);

        f->nr_Y = nr_Y;
        f->nc_Y = nc_Y;

        f->y_size = f->nr_Y * f->nc_Y * f->nch_Y;
        f->x_size = f->nc_X * f->nr_X * f->nch_X;

        f->Y = math21_vector_resize_with_default_value_wrapper(f->Y, f->batch * f->y_size, 0);
        f->dY = math21_vector_resize_with_default_value_wrapper(f->dY, f->batch * f->y_size, 0);
        f_detail->y_wrapper.setWrapper(f->Y, f->batch, f->nch_Y, f->nr_Y, f->nc_Y);
        f_detail->dy_wrapper.setWrapper(f->dY, f->batch, f->nch_Y, f->nr_Y, f->nc_Y);

        if (f->bn) {
            mlfunction_node finput = {0};
            finput.y = f->Y;
            finput.dy = f->dY;
            f->bn->resize(&finput, nc_Y, nr_Y);
        }
        f->workspace_size = get_X_prime_size(f->nr_Y, f->nc_Y,
                                             f->k_size, f->nch_X, f->n_group);
    }

    //  Z = h(Y), Y = W*X + b, or Y = X*W.t + b
//  Y_m = K_m * X_prime + b ...
// float *workspace;// X_prime or dL/dX_prime
// workspace is global space, and has size at least workspace_size.
    void FnConv::forward(const mlfunction_node *finput0,
                         int is_train, PtrR32Wrapper workspace) {
        FnConv *f = this;
        if (is_train) {
            math21_vector_set_wrapper(f->batch * f->y_size, 0, f->dY, 1);
        }

        math21_function_conv2d_forward_wrapper(
                f->Y,
                f->W,
                finput0->y,
                f->nr_X,
                f->nc_X,
                f->nch_X,
                f->nr_Y,
                f->nc_Y,
                f->nch_Y,
                f->y_size,
                f->n_W,
                f->batch,
                f->k_size,
                f->stride,
                f->pad,
                f->n_group,
                workspace, m21_type_NumR32);

        if (f->bn) {
            // ...
            FnBatchnorm *fbn = f->bn;
            fbn->is_train = is_train;
            fbn->forward(0);
        } else {
            // Y += b
            math21_function_conv2d_bias_forward_wrapper(f->Y, f->b, f->batch, f->nch_Y, f->nr_Y * f->nc_Y);
        }

        // Z = h(Y)
        math21_function_activation_vector_wrapper(f->Y, f->y_size * f->batch, f->activation);

        if (f->fnode && f->fnode->id == 106) {
//        f->log("*/K");
//        m21log("...........\n");
//        f->log("*/K");
//        auto data = (const NumR32 *)f->fnode->getDataToCpu(f->fnode, "*/K");
//        math21_tensor_4d_float_log_cpu("K", data, 18, 256, 1, 1);
        }
    }

    // dL/dZ => dL/dK, dL/dX
    void FnConv::backward(mlfunction_node *finput,
                          int is_train, PtrR32Wrapper workspace) {
        FnConv *f = this;
        int nc_dY_m = f->nc_Y * f->nr_Y;

        // dL/dY = dL/dZ *.ele h.d(Y)
        math21_function_activation_gradient_vector_wrapper(f->Y, f->y_size * f->batch, f->activation, f->dY);

        if (f->bn) {
            FnBatchnorm *fbn = f->bn;
            fbn->is_train = is_train;
            fbn->backward(0);
        } else {
            // dL/db += sum(dL/dY(i))
            math21_function_conv2d_bias_backward_wrapper(f->db, f->dY, f->batch, f->nch_Y, nc_dY_m);
        }

        math21_function_conv2d_backward_wrapper(
                f->dY,
                f->W,
                f->dW,
                finput->y,
                finput->dy,
                f->nr_X,
                f->nc_X,
                f->nch_X,
                f->nr_Y,
                f->nc_Y,
                f->nch_Y,
                f->y_size,
                f->n_W,
                f->batch,
                f->k_size,
                f->stride,
                f->pad,
                f->n_group,
                workspace,
                m21_type_NumR32);
    }

    void FnConv::update(OptUpdate *optUpdate) {
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
            math21_optimization_adam_update_wrapper(f->W, f->dW, f->m, f->v, a->beta1, a->beta2,
                                                    a->eps, decay, learning_rate, f->n_W, batch, a->t);
            if (f->bn) {
                f->bn->learning_rate_scale = f->learning_rate_scale;
                f->bn->update(optUpdate);
            } else {
                math21_optimization_adam_update_wrapper(f->b, f->db, f->bias_m, f->bias_v, a->beta1,
                                                        a->beta2, a->eps, decay, learning_rate, f->nch_Y, batch, a->t);
            }
        } else {
            if (f->bn) {
                f->bn->learning_rate_scale = f->learning_rate_scale;
                f->bn->update(optUpdate);
            } else {
                // b = b - alpha * dL/db
                // f->db = -dL/db because of loss function L is -L.
                math21_vector_kx_add_y_wrapper(f->nch_Y, learning_rate / batch, f->db, 1, f->b, 1);
                // dL/db = momentum * dL/db
                math21_vector_kx_wrapper(f->nch_Y, momentum, f->db, 1);
            }

            // dL/dW = dL/dW + decay * W
            math21_vector_kx_add_y_wrapper(f->n_W, -decay * batch, f->W, 1, f->dW, 1);
            // W = W - alpha * dL/dW
            math21_vector_kx_add_y_wrapper(f->n_W, learning_rate / batch, f->dW, 1, f->W, 1);
            // dL/dW = momentum * dL/dW
            math21_vector_kx_wrapper(f->n_W, momentum, f->dW, 1);
        }
        if (f->clip) {
            math21_tool_assert(0);
            math21_vector_clip_wrapper(f->n_W, f->clip, f->W, 1);
        }
    }

    void FnConv::saveState(FILE *file) const {
        auto *f = this;
        math21_vector_serialize_c_wrapper(file, f->Y, f->batch * f->y_size);
        math21_vector_serialize_c_wrapper(file, f->dY, f->batch * f->y_size);
        math21_vector_serialize_c_wrapper(file, f->W, f->n_W);
        math21_vector_serialize_c_wrapper(file, f->dW, f->n_W);
        if (f->bn) {
            f->bn->saveState(file);
        } else {
            math21_vector_serialize_c_wrapper(file, f->b, f->nch_Y);
            math21_vector_serialize_c_wrapper(file, f->db, f->nch_Y);
        }
    }

    // merge f to fb
    void FnConv::mergeTo(FnConv *fb) {
        auto *f = this;
        math21_vector_kx_add_y_cpu(f->n_W, 1, f->W_cpu, 1, fb->W_cpu, 1);
        if (f->bn) {
            f->bn->mergeTo(fb->bn);
        } else {
            math21_vector_kx_add_y_cpu(f->nch_Y, 1, f->b_cpu, 1, fb->b_cpu, 1);
        }
    }

    void FnConv::scale(float s) {
        auto *f = this;
        math21_vector_kx_cpu(f->n_W, s, f->W_cpu, 1);
        if (f->bn) {
            f->bn->scale(s);
        } else {
            math21_vector_kx_cpu(f->nch_Y, s, f->b_cpu, 1);
        }
    }

    void FnConv::pullWrapper(NumB useRolling) {
        auto *f = this;
        math21_vector_pull_wrapper(f->W, f->W_cpu, f->n_W);
        // ye
//    math21_cuda_pull_array(f->dW, f->dW, f->n_W);
//    math21_cuda_pull_array(f->db, f->db, f->n);
        if (f->bn) {
            f->bn->pullWrapper(useRolling);
        } else {
            math21_vector_pull_wrapper(f->b, f->b_cpu, f->nch_Y);
        }
    }

    void FnConv::pushWrapper(NumB useRolling) {
        auto *f = this;
        f->pushByWrapper(f, useRolling);
    }

    // f is pushed by fb
    void FnConv::pushByWrapper(FnConv *fb, NumB useRolling) {
        auto *f = this;
        math21_vector_push_wrapper(f->W, fb->W_cpu, f->n_W);
        // by ye
        // math21_vector_push_wrapper(f->dW, fb->dW, f->n_W);
        // math21_vector_push_wrapper(f->db, fb->db, f->n);
        if (f->bn) {
            f->bn->pushByWrapper(fb->bn, useRolling);
        } else {
            math21_vector_push_wrapper(f->b, fb->b_cpu, f->nch_Y);
        }
    }

    void FnConv::set_mbs(int mini_batch_size) {
        auto *f = this;
        f->batch = mini_batch_size;
        if (f->bn) {
            f->bn->mini_batch_size = mini_batch_size;
        }
    }

    void FnConv::saveTheta(FILE *fp) {
        auto *f = this;
#ifndef MATH21_FLAG_USE_CPU
        f->pullWrapper(1);
#endif

#if defined(MATH21_FLAG_USE_CPU)
        float * weights = f->W;
    float * biases = f->b;
#else
        float *weights = f->W_cpu;
        float *biases = f->b_cpu;
#endif

        int num = f->n_W;
        if (f->bn) {
            f->bn->saveTheta(fp, 0);
        } else {
            fwrite(biases, sizeof(float), f->nch_Y, fp);
        }
        fwrite(weights, sizeof(float), num, fp);
    }

    void FnConv::loadTheta(FILE *fp) {
        auto *f = this;
#if defined(MATH21_FLAG_USE_CPU)
        float * weights = f->W;
    float * biases = f->b;
#else
        float *weights = f->W_cpu;
        float *biases = f->b_cpu;
#endif

        int num = f->nch_X / f->n_group * f->nch_Y * f->k_size * f->k_size;
        if (f->bn) {
            f->bn->loadTheta(fp, 0);
        } else {
            fread(biases, sizeof(float), f->nch_Y, fp);
        }
        fread(weights, sizeof(float), num, fp);
        if (f->flipped) {
            math21_matrix_transpose(weights, f->nch_X * f->k_size * f->k_size, f->nch_Y);
        }
#ifndef MATH21_FLAG_USE_CPU
        f->pushWrapper(1);
#endif
    }

    void FnConv::log(const char *varName) const {
        const auto *f = this;
        auto *f_detail = (const FnConv_detail *) f->detail;
        std::string _varNameNew;
        if (!math21_ml_function_tool_varName_check(f->name, varName, _varNameNew)) {
            return;
        }
        varName = _varNameNew.c_str();

        if (math21_string_is_equal(varName, "summary")) {

//        fprintf(stdout, "conv  %5d x%2d x%2d x%4d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", f->out_c, f->k_size, f->k_size, f->c,
//                f->stride,
//                f->h, f->w, f->c, f->out_h, f->out_w, f->out_c,
//                (2.0 * f->n * f->k_size * f->k_size * f->c / f->groups * f->out_h * f->out_w) / 1000000000.);

            // K shape: nch_Y * group_size_X * nr_K * nc_K;
            // X shape: mbs * num_group * group_size_X * nr_X * nc_X
            if (f->n_group == 1) {
                fprintf(stdout, "%s: (%d, %d, %d, %d) -> (%d, %d, %d, %d), "
                                "K shape = (%d, %d, %d, %d), s = %d, BFLOPs = %5.3f\n",
                        f->name,
                        f->nr_X, f->nc_X, f->nch_X, f->batch,
                        f->nr_Y, f->nc_Y, f->nch_Y, f->batch,
                        f->nch_Y, f->k_size, f->k_size, f->nch_X / f->n_group,
                        f->stride,
                        (2.0 * f->nch_Y * f->k_size * f->k_size * f->nch_X / f->n_group * f->nr_Y * f->nc_Y) /
                        1000000000.);
            } else {
                fprintf(stdout, "%s with groups: (%d, %d, %d, %d, %d) -> (%d, %d, %d, %d, %d), "
                                "K shape = (%d, %d, %d, %d), s = %d, BFLOPs = %5.3f\n",
                        f->name,
                        f->nr_X, f->nc_X, f->nch_X / f->n_group, f->n_group, f->batch,
                        f->nr_Y, f->nc_Y, f->nch_Y / f->n_group, f->n_group, f->batch,
                        f->nch_Y, f->k_size, f->k_size, f->nch_X / f->n_group,
                        f->stride,
                        (2.0 * f->nch_Y * f->k_size * f->k_size * f->nch_X / f->n_group * f->nr_Y * f->nc_Y) /
                        1000000000.);
            }
            return;
        }
        fprintf(stdout, "%s:\n", f->name);
        std::string name = varName;
        m21variable *var;
        if (f_detail->vars.get(varName, var)) {
            var->log(varName);
        } else if (name == "b") {
            if (f->bn) {
                f->bn->log(varName);
            }
        } else if (name == "db") {
            if (f->bn) {
                f->bn->log(varName);
            }
        } else {
            m21log("no variable name ", varName);
        }
    }

    const void *FnConv::getDataToCpu(const char *varName) {
        const auto *f = this;
        auto *f_detail = (FnConv_detail *) f->detail;
        std::string _varNameNew;
        if (!math21_ml_function_tool_varName_check(f->name, varName, _varNameNew)) {
            return 0;
        }
        varName = _varNameNew.c_str();
        fprintf(stdout, "%s:\n", f->name);
        m21variable *var;
        if (f_detail->vars.get(varName, var)) {
            const NumR32 *p = 0;
            var->getDataToCpu(p);
            m21log("p", (void *) p);
//        math21_tensor_4d_float_log_cpu(__FUNCTION__, p, 18, 256, 1, 1);
            return p;
        } else {
            m21log("no variable name ", varName);
            return 0;
        }
    }

    m21rawtensor FnConv::getRawTensorToCpu(const char *varName) {
        const auto *f = this;
        auto *f_detail = (FnConv_detail *) f->detail;
        std::string _varNameNew;
        if (!math21_ml_function_tool_varName_check(f->name, varName, _varNameNew)) {
            m21rawtensor rawtensor = {0};
            return rawtensor;
        }
        varName = _varNameNew.c_str();
        fprintf(stdout, "%s:\n", f->name);
        m21variable *var;
        if (f_detail->vars.get(varName, var)) {
            m21rawtensor p = {0};
            var->getRawTensorToCpu(p);
//        math21_rawtensor_log_cpu(__FUNCTION__ , p);
            return p;
        } else {
            m21log("no variable name ", varName);
            m21rawtensor rawtensor = {0};
            return rawtensor;
        }
    }

}
