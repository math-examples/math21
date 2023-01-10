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
#include "FnLstm.h"

namespace math21 {

    FnLstm::FnLstm() {
        init();
    }

    FnLstm::~FnLstm() {}

    void FnLstm::init() {
        name = 0;
        implementationMode = 0;
        inputs = 0;
        outputs = 0;
        batch = 0;
        fcWi = 0;
        fcWf = 0;
        fcWo = 0;
        fcWg = 0;
        fcWx = 0;
        fcUi = 0;
        fcUf = 0;
        fcUo = 0;
        fcUg = 0;
        fcUh = 0;
        fcW = 0;
        xh_interleaved = 0;
        dxh_interleaved = 0;
        delta = 0;
        output = 0;
        last_output = 0;
        h_0 = 0;
        c_0 = 0;
        temp = 0;
        dc_t = 0;
        i = 0;
        f = 0;
        o = 0;
        g = 0;
        d_i = 0;
        d_f = 0;
        d_o = 0;
        d_g = 0;
        ifog_interleaved = 0;
        difog_interleaved = 0;
        ifog_noninterleaved = 0;
        difog_noninterleaved = 0;
        c = 0;
        dc_tm1_at_t = 0;
        cell = 0;
        h = 0;
        steps = 0;
        i_time_step = 0;
        is_dropout_x = 0;
        is_dropout_h = 0;
        dropout_x = 0;
        dropout_h = 0;
        is_return_sequences = 0;
    }


/*
 * many-to-many fashion using a time distributed node with fnode->mini_batch_size = f->steps * f->batch.
 *
 * have done:
 * change order of f and i in bp
 *
 * todo: unit_forget_bias, add implementation mode 2, REMOVE c_tm1 malloc, REMOVE prev_STATE malloc
 * add stateful as in keras.
 *
 * remove c malloc, remove h malloc
 * RENAME prev_STATE TO H_TM1, RENAME PREV_CELL TO C_HM1
 * math21_vector_clip_wrapper iN FnFullyConnected::backward
 * implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
 * */
    void FnLstm::create(
            int batch, int input_size, int output_size, int n_time_step, int is_use_bias,
            int is_batch_normalize, int is_unit_forget_bias, float dropout_rate_x, float dropout_rate_h, int is_adam,
            int is_return_sequences, int implementationMode) {
        auto f = this;
        int rnn_batch_size = batch / n_time_step;
        f->batch = rnn_batch_size;
        f->steps = n_time_step;
        f->inputs = input_size;
        f->outputs = output_size;
        f->is_return_sequences = is_return_sequences;

        f->implementationMode = implementationMode;
//    f->implementationMode = 1;
//    f->implementationMode = 2;
//    f->implementationMode = 3;

        if (dropout_rate_x > 0) {
            f->is_dropout_x = 1;
            mlfunction_node finput_x = {0};
            finput_x.mini_batch_size = n_time_step * rnn_batch_size;
            finput_x.y_size = input_size;

            f->dropout_x = new FnDropout(&finput_x, dropout_rate_x, n_time_step,
                                         "dropout_x");
        }
        if (dropout_rate_h > 0) {
            f->is_dropout_h = 1;
            mlfunction_node finput_h = {0};
            finput_h.mini_batch_size = n_time_step * rnn_batch_size;
            finput_h.y_size = output_size;
            f->dropout_h = new FnDropout(&finput_h, dropout_rate_h, n_time_step,
                                         "dropout_h");
        }

        if (f->implementationMode == 1) {
            f->fcWi = new FnFullyConnected(rnn_batch_size, n_time_step, input_size,
                                           output_size,
                                           MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR,
                                           is_use_bias,
                                           is_batch_normalize, is_adam, "fcWi");

            f->fcWf = new FnFullyConnected(rnn_batch_size, n_time_step, input_size,
                                           output_size,
                                           MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR,
                                           is_use_bias,
                                           is_batch_normalize, is_adam, "fcWf");
            if (f->fcWf->is_use_bias && is_unit_forget_bias) {
                math21_vector_set_wrapper(f->fcWf->outputs, 1, f->fcWf->biases, 1);
            }

            f->fcWo = new FnFullyConnected(rnn_batch_size, n_time_step, input_size,
                                           output_size,
                                           MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR,
                                           is_use_bias,
                                           is_batch_normalize, is_adam, "fcWo");

            f->fcWg = new FnFullyConnected(rnn_batch_size, n_time_step, input_size,
                                           output_size,
                                           MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR,
                                           is_use_bias,
                                           is_batch_normalize, is_adam, "fcWg");

            f->fcUi = new FnFullyConnected(rnn_batch_size, n_time_step,
                                           output_size,
                                           output_size,
                                           MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR,
                                           0,
                                           is_batch_normalize, is_adam, "fcUi");

            f->fcUf = new FnFullyConnected(rnn_batch_size, n_time_step,
                                           output_size,
                                           output_size,
                                           MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR,
                                           0,
                                           is_batch_normalize, is_adam, "fcUf");

            f->fcUo = new FnFullyConnected(rnn_batch_size, n_time_step,
                                           output_size,
                                           output_size,
                                           MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR,
                                           0,
                                           is_batch_normalize, is_adam, "fcUo");

            f->fcUg = new FnFullyConnected(rnn_batch_size, n_time_step,
                                           output_size,
                                           output_size,
                                           MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR,
                                           0,
                                           is_batch_normalize, is_adam, "fcUg");

            f->i = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);
            f->f = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);
            f->o = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);
            f->g = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);
        } else {
            if (f->implementationMode == 2) {
                f->fcWx = new FnFullyConnected(rnn_batch_size, n_time_step,
                                               input_size,
                                               output_size * 4,
                                               MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR,
                                               is_use_bias,
                                               is_batch_normalize, is_adam, "fcWx");
                if (f->fcWx->is_use_bias && is_unit_forget_bias) {
                    math21_vector_set_wrapper(f->fcWx->outputs / 4, 1, f->fcWx->biases + f->fcWx->outputs / 4, 1);
                }

                f->fcUh = new FnFullyConnected(rnn_batch_size, n_time_step,
                                               output_size,
                                               output_size * 4,
                                               MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR,
                                               0,
                                               is_batch_normalize, is_adam, "fcUh");
            } else {
                MATH21_ASSERT(f->implementationMode == 3);
                f->fcW = new FnFullyConnected(rnn_batch_size, n_time_step,
                                              input_size + output_size,
                                              output_size * 4,
                                              MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR,
                                              is_use_bias,
                                              is_batch_normalize, is_adam, "fcW");
                if (f->fcW->is_use_bias && is_unit_forget_bias) {
                    math21_vector_set_wrapper(f->fcW->outputs / 4, 1, f->fcW->biases + f->fcW->outputs / 4, 1);
                }
                f->xh_interleaved = math21_vector_create_with_default_value_wrapper(
                        rnn_batch_size * (input_size + output_size), 0);
                f->dxh_interleaved = math21_vector_create_with_default_value_wrapper(
                        rnn_batch_size * (input_size + output_size), 0);
            }

            // shape: (mbs, 4, y_size)
            f->ifog_interleaved = math21_vector_create_with_default_value_wrapper(4 * rnn_batch_size * output_size, 0);
            // shape: (4, mbs, y_size)
            f->ifog_noninterleaved = math21_vector_create_with_default_value_wrapper(4 * rnn_batch_size * output_size,
                                                                                     0);
            f->i = f->ifog_noninterleaved;
            f->f = f->i + rnn_batch_size * output_size;
            f->o = f->f + rnn_batch_size * output_size;
            f->g = f->o + rnn_batch_size * output_size;

            f->difog_interleaved = f->ifog_noninterleaved;
            f->difog_noninterleaved = f->ifog_interleaved;
            f->d_i = f->difog_noninterleaved;
            f->d_f = f->d_i + rnn_batch_size * output_size;
            f->d_o = f->d_f + rnn_batch_size * output_size;
            f->d_g = f->d_o + rnn_batch_size * output_size;
        }

        f->output = math21_vector_create_with_default_value_wrapper(n_time_step * rnn_batch_size * output_size, 0);
        f->last_output = f->output + (n_time_step - 1) * rnn_batch_size * output_size;
        f->delta = math21_vector_create_with_default_value_wrapper(n_time_step * rnn_batch_size * output_size, 0);
        f->cell = math21_vector_create_with_default_value_wrapper(n_time_step * rnn_batch_size * output_size, 0);

        f->h_0 = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);
        f->c_0 = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);

        f->c = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);
        // todo: maybe consider h to be pointer of output, but must solve mlfnode_type_rnn first.
        f->h = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);
        f->temp = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);
        f->dc_t = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);
        f->dc_tm1_at_t = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);

        f->name = math21_string_create_from_string("lstm");
    }

    void FnLstm::log(const char *varName) const {
        auto f = this;
        std::string _varNameNew;
        if (!math21_ml_function_tool_varName_check(f->name, varName, _varNameNew)) {
            return;
        }
        varName = _varNameNew.c_str();

        if (math21_string_is_equal(varName, "summary")) {
            varName = "*/summary";
            fprintf(stdout, "lstm: (%d, %d, %d) -> (%d, %d, %d)\n", f->inputs, f->steps, f->batch, f->outputs,
                    f->is_return_sequences ? f->steps : 1,
                    f->batch);
            if (f->is_dropout_x) {
                fprintf(stdout, "\t\t");
                f->dropout_x->log(varName);
            }
            if (f->is_dropout_h) {
                fprintf(stdout, "\t\t");
                f->dropout_h->log(varName);
            }
            if (f->implementationMode == 1) {
                fprintf(stdout, "\t\t");
                f->fcWi->log(varName);
                fprintf(stdout, "\t\t");
                f->fcWf->log(varName);
                fprintf(stdout, "\t\t");
                f->fcWo->log(varName);
                fprintf(stdout, "\t\t");
                f->fcWg->log(varName);
                fprintf(stdout, "\t\t");
                f->fcUi->log(varName);
                fprintf(stdout, "\t\t");
                f->fcUf->log(varName);
                fprintf(stdout, "\t\t");
                f->fcUo->log(varName);
                fprintf(stdout, "\t\t");
                f->fcUg->log(varName);
            } else if (f->implementationMode == 2) {
                fprintf(stdout, "\t\t");
                f->fcWx->log(varName);
                fprintf(stdout, "\t\t");
                f->fcUh->log(varName);
            } else {
                fprintf(stdout, "\t\t");
                f->fcW->log(varName);
            }
            return;
        }
        if (f->is_dropout_x) {
            f->dropout_x->log(varName);
        }
        if (f->is_dropout_h) {
            f->dropout_h->log(varName);
        }
        if (f->implementationMode == 1) {
            f->fcWi->log(varName);
            f->fcWf->log(varName);
            f->fcWo->log(varName);
            f->fcWg->log(varName);
            f->fcUi->log(varName);
            f->fcUf->log(varName);
            f->fcUo->log(varName);
            f->fcUg->log(varName);
        } else if (f->implementationMode == 2) {
            f->fcWx->log(varName);
            f->fcUh->log(varName);
        } else {
            f->fcW->log(varName);
        }
    }

    void _math21_ml_function_lstm_ifog_transpose(FnLstm *f, NumB isForward) {
        if (isForward) {
//        math21_tensor_3d_float_log_wrapper("f->ifog_interleaved", f->ifog_interleaved, f->batch, 4, f->outputs);
            math21_vector_transpose_d1234_to_d1324_wrapper(f->ifog_interleaved, f->ifog_noninterleaved,
                                                           1, f->batch, 4, f->outputs);
//        math21_tensor_3d_float_log_wrapper("f->ifog_noninterleaved", f->ifog_noninterleaved, 4, f->batch, f->outputs);
        } else {
            math21_vector_transpose_d1234_to_d1324_wrapper(f->difog_noninterleaved, f->difog_interleaved,
                                                           1, 4, f->batch, f->outputs);
        }
    }

// not rigorous tensor transpose
    void _math21_ml_function_lstm_xh_set(FnLstm *f,
                                         PtrR32Wrapper x, PtrR32Wrapper h,
                                         PtrR32Wrapper xh, NumB isForward) {
        if (isForward) {
            if (!math21_vector_isEmpty_wrapper(x)) {
                math21_vector_assign_3d_d2_wrapper(x, xh, f->batch, f->inputs + f->outputs, 1, f->inputs, 0, 0);
            }
            if (!math21_vector_isEmpty_wrapper(h)) {
                math21_vector_assign_3d_d2_wrapper(h, xh, f->batch, f->inputs + f->outputs, 1, f->outputs, f->inputs,
                                                   0);
            }
        } else {
            if (!math21_vector_isEmpty_wrapper(x)) {
                math21_vector_assign_3d_d2_wrapper(xh, x, f->batch, f->inputs + f->outputs, 1, f->inputs, 0, 1);
            }
            if (!math21_vector_isEmpty_wrapper(h)) {
                math21_vector_assign_3d_d2_wrapper(xh, h, f->batch, f->inputs + f->outputs, 1, f->outputs, f->inputs,
                                                   1);
            }
        }
    }

    void _math21_ml_function_lstm_x_add_h_and_activate(FnLstm *f, NumB isForward) {
        if (f->implementationMode == 1) {
            // y_i = y_h + y_x for input gate
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->fcWi->output, 1, f->i, 1);
            math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, f->fcUi->output, 1, f->i, 1);
            // y_f = y_h + y_x for forget gate
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->fcWf->output, 1, f->f, 1);
            math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, f->fcUf->output, 1, f->f, 1);
            // y_o = y_h + y_x for output gate
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->fcWo->output, 1, f->o, 1);
            math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, f->fcUo->output, 1, f->o, 1);
            // y_g = y_h + y_x for cell
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->fcWg->output, 1, f->g, 1);
            math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, f->fcUg->output, 1, f->g, 1);
        } else if (f->implementationMode == 2) {
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch * 4, f->fcWx->output, 1, f->ifog_interleaved,
                                                     1);
            math21_vector_kx_add_y_wrapper(f->outputs * f->batch * 4, 1, f->fcUh->output, 1, f->ifog_interleaved, 1);
            _math21_ml_function_lstm_ifog_transpose(f, 1);
        } else {
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch * 4, f->fcW->output, 1, f->ifog_interleaved,
                                                     1);
            if (math21_ml_function_tool_is_debug() && isForward) {
//            math21_tensor_2d_float_log_wrapper("ifog_interleaved", f->ifog_interleaved, f->batch, f->outputs * 4);
            }
            _math21_ml_function_lstm_ifog_transpose(f, 1);
        }
        // i = y_i <- sigmoid(y_i)
        math21_function_activation_vector_wrapper(f->i, f->outputs * f->batch,
                                                  MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC);
        // f = y_f <- sigmoid(y_f)
        math21_function_activation_vector_wrapper(f->f, f->outputs * f->batch,
                                                  MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC);
        // o = y_o <- sigmoid(y_o)
        math21_function_activation_vector_wrapper(f->o, f->outputs * f->batch,
                                                  MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC);
        // g = y_g <- tanh(y_g)
        math21_function_activation_vector_wrapper(f->g, f->outputs * f->batch, MATH21_FUNCTION_ACTIVATION_TYPE_TANH);
    }

/*
h(t) = lstm(x(t), h(t-1)), t = 1, ..., T.

(i, f, o, g)' = (sigm, sigm, sigm, tanh)' (W *(x(t), h(t-1))')
// CEC
c(t) = f * c(t-1) + i * g
h(t) = o * tanh(c(t))
with W :=
            Wi, Ui,
            Wf, Uf,
            Wo, Uo,
            Wg, Ug;

dropout:
(i, f, o, g)' = (sigm, sigm, sigm, tanh)' (W *(x(t)*zx, h(t-1)*zh)')
with zx, zh random masks repeated at all time steps. (Note: zx, zh is independent of t)

// the following is not used
y(t) = Why*h(t) + by
y = W_hy * h(t) + b_y
h_pre, h(1), ..., h(t), ..., h(T)

# References
        - [Long short-term memory](
          http://www.bioinf.jku.at/publications/older/2604.pdf)
        - [Learning to forget: Continual prediction with LSTM](
          http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [A Theoretically Grounded Application of Dropout in
          Recurrent Neural Networks](https://arxiv.org/abs/1512.05287)
 * */
    void FnLstm::forward(mlfunction_node *finput, int is_train) {
        auto f = this;
        int i;

        if (is_train) {
            math21_vector_set_wrapper(f->steps * f->batch * f->outputs, 0, f->delta, 1);
            math21_vector_set_wrapper(f->batch * f->outputs, 0, f->dc_tm1_at_t, 1);
            // h_0 <- h, c_0 <- c, backup for backward
            math21_vector_assign_from_vector_wrapper(f->batch * f->outputs, f->h, 1, f->h_0, 1);
            math21_vector_assign_from_vector_wrapper(f->batch * f->outputs, f->c, 1, f->c_0, 1);
        }

        mlfunction_node finput_fc_x0 = {0};
        mlfunction_node *finput_fc_x = &finput_fc_x0;
        mlfunction_node finput_fc_h0 = {0};
        mlfunction_node *finput_fc_h = &finput_fc_h0;
        mlfunction_node finput_fc_xh0 = {0};
        mlfunction_node *finput_fc_xh = &finput_fc_xh0;
        for (i = 0; i < f->steps; ++i) {
            if (f->is_dropout_x) {
                finput_fc_x->y = finput->y + i * f->inputs * f->batch;
                f->dropout_x->forward(finput_fc_x, is_train);
                finput_fc_x->y = f->dropout_x->y;
            } else {
                finput_fc_x->y = finput->y + i * f->inputs * f->batch;
            }

            if (f->is_dropout_h) {
                finput_fc_h->y = f->h;
                f->dropout_h->forward(finput_fc_h, is_train);
                finput_fc_h->y = f->dropout_h->y;
            } else {
                finput_fc_h->y = f->h;
            }

            if (f->implementationMode == 1) {
                f->fcWi->forward(finput_fc_x, is_train);
                f->fcWf->forward(finput_fc_x, is_train);
                f->fcWo->forward(finput_fc_x, is_train);
                f->fcWg->forward(finput_fc_x, is_train);
                f->fcUi->forward(finput_fc_h, is_train);
                f->fcUf->forward(finput_fc_h, is_train);
                f->fcUo->forward(finput_fc_h, is_train);
                f->fcUg->forward(finput_fc_h, is_train);
            } else if (f->implementationMode == 2) {
                f->fcWx->forward(finput_fc_x, is_train);
                f->fcUh->forward(finput_fc_h, is_train);
            } else {
                // finput_fc_x, finput_fc_h -> finput_fc_xh
                _math21_ml_function_lstm_xh_set(f, finput_fc_x->y, finput_fc_h->y,
                                                f->xh_interleaved, 1);
                finput_fc_xh->y = f->xh_interleaved;

                f->fcW->forward(finput_fc_xh, is_train);
            }
            _math21_ml_function_lstm_x_add_h_and_activate(f, 1);

            // CEC: c(t) = f * c(t-1) + i * g
            // i * g
            // c <- f * c = f * c(t-1)
            // c <- c + i * g
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->i, 1, f->temp, 1);
            math21_vector_xy_wrapper(f->outputs * f->batch, f->g, 1, f->temp, 1);
            math21_vector_xy_wrapper(f->outputs * f->batch, f->f, 1, f->c, 1);
            math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, f->temp, 1, f->c, 1);
            if (math21_ml_function_tool_is_debug()) {
//            math21_tensor_2d_float_log_wrapper("c(t)", f->c, f->batch, f->outputs);
            }

            // h(t) = o * tanh(c(t))
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->c, 1, f->h, 1);
            math21_function_activation_vector_wrapper(f->h, f->outputs * f->batch,
                                                      MATH21_FUNCTION_ACTIVATION_TYPE_TANH);
            math21_vector_xy_wrapper(f->outputs * f->batch, f->o, 1, f->h, 1);
            if (math21_ml_function_tool_is_debug()) {
//            math21_tensor_2d_float_log_wrapper("h(t)", f->h, f->batch, f->outputs);
            }


            // cell(t) <- c = c(t)
            // y = h = h(t)
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->c, 1, f->cell, 1);
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->h, 1, f->output, 1);

            f->increaseByTime(1);
        }
        f->reset();
    }

    // h(t) = lstm(x(t), h(t-1)), t = 1, ..., T.
    // dy => dW, dx
    // BPTT
    // truncated BPTT
    // here not using RTRL
    void FnLstm::backward(mlfunction_node *finput0, int is_train) {
        auto f = this;
        int i;

        f->increaseByTime(f->steps - 1);

        mlfunction_node finput_fc_x0 = {0};
        mlfunction_node *finput_fc_x = &finput_fc_x0;
        mlfunction_node finput_fc_h0 = {0};
        mlfunction_node *finput_fc_h = &finput_fc_h0;
        mlfunction_node finput_fc_xh0 = {0};
        mlfunction_node *finput_fc_xh = &finput_fc_xh0;
        mlfunction_node finput1 = *finput0;
        mlfunction_node *finput = &finput1;

        finput->y += f->inputs * f->batch * (f->steps - 1);
        if (!math21_vector_isEmpty_wrapper(finput->dy)) finput->dy += f->inputs * f->batch * (f->steps - 1);

        for (i = f->steps - 1; i >= 0; --i) {
            // c_tm1 = cell(t-1)
            PtrR32Wrapper c_tm1 = (i == 0) ? f->c_0 : f->cell - f->outputs * f->batch;
            // c_t = c = cell(t)
            PtrR32Wrapper c_t = f->cell;

            // h_tm1 = h(t-1) = y(t-1)
            PtrR32Wrapper h_tm1 = (i == 0) ? f->h_0 : f->output - f->outputs * f->batch;
            // dL/dh(t-1)
            // dh_tm1 = dh(t-1) = dy(t-1)
            // truncated at t = 1, i.e., i=0
            PtrR32Wrapper dh_tm1 = (i == 0) ? math21_vector_getEmpty_R32_wrapper() : f->delta - f->outputs * f->batch;
            // get ifog at t
            _math21_ml_function_lstm_x_add_h_and_activate(f, 0);

            // CEC
            // c(t) = f * c(t-1) + i * g
            // h(t) = y(t) = o * tanh(c(t))

            //// tanh(c(t))
            // temp = tanh(c(t))
            auto &tanh_c_t = f->temp;
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, c_t, 1, tanh_c_t, 1);
            math21_function_activation_vector_wrapper(tanh_c_t, f->outputs * f->batch,
                                                      MATH21_FUNCTION_ACTIVATION_TYPE_TANH);

            //// dc(t)
            // h(t) = y(t) = o * tanh(c(t))
            // o * dy(t)
            // dc(t) = d(tanh) * o * dy(t)
            // dc(t) <- dc_tm1_at_t + dc(t), here dc_tm1_at_t = dc(t) at t + 1
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->delta, 1, f->dc_t, 1);
            math21_vector_xy_wrapper(f->outputs * f->batch, f->o, 1, f->dc_t, 1);
            math21_function_activation_gradient_vector_wrapper(tanh_c_t, f->outputs * f->batch,
                                                               MATH21_FUNCTION_ACTIVATION_TYPE_TANH, f->dc_t);
            math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, f->dc_tm1_at_t, 1, f->dc_t, 1);

            //// c
            // c(t) = f * c(t-1) + i * g
            // dc(t)
            // dc(t-1) at t = f * dc(t)
            // dc_tm1_at_t = dc(t-1) at t
            if (i > 0) {
                math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->dc_t, 1, f->dc_tm1_at_t, 1);
                math21_vector_xy_wrapper(f->outputs * f->batch, f->f, 1, f->dc_tm1_at_t, 1);
            }

            if (f->implementationMode == 1) {
                //// o
                // h(t) = y(t) = o * tanh(c(t))
                // temp = tanh(c(t))
                // do = tanh(c(t)) * dy
                // dneto = d(sigm) * do
                auto &o_temp = f->fcWo->delta;
                math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, tanh_c_t, 1, o_temp, 1);
                math21_vector_xy_wrapper(f->outputs * f->batch, f->delta, 1, o_temp, 1);
                math21_function_activation_gradient_vector_wrapper(f->o, f->outputs * f->batch,
                                                                   MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC, o_temp);
                // dneto
//        math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, o_temp, 1, f->fcWo->delta, 1);
                math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, o_temp, 1, f->fcUo->delta, 1);

                //// g
                // c(t) = f * c(t-1) + i * g
                // dc(t)
                // dg = i * dc(t)
                // dnetg = d(tanh) * dg
                auto &g_temp = f->fcWg->delta;
                math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->dc_t, 1, g_temp, 1);
                math21_vector_xy_wrapper(f->outputs * f->batch, f->i, 1, g_temp, 1);
                math21_function_activation_gradient_vector_wrapper(f->g, f->outputs * f->batch,
                                                                   MATH21_FUNCTION_ACTIVATION_TYPE_TANH, g_temp);
                // dnetg
//        math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, g_temp, 1, f->fcWg->delta, 1);
                math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, g_temp, 1, f->fcUg->delta, 1);

                //// f
                // c(t) = f * c(t-1) + i * g
                // dc(t)
                // df = c(t-1) * dc(t)
                // dnetf = d(sigm) * df
                auto &f_temp = f->fcWf->delta;
                math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->dc_t, 1, f_temp, 1);
                math21_vector_xy_wrapper(f->outputs * f->batch, c_tm1, 1, f_temp, 1);
                math21_function_activation_gradient_vector_wrapper(f->f, f->outputs * f->batch,
                                                                   MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC, f_temp);
                // dnetf
//        math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f_temp, 1, f->fcWf->delta, 1);
                math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f_temp, 1, f->fcUf->delta, 1);

                //// i
                // c(t) = f * c(t-1) + i * g
                // dc(t)
                // di = g * dc(t)
                // dneti = d(sigm) * di
                auto &i_temp = f->fcWi->delta;
                math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->dc_t, 1, i_temp, 1);
                math21_vector_xy_wrapper(f->outputs * f->batch, f->g, 1, i_temp, 1);
                math21_function_activation_gradient_vector_wrapper(f->i, f->outputs * f->batch,
                                                                   MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC, i_temp);
                // dneti
//        math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, i_temp, 1, f->fcWi->delta, 1);
                math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, i_temp, 1, f->fcUi->delta, 1);
            } else {
//// o
                // h(t) = y(t) = o * tanh(c(t))
                // temp = tanh(c(t))
                // do = tanh(c(t)) * dy
                // dneto = d(sigm) * do
                auto &o_temp = f->d_o;
                math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, tanh_c_t, 1, o_temp, 1);
                math21_vector_xy_wrapper(f->outputs * f->batch, f->delta, 1, o_temp, 1);
                math21_function_activation_gradient_vector_wrapper(f->o, f->outputs * f->batch,
                                                                   MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC, o_temp);

                //// g
                // c(t) = f * c(t-1) + i * g
                // dc(t)
                // dg = i * dc(t)
                // dnetg = d(tanh) * dg
                auto &g_temp = f->d_g;
                math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->dc_t, 1, g_temp, 1);
                math21_vector_xy_wrapper(f->outputs * f->batch, f->i, 1, g_temp, 1);
                math21_function_activation_gradient_vector_wrapper(f->g, f->outputs * f->batch,
                                                                   MATH21_FUNCTION_ACTIVATION_TYPE_TANH, g_temp);

                //// f
                // c(t) = f * c(t-1) + i * g
                // dc(t)
                // df = c(t-1) * dc(t)
                // dnetf = d(sigm) * df
                auto &f_temp = f->d_f;
                math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->dc_t, 1, f_temp, 1);
                math21_vector_xy_wrapper(f->outputs * f->batch, c_tm1, 1, f_temp, 1);
                math21_function_activation_gradient_vector_wrapper(f->f, f->outputs * f->batch,
                                                                   MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC, f_temp);

                //// i
                // c(t) = f * c(t-1) + i * g
                // dc(t)
                // di = g * dc(t)
                // dneti = d(sigm) * di
                auto &i_temp = f->d_i;
                math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->dc_t, 1, i_temp, 1);
                math21_vector_xy_wrapper(f->outputs * f->batch, f->g, 1, i_temp, 1);
                math21_function_activation_gradient_vector_wrapper(f->i, f->outputs * f->batch,
                                                                   MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC, i_temp);


                _math21_ml_function_lstm_ifog_transpose(f, 0);

                if (f->implementationMode == 2) {
                    // dnet
                    math21_vector_assign_from_vector_wrapper(4 * f->outputs * f->batch, f->difog_interleaved, 1,
                                                             f->fcWx->delta,
                                                             1);
                    math21_vector_assign_from_vector_wrapper(4 * f->outputs * f->batch, f->difog_interleaved, 1,
                                                             f->fcUh->delta,
                                                             1);
                } else {
                    // dnet
                    math21_vector_assign_from_vector_wrapper(4 * f->outputs * f->batch, f->difog_interleaved, 1,
                                                             f->fcW->delta,
                                                             1);
                }
            }

            ////
            if (f->is_dropout_x) {
                finput_fc_x->y = f->dropout_x->y;
                finput_fc_x->dy = f->dropout_x->dy;
            } else {
                finput_fc_x->y = finput->y;
                finput_fc_x->dy = finput->dy;
            }

            if (f->is_dropout_h) {
                finput_fc_h->y = f->dropout_h->y;
                finput_fc_h->dy = f->dropout_h->dy;
            } else {
                finput_fc_h->y = h_tm1;
                finput_fc_h->dy = dh_tm1;
            }

            if (f->implementationMode == 1) {
                // dneto => dwo, duo, dx, dh
                f->fcWo->backward(finput_fc_x, is_train);
                f->fcUo->backward(finput_fc_h, is_train);
                // dnetg => dwg, dug, dx, dh
                f->fcWg->backward(finput_fc_x, is_train);
                f->fcUg->backward(finput_fc_h, is_train);
                // dnetf => dwf, duf, dx, dh
                f->fcWf->backward(finput_fc_x, is_train);
                f->fcUf->backward(finput_fc_h, is_train);
                // dneti => dwi, dui, dx, dh
                f->fcWi->backward(finput_fc_x, is_train);
                f->fcUi->backward(finput_fc_h, is_train);
            } else if (f->implementationMode == 2) {
                // dnet => dw, du, dx, dh
                f->fcWx->backward(finput_fc_x, is_train);
                f->fcUh->backward(finput_fc_h, is_train);
            } else {
                _math21_ml_function_lstm_xh_set(f, finput_fc_x->y, finput_fc_h->y,
                                                f->xh_interleaved, 1);
                _math21_ml_function_lstm_xh_set(f, finput_fc_x->dy, finput_fc_h->dy,
                                                f->dxh_interleaved, 1);
                finput_fc_xh->y = f->xh_interleaved;
                finput_fc_xh->dy = f->dxh_interleaved;

                // dnet => dw, dx, dh
                f->fcW->backward(finput_fc_xh, is_train);

                _math21_ml_function_lstm_xh_set(f, finput_fc_x->dy, finput_fc_h->dy,
                                                f->dxh_interleaved, 0);
            }

            if (f->is_dropout_x) {
                finput_fc_x->y = finput->y;
                finput_fc_x->dy = finput->dy;
                f->dropout_x->backward(finput_fc_x);
            }
            if (f->is_dropout_h) {
                finput_fc_h->y = h_tm1;
                finput_fc_h->dy = dh_tm1;
                f->dropout_h->backward(finput_fc_h);
            }

            finput->y -= f->inputs * f->batch;
            if (!math21_vector_isEmpty_wrapper(finput->dy)) finput->dy -= f->inputs * f->batch;
            f->increaseByTime(-1);
        }
        f->reset();

//    math21_ml_function_debug_function_save_state(f, 0);
    }

    void FnLstm::update(OptUpdate *optUpdate) {
        auto f = this;
        if (f->implementationMode == 1) {
            f->fcWi->update(optUpdate);
            f->fcWf->update(optUpdate);
            f->fcWo->update(optUpdate);
            f->fcWg->update(optUpdate);
            f->fcUi->update(optUpdate);
            f->fcUf->update(optUpdate);
            f->fcUo->update(optUpdate);
            f->fcUg->update(optUpdate);
        } else if (f->implementationMode == 2) {
            f->fcWx->update(optUpdate);
            f->fcUh->update(optUpdate);
        } else {
            f->fcW->update(optUpdate);
        }
    }

    void FnLstm::saveState(FILE *file) const {
        auto f = this;
        math21_vector_serialize_c_wrapper(file, f->output, f->steps * f->batch * f->outputs);
        math21_vector_serialize_c_wrapper(file, f->delta, f->steps * f->batch * f->outputs);
        math21_vector_serialize_c_wrapper(file, f->cell, f->steps * f->batch * f->outputs);
        if (f->implementationMode == 1) {
            f->fcWi->saveState(file);
            f->fcWf->saveState(file);
            f->fcWo->saveState(file);
            f->fcWg->saveState(file);
            f->fcUi->saveState(file);
            f->fcUf->saveState(file);
            f->fcUo->saveState(file);
            f->fcUg->saveState(file);
        } else if (f->implementationMode == 2) {
            f->fcWx->saveState(file);
            f->fcUh->saveState(file);
        } else {
            f->fcW->saveState(file);
        }
    }

    void FnLstm::increaseByTime(int time_steps) {
        auto f = this;
        f->i_time_step += time_steps;
        int num = f->outputs * f->batch * time_steps;
        f->output += num;
        f->delta += num;
        f->cell += num;
        if (f->is_dropout_x) {
            f->dropout_x->increaseByTime(time_steps);
        }
        if (f->is_dropout_h) {
            f->dropout_h->increaseByTime(time_steps);
        }
        if (f->implementationMode == 1) {
            f->fcWi->increaseByTime(time_steps);
            f->fcWf->increaseByTime(time_steps);
            f->fcWo->increaseByTime(time_steps);
            f->fcWg->increaseByTime(time_steps);
            f->fcUi->increaseByTime(time_steps);
            f->fcUf->increaseByTime(time_steps);
            f->fcUo->increaseByTime(time_steps);
            f->fcUg->increaseByTime(time_steps);
        } else if (f->implementationMode == 2) {
            f->fcWx->increaseByTime(time_steps);
            f->fcUh->increaseByTime(time_steps);
        } else {
            f->fcW->increaseByTime(time_steps);
        }
    }

    void FnLstm::reset() {
        auto f = this;
        int num = f->outputs * f->batch * f->i_time_step;
        f->output -= num;
        f->delta -= num;
        f->cell -= num;
        f->i_time_step = 0;

        if (f->is_dropout_x) {
            f->dropout_x->reset();
        }
        if (f->is_dropout_h) {
            f->dropout_h->reset();
        }
        if (f->implementationMode == 1) {
            f->fcWi->reset();
            f->fcWf->reset();
            f->fcWo->reset();
            f->fcWg->reset();
            f->fcUi->reset();
            f->fcUf->reset();
            f->fcUo->reset();
            f->fcUg->reset();
        } else if (f->implementationMode == 2) {
            f->fcWx->reset();
            f->fcUh->reset();
        } else {
            f->fcW->reset();
        }
    }

    void FnLstm::setMbs(int mini_batch_size) {
        auto f = this;
        f->batch = mini_batch_size;
        if (f->implementationMode == 1) {
            f->fcWi->setMbs(mini_batch_size);
            f->fcWf->setMbs(mini_batch_size);
            f->fcWo->setMbs(mini_batch_size);
            f->fcWg->setMbs(mini_batch_size);
            f->fcUi->setMbs(mini_batch_size);
            f->fcUf->setMbs(mini_batch_size);
            f->fcUo->setMbs(mini_batch_size);
            f->fcUg->setMbs(mini_batch_size);
        } else if (f->implementationMode == 2) {
            f->fcWx->setMbs(mini_batch_size);
            f->fcUh->setMbs(mini_batch_size);
        } else {
            f->fcW->setMbs(mini_batch_size);
        }
    }

    void FnLstm::saveThetaOrderBwsmv(FILE *fp) {
        auto f = this;
        math21_tool_assert(f->implementationMode == 1);
        f->fcUi->saveThetaOrderBwsmv(fp);
        f->fcUf->saveThetaOrderBwsmv(fp);
        f->fcUo->saveThetaOrderBwsmv(fp);
        f->fcUg->saveThetaOrderBwsmv(fp);
        f->fcWi->saveThetaOrderBwsmv(fp);
        f->fcWf->saveThetaOrderBwsmv(fp);
        f->fcWo->saveThetaOrderBwsmv(fp);
        f->fcWg->saveThetaOrderBwsmv(fp);
    }

    void FnLstm::loadThetaOrderBwsmvFlipped(FILE *fp, int flipped) {
        auto f = this;
        math21_tool_assert(f->implementationMode == 1);
        f->fcUi->loadThetaOrderBwsmvFlipped(fp, flipped);
        f->fcUf->loadThetaOrderBwsmvFlipped(fp, flipped);
        f->fcUo->loadThetaOrderBwsmvFlipped(fp, flipped);
        f->fcUg->loadThetaOrderBwsmvFlipped(fp, flipped);
        f->fcWi->loadThetaOrderBwsmvFlipped(fp, flipped);
        f->fcWf->loadThetaOrderBwsmvFlipped(fp, flipped);
        f->fcWo->loadThetaOrderBwsmvFlipped(fp, flipped);
        f->fcWg->loadThetaOrderBwsmvFlipped(fp, flipped);
    }

    void FnLstm::saveTheta(FILE *fp) {
        auto f = this;
        if (f->implementationMode == 1) {
            f->fcWi->saveTheta(fp);
            f->fcWf->saveTheta(fp);
            f->fcWo->saveTheta(fp);
            f->fcWg->saveTheta(fp);
            f->fcUi->saveTheta(fp);
            f->fcUf->saveTheta(fp);
            f->fcUo->saveTheta(fp);
            f->fcUg->saveTheta(fp);
        } else if (f->implementationMode == 2) {
            f->fcWx->saveTheta(fp);
            f->fcUh->saveTheta(fp);
        } else {
            f->fcW->saveTheta(fp);
        }
    }

    void FnLstm::loadTheta(FILE *fp) {
        auto f = this;
        if (f->implementationMode == 1) {
            f->fcWi->loadTheta(fp);
            f->fcWf->loadTheta(fp);
            f->fcWo->loadTheta(fp);
            f->fcWg->loadTheta(fp);
            f->fcUi->loadTheta(fp);
            f->fcUf->loadTheta(fp);
            f->fcUo->loadTheta(fp);
            f->fcUg->loadTheta(fp);
        } else if (f->implementationMode == 2) {
            f->fcWx->loadTheta(fp);
            f->fcUh->loadTheta(fp);
        } else {
            f->fcW->loadTheta(fp);
        }
    }

    void FnLstm::resetState(int b) {
        auto f = this;
        math21_vector_set_wrapper(f->outputs, 0, f->h + f->outputs * b, 1);
    }
}
