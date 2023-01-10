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

#pragma once

#include "inner.h"

namespace math21 {
    namespace ad {
        namespace rnn_detail {
            struct ad_rnn_params {
                PointAd params_1d, init_hiddens, change_para, predict_para;
            };

            struct ad_lstm_params {
                PointAd params_1d, init_hiddens, change_para, predict_para,
                        init_cells, forget_para, ingate_para, outgate_para;
            };

            void split_rnn_params(const PointAd &params_point, ad_rnn_params &params,
                                  NumN input_size, NumN state_size, NumN output_size);

            void split_lstm_params(const PointAd &params_point, ad_lstm_params &params,
                                   NumN input_size, NumN state_size, NumN output_size);

            PointAd concat_and_multiply(const PointAd &weights, const Seqce<PointAd > &args);

            void updata_rnn(const PointAd &input, PointAd &hiddens, const ad_rnn_params &params);

            void update_lstm(const PointAd &input, PointAd &hiddens, PointAd &cells, const ad_lstm_params &params);

            PointAd hiddens_to_output_probs(const PointAd &hiddens, const PointAd &predict_para);
        }

        typedef struct {
            PointAd x;
            PointAd y;
        } lstm_f_pair_args;

        typedef struct {
            NumB printText;
            PointAd x;
            PointAd y;
            PointAd logprobs;
            PointAd data_targets;
        } lstm_f_callback_args;

        NumN ad_rnn_calculate_params_size(NumN input_size, NumN state_size, NumN output_size);

        NumN ad_lstm_calculate_params_size(NumN input_size, NumN state_size, NumN output_size);

        void ad_rnn_init_params(
                const PointAd &params_point,
                NumN input_size, NumN state_size, NumN output_size, NumR param_scale = 0.01);

        void ad_lstm_init_params(
                const PointAd &params_point,
                NumN input_size, NumN state_size, NumN output_size, NumR param_scale = 0.01);

        PointAd ad_rnn_predict(const PointAd &params_point, const PointAd &data_inputs,
                                NumN input_size, NumN state_size, NumN output_size);

        PointAd ad_lstm_predict(const PointAd &params_point, const PointAd &data_inputs,
                                 NumN input_size, NumN state_size, NumN output_size);

        void ad_lstm_only_hiddens(const PointAd &params_point, const PointAd &data_inputs,
                               Seqce <PointAd > &outputs,
                               NumN input_size, NumN state_size, NumN output_size);

        PointAd ad_rnn_log_likelihood(
                const PointAd &params_point, const PointAd &data_inputs, const PointAd &data_targets,
                NumN input_size, NumN state_size, NumN output_size);

        PointAd ad_rnn_part_log_likelihood(const PointAd &logprobs, const PointAd &data_targets);
    }

    void math21_data_text_lstm_generate_text(const TenR &x_value_final,
                                             NumN input_size,
                                             NumN state_size,
                                             NumN output_size,
                                             NumN n_lines,
                                             NumN sequence_length,
                                             NumN alphabet_size,
                                             NumN deviceType = m21_device_type_default);

    void math21_data_text_lstm_print_training_prediction(const TenR &x_cur, void *data);

    struct m21lstm_text_type_config {
        NumB predict;

        std::string textPath;
        std::string functionParasPath;
        std::string functionParasPath_init; // can be empty
        std::string functionParasPath_save; // can be empty

        // data
        NumN max_batches;
        NumN alphabet_size;

        // f
        NumN input_size;
        NumN state_size;
        NumN output_size;
        NumN time_steps;
        NumN deviceType;

        // optimization
        NumN num_iters;
        NumR step_size;
        NumB printTextInOpt;

        // generate text
        NumN n_lines;
        NumN sequence_length;

        NumB debug;

        m21lstm_text_type_config();

        void log(const char *name = 0) const;
    };

    // Create a graph using text, code, or graph. They are all equivalent.
    // Here graph is created from code.
    void math21_ml_lstm_text(const m21lstm_text_type_config &config);
}