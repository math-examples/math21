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

#include <math21.h>

using namespace math21;
using namespace numerical_recipes;
using namespace ad;


// x shape: n_time_step * mbs * x_size, x_size = 128 or 256, and x in x_size is one-hot.
void math21_data_text_get_one_hot(MatR &x_mat, const unsigned char *data, VecSize &_mb_offsets, NumN x_size,
                                  NumSize n_data_x, NumN mbs, NumN n_time_step) {
    x_mat.setSize(n_time_step, mbs, x_size);
    auto x = x_mat.getDataAddress();
    NumSize *mb_offsets = _mb_offsets.getDataAddress();
    NumN i, j;
    for (j = 0; j < n_time_step; ++j) {
        for (i = 0; i < mbs; ++i) {
            unsigned char curr = data[(mb_offsets[i]) % n_data_x];
            if (curr == 0 || curr >= x_size) {
                math21_error("Bad char");
            }
            x[(j * mbs + i) * x_size + curr] = 1;
            mb_offsets[i] = (mb_offsets[i] + 1) % n_data_x;
        }
    }
}

void math21_data_rnn_read_text() {
    std::string filename = math21_string_to_string(MATH21_WORKING_PATH) + "/../y/autograd-master/examples/rnn.py";
    auto text = math21_string_read_file(filename.c_str());

    NumSize n_data_1d;
    NumN rnn_batch_size = 45;
    NumN n_time_step = 5;
    NumN x_size = 128;

    n_data_1d = strlen((const char *) text);
    VecSize rnn_batch_offsets;
    math21_pr_rand_VecSize(rnn_batch_offsets, rnn_batch_size);

    MatR x_mat;
    math21_data_text_get_one_hot(x_mat, text, rnn_batch_offsets, x_size,
                                 n_data_1d, rnn_batch_size, n_time_step);

    math21_vector_free_cpu(text);

    x_mat.log();
    std::string bin_file = math21_string_to_string(MATH21_WORKING_PATH) + "/../y/z2.bin";
    math21_io_save(bin_file.c_str(), x_mat);
}

void test_std() {
    m21log(__FUNCTION__);
    std::vector<int> v;
    NumN size = 10;
    std::string s;
    math21_tool_std_string_resize(s, size);
    m21log(s.size());
    math21_tool_std_vector_resize(v, size);
    m21log(v.size());
}

void test2() {
    TenR x(10);
    x.letters();
    x.log("xxxxx");

    m21point px = math21_cast_to_point(x);
    math21_point_log(px);
    auto py = math21_test_ad_logsumexp_like(px);
    math21_point_log(py);
    exit(0);
}

int main(int argc, char **) {
    timer time;
    time.start();

    test_std();
//     math21_data_rnn_read_text();
    math21_destroy();

    time.end();
    if (time.time() > 0) {
        m21log("\ntime used", time.time());
    }
    printf("\nmath21 unit test finish!\n");
    return 0;
}