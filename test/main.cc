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

#include <cstdio>
#include "test.h"

using namespace math21;

int main(int argc, char **argv) {
    timer time;
    time.start();
//    test_data_structure();
//    test_tensor();
//    test_gpu_matrix_multiply();
    test_ad();
//    test_opt();
//    test_algebra();
//    test_draw();
//    test_3rdparty_tools();
//    f_ex_cos_sin_lm_test();
//    f_ex_x3_x2_x_fdm_test();
//    f_ex_cos_sin_lm_fdm_test();

    math21_destroy();

    time.end();
    if (time.time() > 0) {
        m21log("\ntime used", time.time());
    }
    printf("\nmath21 test finish!\n");
    return 0;
}
