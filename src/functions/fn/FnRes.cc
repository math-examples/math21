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
#include "FnConcatenate.h"
#include "FnRes.h"

namespace math21 {

    FnRes::FnRes() {
        init();
    }

    FnRes::~FnRes() {}

    void FnRes::init() {
        name = 0;
        mini_batch_size = 0;

        h = 0, w = 0, c = 0;
        out_h = 0, out_w = 0, out_c = 0;
        inputs = 0;
        outputs = 0;
        index = 0;
        delta = 0;
        output = 0;

        k1 = 0, k2 = 0;
//        activation = 0;
    }

    // shortcut
    void FnRes::create(
            int mini_batch_size_, int index,
            int w, int h, int c, int w2, int h2,
            int c2, NumN deviceType) {
        this->mini_batch_size = mini_batch_size_;
        y.setDeviceType(deviceType);
        dy.setDeviceType(deviceType);
        this->name = math21_string_create_from_string("res");

        this->w = w2;
        this->h = h2;
        this->c = c2;
        this->out_w = w;
        this->out_h = h;
        this->out_c = c;
        this->outputs = w * h * c;
        this->inputs = this->outputs;
        this->index = index;

        y.setSize(mini_batch_size, out_c, out_h, out_w);
        y = 0;
        dy.setSize(mini_batch_size, out_c, out_h, out_w);
        dy = 0;
        this->output = (PtrR32Wrapper) y.getDataAddressWrapper();
        this->delta = (PtrR32Wrapper) dy.getDataAddressWrapper();
    }

    void FnRes::resize(int h, int w) {
        assert(this->w == this->out_w);
        assert(this->h == this->out_h);
        this->w = this->out_w = w;
        this->h = this->out_h = h;
        this->outputs = w * h * this->out_c;
        this->inputs = this->outputs;

        y.setSize(mini_batch_size, out_c, out_h, out_w);
        y = 0;
        dy.setSize(mini_batch_size, out_c, out_h, out_w);
        dy = 0;
        this->output = (PtrR32Wrapper) y.getDataAddressWrapper();
        this->delta = (PtrR32Wrapper) dy.getDataAddressWrapper();
    }

    void FnRes::log(const char *varName) const {
        fprintf(stdout, "%s: (%d, %d, %d, %d) -> (%d, %d, %d, %d), index = %d\n",
                name,
                h, w, c, mini_batch_size,
                out_h, out_w, out_c, mini_batch_size,
                index
        );
    }


    // Z = h(Y), Y = k1*X1 + k2*X2
    void FnRes::forward(mlfunction_net *fnet, mlfunction_node *finput) {
        if (fnet->is_train) {
            math21_vector_set_wrapper(mini_batch_size * outputs, 0, delta, 1);
        }
        // Y <- X2
        math21_vector_assign_from_vector_wrapper(outputs * mini_batch_size, finput->y, 1, output, 1);

        // Y <- k1*X1 + k2*Y
        math21_vector_feature2d_add_2_wrapper(mini_batch_size,
                                              k1, fnet->nodes[index]->y, c, h, w,
                                              k2, output, out_c, out_h, out_w);

        // Z = h(Y)
        math21_function_activation_vector_wrapper(output, outputs * mini_batch_size, activation);
    }

    void FnRes::backward(mlfunction_net *fnet, mlfunction_node *finput) {
        // dL/dY = dL/dZ *.ele h.d(Y)
        math21_function_activation_gradient_vector_wrapper(output, outputs * mini_batch_size, activation, delta);

        math21_tool_assert_to_do_remove(1);
        // dL/dX2 += k2*dL/dY
        if (1) {
            math21_vector_kx_add_y_wrapper(outputs * mini_batch_size, k2, delta, 1, finput->dy, 1);
        }

        // by ye
        // dL/dX2 += k2*dL/dY
        // dL/dX2 += dL/dY
        if (0) {
            math21_vector_kx_add_y_wrapper(outputs * mini_batch_size, 1, delta, 1, finput->dy, 1);
            // dL/dX2 += (k2 -1)*dL/dY
            math21_vector_feature2d_add_3_wrapper(
                    mini_batch_size,
                    0, fnet->nodes[index]->y, c, h, w,
                    (k2 - 1), delta, out_c, out_h, out_w,
                    1, finput->dy, out_c, out_h, out_w
            );
        }
        // dL/dX1 += k1*dL/dY
        math21_vector_feature2d_add_2_wrapper(mini_batch_size,
                                              k1, delta, out_c, out_h, out_w,
                                              1, fnet->nodes[index]->dy, c, h, w);
    }

    void FnRes::saveState(FILE *file) const {
        math21_vector_serialize_c_wrapper(file, output, mini_batch_size * outputs);
        math21_vector_serialize_c_wrapper(file, delta, mini_batch_size * outputs);
    }
}
