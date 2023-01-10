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
#include "FnRoute.h"

namespace math21 {

    FnRoute::FnRoute() {
        init();
    }

    FnRoute::~FnRoute() {}

    void FnRoute::init() {
        name = 0;
        mini_batch_size = 0;
    }

    // y has shape (nch, nr, nc)
    void FnRoute::create(const mlfunction_net *fnet,
                         int mini_batch_size_, const VecN &inputLayers_, NumN deviceType) {
        mini_batch_size = mini_batch_size_;
        inputLayers = inputLayers_;
        y.setDeviceType(deviceType);
        dy.setDeviceType(deviceType);
        name = math21_string_create_from_string("route");
        resize(fnet);
    }

    void FnRoute::resize(const mlfunction_net *fnet) {
        VecN inputSizes(inputLayers.size());

        int i;
        const mlfunction_node *first = fnet->nodes[inputLayers(1)];
        //    math21_ml_function_node_log(first, "first");
        int out_h, out_w, out_c; // nr_Y, nc_Y, nch_Y
        out_c = first->y_dim[0];
        out_h = first->y_dim[1];
        out_w = first->y_dim[2];
        inputSizes(1) = first->y_size;
        for (i = 1; i < inputLayers.size(); ++i) {
            int index = inputLayers(i+1);
            const mlfunction_node *next = fnet->nodes[index];
            inputSizes(i + 1) = next->y_size;
            if (next->y_dim[1] == first->y_dim[1] && next->y_dim[2] == first->y_dim[2]) {
                out_c += next->y_dim[0];
            } else {
                printf("%d %d, %d %d\n", next->y_dim[1], next->y_dim[2], first->y_dim[1], first->y_dim[2]);
                math21_error("route fail"); // ye add
            }
        }
        int outputs = (int)math21_op_vector_sum(inputSizes);
        MATH21_ASSERT(outputs==out_c*out_h*out_w);
        y.setSize(mini_batch_size, out_c, out_h, out_w);
        y = 0;
        dy.setSize(mini_batch_size, out_c, out_h, out_w);
        dy = 0;
    }

    void FnRoute::log(const char *varName) const {
        fprintf(stdout, "%s  ", name);
        fprintf(stdout, " %d", inputLayers(1)); // todo: may use id
        int i;
        for (i = 1; i < inputLayers.size(); ++i) {
            fprintf(stdout, " %d", inputLayers(i+1));
        }
        fprintf(stdout, "\n");
    }

    // concatenate convolution features
    // Y = (X1, Xn), Y = (y1, yb), Xi = (xi1, xib), yb = (x1b, xnb)
    void FnRoute::forward(mlfunction_net *fnet) {
        if (fnet->is_train) dy = 0;
        int ilayer;
        Seqce<FnMatType> xs((NumN) inputLayers.size());
        for (ilayer = 0; ilayer < inputLayers.size(); ++ilayer) {
            int index = inputLayers(ilayer+1);
            auto input = fnet->nodes[index]->y;
            auto input_dim = fnet->nodes[index]->y_dim;

            NumN i = (NumN) ilayer + 1;
            VecN d(4);
            d = mini_batch_size, input_dim[0], input_dim[1], input_dim[2];
            math21_operator_tensor_set_data_wrapper(xs.at(i), d, input);
        }

        FnConcatenate concatenate(VecN(), 2);
        auto data_bak = y.getDataAddressWrapper();
        concatenate.forward(xs, y);
        auto data_new = y.getDataAddressWrapper();
        MATH21_ASSERT_CODE(data_bak == data_new, "Space of y changed.");
    }

// (dX1, dXn) += dY, dY = (dy1, dyb), dXi = (dxi1, dxib), (dx1b, dxnb) += dyb
    void FnRoute::backward(mlfunction_net *fnet) {
        int ilayer;
        Seqce<FnMatType> dxs((NumN) inputLayers.size());
        VecN dis(inputLayers.size());

        for (ilayer = 0; ilayer < inputLayers.size(); ++ilayer) {
            int index = inputLayers(ilayer+1);
            PtrR32Wrapper delta_ = fnet->nodes[index]->dy;
            auto input_dim = fnet->nodes[index]->y_dim;

            NumN i = (NumN) ilayer + 1;
            VecN d(4);
            d = mini_batch_size, input_dim[0], input_dim[1], input_dim[2];
            math21_operator_tensor_set_data_wrapper(dxs.at(i), d, delta_);
            dis(i) = input_dim[0];
        }
        FnConcatenate concatenate(dis, 2);
        concatenate.backward_addto(dy, dxs);
    }

    void FnRoute::saveState(FILE *file) const {
        math21_vector_serialize_c_wrapper(file, (PtrR32Wrapper) y.getDataAddressWrapper(),
                                          y.size());
        math21_vector_serialize_c_wrapper(file, (PtrR32Wrapper) dy.getDataAddressWrapper(),
                                          dy.size());
    }
}
