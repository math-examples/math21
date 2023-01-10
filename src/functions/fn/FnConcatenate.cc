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

namespace math21 {
    FnConcatenate::FnConcatenate(const VecN &dis_, NumZ axis_) : dis(dis_), axis(axis_) {}

    FnConcatenate::~FnConcatenate() {}

    NumB FnConcatenate::forward(const Seqce<FnMatType> &xs, FnMatType &y) {
        math21_op_tensor_concatenate(xs, y, axis);
        return 1;
    }

    NumB FnConcatenate::backward(const FnMatType &dy, Seqce<FnMatType> &dxs) {
        math21_op_tensor_split(dy, dxs, dis, axis);
        return 1;
    }

    // todo: fuse
    NumB FnConcatenate::backward_addto(const FnMatType &dy, Seqce<FnMatType> &dxs) {
        math21_op_tensor_split(dy, tmps, dis, axis);
        for (NumN i = 1; i <= dxs.size(); ++i){
            math21_op_add_onto_2(tmps(i), dxs(i));
        }
        return 1;
    }
}