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
#include "FnMatMul.h"

namespace math21 {
    FnMatMul::FnMatMul(const VecN &dis_, NumZ axis_) : dis(dis_), axis(axis_) {}

    FnMatMul::~FnMatMul() {}

    NumB FnMatMul::forward(const Seqce<FnMatType> &xs, FnMatType &y) {

        return 1;
    }

    NumB FnMatMul::backward(const FnMatType &dy, Seqce<FnMatType> &dxs) {

        return 1;
    }

    // todo: fuse
    NumB FnMatMul::backward_addto(const FnMatType &dy, Seqce<FnMatType> &dxs) {

        return 1;
    }
}