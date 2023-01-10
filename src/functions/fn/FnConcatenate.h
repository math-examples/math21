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

#include "Fn.h"

namespace math21 {
    class FnConcatenate {
    private:
        VecN dis;
        NumZ axis;

        Seqce<FnMatType> tmps;
    public:
        FnConcatenate(const VecN &dis, NumZ axis_);

        virtual ~FnConcatenate();

        NumB forward(const Seqce<FnMatType> &xs, FnMatType &y);

        NumB backward(const FnMatType &dy, Seqce<FnMatType> &dxs);

        NumB backward_addto(const FnMatType &dy, Seqce<FnMatType> &dxs);
    };
}