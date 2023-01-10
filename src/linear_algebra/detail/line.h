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
    class m21Line2DIntegerIterator {
    private:
        NumZ A;
        NumZ B;
        NumZ one;
        NumZ D;
        NumZ x;
        NumZ y;
        NumN n;
        NumN i;
        NumB octant18;

        void line_octant_1_8(NumZ x0, NumZ y0, NumZ x1, NumZ y1);

    public:
        m21Line2DIntegerIterator(NumZ x0, NumZ y0, NumZ x1, NumZ y1);

        NumB next();

        NumN size() const;

        void pos(NumZ &x_, NumZ &y_) const;
    };

    void testLine2DIntegerIterator();
}