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

#include "inner.h"
#include "f_ex_sin.h"

namespace math21 {
    sine::sine() : Functional() {
        A.setSize(1);
        x0.setSize(1);
        x0 = 300;
    }

    NumR sine::valueAt(const VecR &x) {
        return xjsin(x(1));
    }

    NumN sine::getXDim() {
        return 1;
    }

    const VecR &sine::getX0() {
        return x0;
    }

    const VecR &sine::derivativeValueAt(const VecR &x) {
        A(1) = xjcos(x(1));
        return A;
    }
}