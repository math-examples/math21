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

    class NameBaseClass {
    public:
        virtual std::string getClassName() const = 0;
    };

    /*
     * y=T(x)
     *
     * */
    class FunctionNd : public NameBaseClass, public think::Operator {
    private:
        NumN debugValue;
    public:
        FunctionNd() { debugValue = 0; }

        virtual ~FunctionNd() {}

        // compute f(x)
        virtual NumB valueAt(const VecR &x, VecR &f) { return 0; }

        std::string getClassName() const override {
            return "FunctionNd";
        }

        void setDebugValue(NumN value_) {
            debugValue = value_;
        }

        NumN getDebugValue() const {
            return debugValue;
        }
    };

}