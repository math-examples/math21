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
#include "basic/files.h"

namespace math21 {
    struct OptUpdate_LevMar {
    public:
        NumN max_iters;
        void *data;
        const MatR *y;
        const VecR *theta_0;

        void (*f)(const VecR &paras, MatR &y, const void *data);

        VecR *theta_est;
        NumN logLevel;

        OptUpdate_LevMar() {
            max_iters = 200;
            data = 0;
            y = 0;
            theta_0 = 0;
            f = 0;
            theta_est = 0;
            logLevel = 0;
        }
    };

    void math21_opt_levmar_fdm(void *data, const MatR &y, const VecR &theta_0,
                               void (*f)(const VecR &paras, MatR &y, const void *data), VecR &theta_est,
                               NumN max_iters, NumN logLevel = 0);

    struct OptParasLevMarq {
        VecN mask;
        NumN maxIters;
        NumN strategy; // see LevMarStrategySimple
        NumR eps;
        NumN logLevel;

        OptParasLevMarq() {
            maxIters = 10;
            strategy = LevMarStrategyMatlab;
            eps = FLT_EPSILON;
            logLevel = 0;
        }
    };

    // find the least squares solution x_min, x_min = argmin L(x), where L(x) = 1/2 * ||f(x)||^2
    // dL/dx = dL/df * df/dx = f.t * J
    // Note if L = 1/2 * ||f(x) - f_true||^2, then f(x) <- f(x) - f_true
    class OptCallbackLevMarq : public NameBaseClass {
    private:
        NumN debugValue;
    public:
        OptCallbackLevMarq() { debugValue = 0; }

        virtual ~OptCallbackLevMarq() {}

        // compute f(x) and J
        // theta and value must be vectors. J is matrix.
        // value = f_est - f_true, if f(x) <- f(x) - f_true
        // value = f_est,   otherwise
        virtual NumB compute(const VecR &x, VecR &f, MatR *pJ) { return 0; }

        std::string getClassName() const override {
            return "OptCallbackLevMarq";
        }

        void setDebugValue(NumN value_) {
            debugValue = value_;
        }

        NumN getDebugValue() const {
            return debugValue;
        }
    };

    /*
     * let
     * m = J.nrows(), n = J.ncols()
     * Caller must make sure that m >= n when strategy = 3
     * see MATH21_ASSERT(m >= n, "Necessary condition not met!") in lm
     * */
    class OptCallbackSparseLevMarq : public NameBaseClass {
    private:
        NumN debugValue;
    public:
        OptCallbackSparseLevMarq() { debugValue = 0; }

        virtual ~OptCallbackSparseLevMarq() {}

        // dLt = L'.t = Jtf = J.t * f
        // JtJ = J.t * J
        // loss2 = 2*L(x) = ||f(x)||^2, L(x) = 1/2 * ||f(x)||^2
        virtual NumB compute(const VecR &x, VecR *Jtf, MatR *JtJ, NumR *loss2) {
            return 0;
        }

        std::string getClassName() const override {
            return "OptCallbackSparseLevMarq";
        }

        virtual void setDebugValue(NumN debugValue_) {
            debugValue = debugValue_;
        }

        virtual NumN getDebugValue() const {
            return debugValue;
        }
    };

    OptDetail *math21_opt_create_LevMarq(const OptParasLevMarq &paras, FunctionNd *cb);

    OptDetail *math21_opt_create_LevMarq(const OptParasLevMarq &paras, OptCallbackLevMarq *cb);

    OptDetail *math21_opt_create_LevMarq(const OptParasLevMarq &paras, OptCallbackSparseLevMarq *cb);

    void math21_opt_destroy_LevMarq(OptDetail *);
}