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

#include "../op/files.h"
#include "update/files.h"
#include "adam.h"

namespace math21 {
    template<typename T, typename FunType>
    void sgd_with_momentum(const FunType &grad, VecR &x, const FunType &callback = 0,
                           NumN num_iters = 200, NumR step_size = 0.1, NumR mass = 0.9) {
        // Stochastic gradient descent with momentum
        // grad() must have signature grad(x, g, i), where i is the iteration number.
        // x is input and output.
        VecR velocity(x.size());
        velocity = 0;
        NumN i;
        VecR g;
        for (i = 1; i <= num_iters; ++i) {
            grad(x, g, i);
            if (callback) {
                callback(x, g, i);
            }
            // velocity = mass * velocity - (1.0 - mass) * g
            math21_operator_container_linear_to_A(mass, velocity, -(1.0 - mass), g);
            // x = x + step_size * velocity
            math21_operator_container_linear_to_A(1, x, step_size, velocity);
        }
    }

    // deprecate, use below
    void math21_opt_adam(
            TenR &x,
            void (*grad)(const TenR &x, TenR &dy, NumN iter, void *grad_data),
            void *grad_data,
            void (*callback)(const TenR &x_cur, const TenR &gradient, NumN iter, void *callback_data),
            void *callback_data,
            const OptParasAdam &parasAdam) {
        TenR m;
        m.setDeviceType(x.getDeviceType());
        m.setSize(x.size());
        TenR v;
        v.setDeviceType(x.getDeviceType());
        v.setSize(x.size());
        math21_op_vector_set_value(m, 0);
        math21_op_vector_set_value(v, 0);
        TenR g;
        g.setDeviceType(x.getDeviceType());
        for (NumN i = 1; i <= parasAdam.num_iters; ++i) {
            grad(x, g, i, grad_data);
            if (callback) {
                callback(x, g, i, callback_data);
            }
            math21_op_vector_kx_onto(-1, g);
            math21_op_optimization_adam_update(
                    x, g, m,
                    v, parasAdam.b1, parasAdam.b2,
                    parasAdam.eps, 0, parasAdam.step_size, 0, i);
        }
    }

    void math21_opt_adam(
            TenR &x, const OptParasAdam &parasAdam, const OptCallbackAdam &gradCb,
            const OptPrintCallbackAdam *printCb) {
        TenR m;
        m.setDeviceType(x.getDeviceType());
        m.setSize(x.size());
        TenR v;
        v.setDeviceType(x.getDeviceType());
        v.setSize(x.size());
        math21_op_vector_set_value(m, 0);
        math21_op_vector_set_value(v, 0);
        TenR g;
        g.setDeviceType(x.getDeviceType());
        for (NumN i = 1; i <= parasAdam.num_iters; ++i) {
            gradCb.compute(x, g, i);
            if (printCb) {
                printCb->log(x, g, i);
            }
            math21_op_vector_kx_onto(-1, g);
            math21_op_optimization_adam_update(
                    x, g, m,
                    v, parasAdam.b1, parasAdam.b2,
                    parasAdam.eps, 0, parasAdam.step_size, 0, i);
        }
    }

    class OptAdam : public OptDetail {
    private:
        const OptParasAdam &parasAdam;
        const OptCallbackAdam &gradCb;
        const OptPrintCallbackAdam *printCb;

    public:
        OptAdam(const OptParasAdam &parasAdam, const OptCallbackAdam &gradCb, const OptPrintCallbackAdam *printCb)
                : parasAdam(parasAdam), gradCb(gradCb), printCb(printCb) {}

        NumB run(VecR &x0) override {
            math21_opt_adam(x0, parasAdam, gradCb, printCb);
            return 1;
        }
    };


    OptDetail *math21_opt_create_adam(const OptParasAdam &parasAdam, const OptCallbackAdam &gradCb,
                                      const OptPrintCallbackAdam *printCb) {
        return new OptAdam(parasAdam, gradCb, printCb);
    }

    void math21_opt_destroy_adam(OptDetail *detail) {
        delete (OptAdam *) (detail);
    }
}