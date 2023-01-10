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

#include "SteepestDescent.h"

using namespace math21;


SteepestDescent::SteepestDescent(sd_update_rule &update_rule, OptimizationInterface &oi) : update_rule(update_rule),
                                                                                           oi(oi) {
}

Functional &SteepestDescent::getFunctional() {
    return update_rule.f;
}

NumN SteepestDescent::getTime() {
    return update_rule.time;
}

void SteepestDescent::solve() {
    NumN stopTime = 0;
    while (1) {
        update_rule.update();
        if (xjabs(update_rule.y - update_rule.y_old) < XJ_EPS) {
            stopTime++;
            if (stopTime > 5) {
//                break;
            }
        } else {
            if (stopTime != 0) {
                stopTime = 0;
            }
        }
        if (update_rule.time % 1 == 0) {
            m21log("time", update_rule.time);
            m21log("y", update_rule.y_old);
        }
        update_rule.time++;
        oi.onFinishOneInteration(*this);
        if (update_rule.time >= update_rule.time_max) {
            break;
        }
    }
//    update_rule.x.log("minima");
}

// Taylor series: f(x+a) = sum(j:0->inf)(1/j! * (a * delta(x))^j * f(x)), where (d^j)f/dx^j = delta(x)^j * f(x)
// So the first few terms of the expansion are f(x+a) = f(x) + a.t * g(x) + 1/2 * a.t * H(x) * a,
// where g is gradient of f, H Hessian matrix of f.
// Weisstein, Eric W. "Taylor Series." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/TaylorSeries.html

// Newton's method
// For root-finding, we seek the zeros of f(x).
// For optimization, we seek the zeros of f'(x).
// 1) Newton root-finding in 1d:
// f(xk + a) = f(xk) + f'(xk)a, and in root-finding our goal is to find a such that f(xk + a) =0
// so x(k+1) = xk + a = xk - f(xk)/f'(xk)
// 2) optimization in 1d
// f(xk + a) = f(xk) + f'(xk)*a + 1/2 * f''(xk)a^2
// => df(xk + a)/da = f'(xk + a) = f'(xk) + f''(xk)*a
// Our goal is to find a such that f'(xk + a) = 0, so x(k+1) = xk + a = xk - f'(xk)/f''(xk)
// 3) optimization in nd
// f(xk + a) = f(xk) + f'(xk).t * a + 1/2 * a.t * f''(xk) * a
// => x(k+1) = xk + a = xk - f'(xk)/f''(xk) = xk - H.inv * f'(xk)
// https://relate.cs.illinois.edu/course/cs357-f15/file-version/03473f64afb954c74c02e8988f518de3eddf49a4/media/cs357-slides-newton2.pdf
