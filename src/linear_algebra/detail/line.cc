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

#include "line.h"

namespace math21 {
    /*
 * Ax+By+C=0
 * y = x * dy/dx + b = kx+b
 * So A=dy=y1-y0, B=-dx=-(x1-x0), C=b*dx
 *
 * let 0<k<1, dx>0 => dy>0
 *
 * octant 1: (0<k<1)
 * let 0<k<1, dx>0, we want to get (x, y) from line starting from (x0, y0) to (x1, y1), here x0, y0, x1, y1, x, y is NumZ.
 * We have
 * (1)
 *    D1:=f(x0+1, y0+0.5) = A + 0.5B
 *    D1 >0  <=> (x, y) = (x0+1, y0+1)
 *    D1 <=0 <=> (x, y) = (x0+1, y0)
 * (2)
 *    let Di = f(xi ,ym)
 *    Di >0  <=> (x, y) = (xi, ym+0.5) => D(i+1) = f(xi+1, ym+1) => dD(i+1) = D(i+1) - Di = A+B
 *    Di <=0 <=> (x, y) = (xi, ym-0.5) => D(i+1) = f(xi+1, ym)   => dD(i+1) = D(i+1) - Di = A
 *
 * octant 8:
 * reflect the line over x axis
 * octant 2 or 3:
 * reflect the line over line y=x
 * # References
        - [https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm]
        - [Jack E. Bresenham, Algorithm for Computer Control of a Digital Plotter,
        IBM Systems Journal, 4(1):25-30, 1965](https://gitlab.cecs.anu.edu.au/pages/2018-S1/courses/comp1100/assignments/02/Bresenham.pdf)
 * */
        void m21Line2DIntegerIterator::line_octant_1_8(NumZ x0, NumZ y0, NumZ x1, NumZ y1) {
            MATH21_ASSERT(x1 >= x0)
            A = y1 - y0;
            B = x0 - x1;
            one = 1;
            if (A < 0) {
                A = -A;
                one = -1;
            }
            D = 2 * A + B; // D := 2*Di
            x = x0 - 1;
            y = y0;
            n = x1 + 1 - x0;
            i = 0;
        }

    m21Line2DIntegerIterator::m21Line2DIntegerIterator(NumZ x0, NumZ y0, NumZ x1, NumZ y1) {
            if (xjabs(y1 - y0) < xjabs(x1 - x0)) {
                octant18 = 1;
                if (x0 > x1) {
                    line_octant_1_8(x1, y1, x0, y0); // octant 4, 5
                } else {
                    line_octant_1_8(x0, y0, x1, y1); // octant 8, 1
                }
            } else {
                octant18 = 0;
                if (y0 > y1) {
                    line_octant_1_8(y1, x1, y0, x0); // octant 6, 7
                } else {
                    line_octant_1_8(y0, x0, y1, x1); // octant 2, 3
                }
            }
        }

        NumB m21Line2DIntegerIterator::next() {
            if (i == n)return 0;
            if (i > 0) {
                if (D > 0) {
                    y += one;
                    D += 2 * (A + B);
                } else {
                    D += 2 * A;
                }
            }
            ++x;
            ++i;
            return 1;
        }

        NumN m21Line2DIntegerIterator::size() const {
            return n;
        }

        void m21Line2DIntegerIterator::pos(NumZ &x_, NumZ &y_) const {
            if (octant18) {
                x_ = x;
                y_ = y;
            } else {
                x_ = y;
                y_ = x;
            }
        }

    void testLine2DIntegerIterator() {
        NumZ x0, y0, x1, y1;
        NumZ x, y;
        x0 = -8;
        y0 = 8;
        x1 = -15;
        y1 = -5;
        m21Line2DIntegerIterator line2DIntegerIterator(x0, y0, x1, y1);
        while (line2DIntegerIterator.next()) {
            line2DIntegerIterator.pos(x, y);
            printf("(%d, %d), ", x, y);
        }
    }
}