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

#include <map>
#include <random>
#include "../functions/files.h"
#include "../algebra/files.h"
#include "../linear_algebra/files.h"
#include "../image/files.h"
#include "inner.h"
#include "plot.h"

namespace math21 {

    void test_draw_random_normal_method1(NumB is_print) {
        RanNormal ran;

        NumN n = 200000;
        NumN nstars = 100;

        VecR m(n);
        ran.set(5, 2);
        math21_random_draw(m, ran);
        VecN p(10);
        p = 0;

        for (NumN i = 0; i < n; ++i) {
            double number = m(i + 1);
            if ((number >= 0.0) && (number < 10.0)) ++p(int(number) + 1);
        }

        if (is_print) {
            const char *name = "normal_distribution (5.0,2.0):";
            math21_pr_log_histogram_Y(p, name);
        }
    }

    void test_draw_random_normal_method2(NumB is_print) {
        const int nrolls = 200000;  // number of experiments
        const int nstars = 100;    // maximum number of stars to distribute

        std::default_random_engine generator;
        std::normal_distribution<double> distribution(5.0, 2.0);

        VecN p(10);
        p = 0;

        for (int i = 0; i < nrolls; ++i) {
            double number = distribution(generator);
            if ((number >= 0.0) && (number < 10.0)) ++p(int(number) + 1);
        }

        if (is_print) {
            const char *name = "normal_distribution (5.0,2.0):";
            math21_pr_log_histogram_Y(p, name);
        }
    }

    void test_time_random_normal() {
        NumN n = 1000;
        MATH21_PRINT_TIME_START()
        for (NumN i = 1; i <= n; ++i) {
            test_draw_random_normal_method1(0);
        }
        MATH21_PRINT_TIME_END("")
        MATH21_PRINT_TIME_START()
        for (NumN i = 1; i <= n; ++i) {
            test_draw_random_normal_method2(0);
        }
        MATH21_PRINT_TIME_END("")
    }

    void test_draw_random_uniform_method1(NumB is_print) {
        RanUniform ranUniform;
        ranUniform.set(0, 9);
        NumN n = 20000;
        VecN M(n);
        math21_random_draw(M, ranUniform);

        Dict<NumN, NumN> hist;
        for (NumN i = 1; i <= 20000; ++i) {
            if (hist.has(M(i))) {
                NumN &y = hist.valueAt(M(i));
                ++y;
            } else {
                hist.add(M(i), 1);
            }
        }
        if (is_print) {
            const char *name = "uniform distribution (0, 9):";
            VecN X, Y;
            hist.getX(X);
            hist.getY(Y);
            math21_pr_log_histogram_XY(X, Y, name);
        }
    }

    void test_draw_random_uniform_method2(NumB is_print) {
        std::random_device rd;
        std::uniform_int_distribution<int> dist(0, 9);

        Dict<NumN, NumN> hist;
        for (NumN i = 1; i <= 20000; ++i) {
            auto x = (NumN) dist(rd);
            if (hist.has(x)) {
                NumN &y = hist.valueAt(x);
                ++y;
            } else {
                hist.add(x, 1);
            }
        }
        if (is_print) {
            const char *name = "uniform distribution (0, 9):";
            VecN X, Y;
            hist.getX(X);
            hist.getY(Y);
            math21_pr_log_histogram_XY(X, Y, name);
        }
    }

    void test_time_random_uniform() {
        NumN n = 1000;
        MATH21_PRINT_TIME_START()
        for (NumN i = 1; i <= n; ++i) {
            test_draw_random_uniform_method1(0);
        }
        MATH21_PRINT_TIME_END("")
        MATH21_PRINT_TIME_START()
        for (NumN i = 1; i <= n; ++i) {
            test_draw_random_uniform_method2(0);
        }
        MATH21_PRINT_TIME_END("")
    }

    void test_draw_random_binomial(NumB is_print) {
        NumN size = 4;
        RanBinomial ran(size, 0.5);

        NumN n = 200000;
        VecN m(n);
        math21_random_draw(m, ran);
        VecN p(size + 1);
        p = 0;

        for (NumN i = 1; i <= m.size(); ++i) {
            NumN number = m(i);
            p(number + 1)++;
        }

        p.log("p");
        if (is_print) {
            const char *name = "binomial_distribution (9, 0.5):";
            math21_pr_log_histogram_Y(p, name);
        }
    }

    void test_draw_random_normal() {
        test_draw_random_normal_method1(1);
        test_draw_random_normal_method2(1);

//        test_time_random_normal();
    }

    void test_draw_random_uniform() {
        test_draw_random_uniform_method1(1);
        test_draw_random_uniform_method2(1);

//        test_time_random_uniform();
    }

    void test_draw_random_binomial() {
        test_draw_random_binomial(1);
    }

    void math21_plot_set_color_red(VecN &color) {
        if (!color.isSameSize(3) && !color.isSameSize(4)) {
            color.setSize(3);
        }
        if (color.size() == 3) {
            color = 255, 0, 0;
        } else {
            color = 255, 0, 0, 255;
        }
    }

    void math21_plot_set_color_green(VecN &color) {
        if (!color.isSameSize(3) && !color.isSameSize(4)) {
            color.setSize(3);
        }
        if (color.size() == 3) {
            color = 0, 255, 0;
        } else {
            color = 0, 255, 0, 255;
        }
    }

    void math21_plot_set_color_blue(VecN &color) {
        if (!color.isSameSize(3) && !color.isSameSize(4)) {
            color.setSize(3);
        }
        if (color.size() == 3) {
            color = 0, 0, 255;
        } else {
            color = 0, 0, 255, 255;
        }
    }

    void math21_plot_set_color(VecN &color) {
        RanUniform ranUniform;
        ranUniform.set(0, 255);
        if (!color.isSameSize(3)) {
            color.setSize(3);
        }
        math21_random_draw(color, ranUniform);
    }

    void math21_plot_set_color(Seqce<VecN> &colors) {
        RanUniform ranUniform;
        ranUniform.set(0, 255);
        for (NumN i = 1; i <= colors.size(); ++i) {
            if (!colors.at(i).isSameSize(3)) {
                colors.at(i).setSize(3);
            }
            math21_random_draw(colors.at(i), ranUniform);
        }
        if (colors.size() >= 1) {
            colors.at(1) = 255, 0, 0;
        }
        if (colors.size() >= 2) {
            colors.at(2) = 0, 255, 0;
        }
        if (colors.size() >= 3) {
            colors.at(3) = 0, 0, 255;
        }
        if (colors.size() >= 4) {
            colors.at(4) = 255, 255, 0;
        }
        if (colors.size() >= 5) {
            colors.at(5) = 255, 0, 255;
        }
        if (colors.size() >= 6) {
            colors.at(6) = 0, 255, 255;
        }
        if (colors.size() >= 7) {
            colors.at(7) = 255, 255, 255;
        }
    }

    m21_plot_args &m21_plot_args::operator=(const m21_plot_args &args) {
        this->color = args.color;
        this->border = args.border;
        this->radius = args.radius;
        return *this;
    }

    void math21_plot_point_no_radius(NumR x, NumR y, TenR &A, const m21_plot_args &plotArgs) {
        MATH21_ASSERT(A.dims() == 3)
        MATH21_ASSERT(A.dim(1) >= 3)
        NumN nr = A.dim(2);
        NumN nc = A.dim(3);
        NumN nch = xjmin(A.dim(1), plotArgs.color.size());
        NumN i = (NumN) x;
        NumN j = (NumN) y;
        NumN s1, e1, s2, e2;
        s1 = 1;
        e1 = nr;
        s2 = 1;
        e2 = nc;
        if (!plotArgs.border.isEmpty()) {
            s1 = xjmax(s1, plotArgs.border(1).left());
            e1 = xjmin(e1, plotArgs.border(1).right());
            s2 = xjmax(s2, plotArgs.border(2).left());
            e2 = xjmin(e2, plotArgs.border(2).right());
        }
        if (xjIsIn(i, s1, e1) && xjIsIn(j, s2, e2)) {
            for (NumN k = 1; k <= nch; ++k) {
                A(k, i, j) = plotArgs.color(k);
            }
        }
    }

    // plot data to image
    void math21_plot_mat_data_point_no_radius(const MatR &data, TenR &A, const m21_plot_args &plotArgs) {
        for (NumN i = 1; i <= data.nrows(); ++i) {
            math21_plot_point_no_radius(data(i, 1), data(i, 2), A, plotArgs);
        }
    }

    // radius > 0 : draw from circle.
    // radius < 0 : draw from disk
    void math21_plot_point_with_radius(NumR x, NumR y, TenR &A, const m21_plot_args &plotArgs) {
        MATH21_ASSERT(A.dims() == 3)
        MATH21_ASSERT(A.dim(1) >= 3)
        if (plotArgs.radius == 0) {
            math21_plot_point_no_radius(x, y, A, plotArgs);
        } else {
            MatR data;
            if (plotArgs.radius > 0) {
                math21_geometry_generate_circle(data, x, y, plotArgs.radius, (NumN) (10 * plotArgs.radius));
            } else {
                NumR radius = -plotArgs.radius;
                math21_geometry_generate_disk(data, x, y, radius, (NumN) (10 * radius * radius));
            }
            math21_plot_mat_data_point_no_radius(data, A, plotArgs);
        }
    }

    // see math21_image_draw_line_vertical
    void math21_plot_line(NumR x1, NumR y1, NumR x2, NumR y2,
                          TenR &A, const m21_plot_args &plotArgs) {
        MatR data;
        math21_geometry_generate_line(data, x1, y1, x2, y2);
        math21_plot_mat_data_point_no_radius(data, A, plotArgs);
    }

    // plot data to image
    void math21_plot_mat_data_point_no_option(const MatR &data, TenR &A, const m21_plot_args &plotArgs) {
        for (NumN i = 1; i <= data.nrows(); ++i) {
            math21_plot_point_with_radius(data(i, 1), data(i, 2), A, plotArgs);
        }
    }

    void math21_plot_mat_data_line(const MatR &data, TenR &A, const m21_plot_args &plotArgs) {
        for (NumN i = 2; i <= data.nrows(); ++i) {
            math21_plot_line(data(i - 1, 1), data(i - 1, 2), data(i, 1), data(i, 2), A, plotArgs);
        }
    }

    // plot data to image
    void math21_plot_mat_data_point_with_option(
            const MatR &data, TenR &A, const m21_plot_args &plotArgs, NumN index_i) {
        for (NumN i = 1; i <= data.nrows(); ++i) {
            if (index_i != 0) {
                if (i != index_i) {
                    continue;
                }
            }
            math21_plot_point_with_radius(data(i, 1), data(i, 2), A, plotArgs);
        }
    }

    // plot data to image
    void math21_plot_container_data(const Seqce<MatR> &data, TenR &A, const m21_plot_args &plotArgs) {
        for (NumN i = 1; i <= data.size(); ++i) {
            math21_plot_mat_data_point_no_option(data(i), A, plotArgs);
        }
    }

    void math21_plot_container_with_option(const Seqce<MatR> &data, TenR &A, NumN index_i, NumN index_j) {
        NumN side = xjmin(math21_image_get_nr(A), math21_image_get_nc(A));
        m21_plot_args plotArgs;
        plotArgs.color.setSize(3);
        plotArgs.radius = -(side / 100.0);
        for (NumN i = 1; i <= data.size(); ++i) {
            if (index_i != 0) {
                if (i != index_i) {
                    continue;
                }
            }
            RanUniform ranUniform;
            ranUniform.set(0, 255);
            if (i == 1) {
                math21_plot_set_color_red(plotArgs.color);
            } else if (i == 2) {
                math21_plot_set_color_green(plotArgs.color);
            } else if (i == 3) {
                math21_plot_set_color_blue(plotArgs.color);
            } else { // see math21_image_color_get_red
                math21_random_draw(plotArgs.color, ranUniform);
            }
            if (index_j != 0) {
                math21_plot_point_with_radius(data(i)(index_j, 1), data(i)(index_j, 2), A, plotArgs);
            } else {
                math21_plot_mat_data_point_no_option(data(i), A, plotArgs);
            }

        }
    }
}