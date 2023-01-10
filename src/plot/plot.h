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

#include "inner_header.h"

namespace math21 {

    void test_time_random_normal();

    void test_draw_random_normal();

    void test_time_random_uniform();

    void test_draw_random_uniform();

    void test_draw_random_binomial();

    void math21_plot_set_color_red(VecN &color);

    void math21_plot_set_color_green(VecN &color);

    void math21_plot_set_color_blue(VecN &color);

    void math21_plot_set_color(VecN &color);

    void math21_plot_set_color(Seqce <VecN> &colors);

    struct m21_plot_args {
        VecN color;
        Interval2D border;
        NumR radius;

        m21_plot_args() {radius=0;}

        m21_plot_args(const m21_plot_args &args) {
            *this = args;
        }

        m21_plot_args &operator=(const m21_plot_args &);
    };

    void math21_plot_point_no_radius(NumR x, NumR y, TenR &A, const m21_plot_args &plotArgs);

    void math21_plot_point_with_radius(NumR x, NumR y, TenR &A, const m21_plot_args &plotArgs);

    void math21_plot_line(NumR x1, NumR y1, NumR x2, NumR y2,
                          TenR &A, const m21_plot_args &plotArgs);

    // plot data to image
    void math21_plot_mat_data_point_no_option(const MatR &data, TenR &A, const m21_plot_args &plotArgs);

    void math21_plot_mat_data_line(const MatR &data, TenR &A, const m21_plot_args &plotArgs);

    void math21_plot_mat_data_point_with_option(
            const MatR &data, TenR &A, const m21_plot_args &plotArgs, NumN index_i = 0);

    // plot data to image
    void math21_plot_container_data(const Seqce <MatR> &data, TenR &A, const m21_plot_args &plotArgs);

    void math21_plot_container_with_option(const Seqce <MatR> &data, TenR &A, NumN index_i, NumN index_j);
}