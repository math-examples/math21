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

#include "inner_simple.h"

namespace math21 {
    void math21_geometry_generate_line(MatR &data, NumR x1, NumR y1, NumR x2, NumR y2);

    void math21_geometry_generate_circle(MatR &data, NumR x, NumR y, NumR radius, NumN n);

    void math21_geometry_generate_ellipse(MatR &data, const MatR &A, const MatR &t, NumN n);

    void math21_geometry_generate_disk(MatR &data, NumR x, NumR y, NumR radius, NumN n);

    void math21_geometry_generate_sphere(MatR &A, NumN n);

    void math21_geometry_generator_color(TenR &colors, NumN n_colors, NumN n_points, NumB colorLast=0);

    void math21_geometry_generate_cube(TenR &cube, NumN n = 300, NumB useColor=1);

    void math21_geometry_generate_gmm(MatR &data,
                                      const VecR &log_proportions, const MatR &means,
                                      const TenR &covs_sqrt, NumB useColor = 0);
}