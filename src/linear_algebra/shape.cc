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

#include "detail/files.h"
#include "inner_cc.h"
#include "../image/files.h"
#include "geometryTrans.h"
#include "shape.h"

namespace math21 {

    void math21_geometry_generate_line(MatR &data, NumR x1, NumR y1, NumR x2, NumR y2) {
        NumZ x, y;
        m21Line2DIntegerIterator iterator((NumZ) xjround(x1), (NumZ) xjround(y1),
                                          (NumZ) xjround(x2), (NumZ) xjround(y2));
        NumN n = iterator.size();
        if (!data.isSameSize(n, 2)) {
            data.setSize(n, 2);
        }
        NumN i = 1;
        while (iterator.next()) {
            iterator.pos(x, y);
            data(i, 1) = x;
            data(i, 2) = y;
            ++i;
        }
    }

    // draw n points from circle, radius >= 0
    void math21_geometry_generate_circle(MatR &data, NumR x, NumR y, NumR radius, NumN n) {
        if (!data.isSameSize(n, 2)) {
            data.setSize(n, 2);
        }
        MATH21_ASSERT(n > 0)
        NumR theta = (2 * MATH21_PI) / n;
        for (NumN i = 0; i < n; ++i) {
            data(i + 1, 1) = x + radius * xjcos(i * theta);
            data(i + 1, 2) = y + radius * xjsin(i * theta);
        }
    }

    // Linear transformations take ellipses to ellipses. (need proof)
    // draw n points from ellipse
    void math21_geometry_generate_ellipse(MatR &data, const MatR &A, const MatR &t, NumN n) {
        if (!data.isSameSize(n, 2)) {
            data.setSize(n, 2);
        }
        MATH21_ASSERT(n > 0);
        MATH21_ASSERT(A.isSameSize(2, 2));
        MATH21_ASSERT(t.isSameSize(2));
        VecR theta(n);
        theta.letters(0);
        math21_op_vector_kx_onto(2 * MATH21_PI / n, theta);
        VecR v;
        math21_op_cos(theta, v);
        math21_op_matrix_set_col(v, data, 1);
        math21_op_sin(theta, v);
        math21_op_matrix_set_col(v, data, 2);
        MatR data2;
        math21_geometry_affine_non_homogeneous(A, t, data, data2);
        data = data2; // swap is dangerous.
    }

    // Todo: prove
    // draw n points from disk uniformly, radius >= 0
    void math21_geometry_generate_disk(MatR &data, NumR x, NumR y, NumR radius0, NumN n) {
        MATH21_ASSERT(n > 0)
        if (!data.isSameSize(n, 2)) {
            data.setSize(n, 2);
        }
        RanUniform ranUniform;
        NumR ran;
        for (NumN i = 1; i <= n; ++i) {
            math21_random_draw(ran, ranUniform);
            NumR radius;
            radius = radius0 * xjsqrt(ran);
            math21_random_draw(ran, ranUniform);
            NumR theta = 2 * MATH21_PI * ran;
            data(i, 1) = x + radius * xjcos(theta);
            data(i, 2) = y + radius * xjsin(theta);
        }
    }

    // need proof
    // draw from unit sphere uniformly
    void math21_geometry_generate_sphere(MatR &A, NumN n) {
        if (A.isSameSize(n, 3) == 0) {
            A.setSize(n, 3);
        }
        NumR theta1, theta2, r2;
        RanUniform ranUniform1;
        ranUniform1.set(0, 2 * MATH21_PI);
        RanUniform ranUniform2;
        ranUniform2.set(-1, 1);
        for (NumN i = 1; i <= n; ++i) {
            math21_random_draw(theta1, ranUniform1);
            math21_random_draw(r2, ranUniform2);
            theta2 = xjacos(r2) - 0.5 * MATH21_PI;
            A(i, 1) = xjcos(theta2) * xjcos(theta1);
            A(i, 2) = xjcos(theta2) * xjsin(theta1);
            A(i, 3) = xjsin(theta2);
        }
    }

    // colors: n_colors * 3 * n_points, or n_colors * n_points * 3
    void math21_geometry_generator_color(TenR &colors, NumN n_colors, NumN n_points, NumB colorLast) {
        NumN n = n_colors;
        colors.setSize(n, 3, n_points);
        VecR color;
        for (NumN i = 1; i <= n; ++i) {
            math21_image_color_generate(color, i, n);
            TenR sub;
            math21_operator_share_tensor_row_i(colors, i, sub);
            for (NumN j = 1; j <= 3; ++j) {
                TenR v;
                math21_operator_share_tensor_row_i(sub, j, v);
                v = color(j);
            }
        }
        if (colorLast) {
            TenR colors2;
            math21_op_tensor_swap_axes(colors, colors2, 2, 3);
            colors = colors2;
        }
    }

    // [0, 1]^3
    void math21_geometry_generate_cube(TenR &data, NumN n, NumB useColor) {
        TenR cube;
        if (useColor) {
            cube.setSize(6, 6, n, n);
        } else {
            cube.setSize(6, 3, n, n);
        }
        TenR colors;
        if (useColor) {
            TenR x;
            math21_geometry_generator_color(x, 6, n * n);
            math21_operator_share_to_tensor4d(x, colors, x.dim(1), x.dim(2), n, n);
        }

        TenR facex0, facex1, facey0, facey1, facez0, facez1;
        Seqce<TenR *> faces(6);
        faces = &facez0, &facez1, &facex0, &facex1, &facey0, &facey1;

        for (NumN i = 1; i <= 6; ++i) {
            TenR oneface;
            math21_operator_share_tensor_row_i(cube, i, oneface);
            TenR part1;
            math21_operator_share_tensor_rows(oneface, 0, 3, part1);
            math21_operator_share_copy(part1, *faces(i));
            if (useColor) {
                TenR part2;
                math21_operator_share_tensor_rows(oneface, 3, 3, part2);
                TenR color;
                math21_operator_share_tensor_row_i(colors, i, color);
                part2 = color;
            }
        }

        VecR t(n);
        t.letters(0);
        math21_op_vector_kx_onto(1.0 / n, t);
        TenR s(2, n, n);
        for (NumN i1 = 1; i1 <= n; ++i1) {
            for (NumN i2 = 1; i2 <= n; ++i2) {
                s(1, i1, i2) = t(i1);
                s(2, i1, i2) = t(i2);
            }
        }

        TenR sub;
        math21_operator_share_tensor_rows(facez0, 0, 2, sub);
        sub = s;
        TenR v;
        math21_operator_share_tensor_rows(facez0, 2, 1, v);
        v = 0;
        math21_operator_share_tensor_rows(facez1, 0, 2, sub);
        sub = s;
        math21_operator_share_tensor_rows(facez1, 2, 1, v);
        v = 1;

        facey0 = facez0;
        facey1 = facez1;
        math21_op_tensor_swap_rows(facey0, 2, 3, 1);
        math21_op_tensor_swap_rows(facey1, 2, 3, 1);
        facex0 = facey0;
        facex1 = facey1;
        math21_op_tensor_swap_rows(facex0, 1, 2, 1);
        math21_op_tensor_swap_rows(facex1, 1, 2, 1);

        math21_op_tensor_move_axis(cube, data, 2, 4);

        math21_operator_reshape_to_matrix(data, data.size() / data.dim(data.dims()), data.dim(data.dims()));
    }

    // log_proportions: n_component
    // means: n_component * n_feature
    // covs_sqrt: n_component * n_feature * n_feature
    void math21_geometry_generate_gmm(MatR &data,
                                      const VecR &log_proportions, const MatR &means,
                                      const TenR &covs_sqrt, NumB useColor) {
        MATH21_ASSERT(means.dim(2) == 2, "only support 2d data!");
        NumN n_component, n_feature;
        n_component = means.dim(1);
        n_feature = means.dim(2);
        NumN n_points = 100;
        data.setSize(n_component, n_points, n_feature);
        TenR colors; // colors: n_component * n_points * 3
        if (useColor) {
            math21_geometry_generator_color(colors, n_component, n_points, 1);
        }
        for (NumN i = 1; i <= n_component; ++i) {
            NumR proportion = xjexp(log_proportions(i)); // (0, 1], can be regarded as alpha
            NumR alpha = xjmin((NumR) 1, 10 * proportion);
            if (useColor) {
                MatR color;
                math21_operator_share_tensor_row_i(colors, i, color);
                math21_op_mul_onto(alpha, color);
            }
            MatR datai;
            math21_operator_share_tensor_row_i(data, i, datai);
            VecR mean;
            math21_operator_share_tensor_row_i(means, i, mean);
            MatR cov_sqrt;
            math21_operator_share_tensor_row_i(covs_sqrt, i, cov_sqrt);
            // covariance matrix is related to a linear transformation. (need proof)
            MatR A;
            math21_op_mul(2, cov_sqrt, A);
            math21_geometry_generate_ellipse(datai, A, mean, n_points);
        }
        if (useColor) {
            MatR data2;
            math21_op_tensor_concatenate(data, colors, data2, 3);
            data = data2;
        }
        math21_operator_reshape_to_matrix(data, data.size() / data.dim(data.dims()), data.dim(data.dims()));
    }

}