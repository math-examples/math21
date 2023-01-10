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
#include "geometryTrans.h"
#include "transOpt.h"
#include "../op/files.h"
#include "../opt/files.h"

namespace math21 {
    class ProjectionRefineCallback : public OptCallbackLevMarq {
    private:
        MatR x, y;
        NumN type;
        NumN dof;
    public:
        ProjectionRefineCallback(const MatR &x0, const MatR &y0, NumN type_) {
            math21_operator_share_copy(x0, x);
            math21_operator_share_copy(y0, y);
            NumN b = x.nrows();
            NumN n = x.ncols() + 1;
            NumN m = y.ncols() + 1;
            MATH21_ASSERT(b > 0 && n > 1 && m > 1 && b == y.nrows());
            type = type_;

            if (type == m21_flag_projection_affine) {
                dof = (m - 1) * n;
            } else {
                dof = m * n - 1;
            }
        }

        NumB compute(const VecR &theta, VecR &value_raw, MatR *pJ) override {
            NumN b = x.nrows();
            NumN m = y.ncols() + 1;
            NumN n = x.ncols() + 1;
            MATH21_ASSERT(theta.size() == dof);

            MatR H;
            H.setSize(m, n);
            H = 0;
            H(m, n) = 1;
            math21_op_vector_set_by_vector(theta, H);

            MatR value;
            value_raw.setSize(b * (m - 1));
            math21_operator_share_to_matrix(value_raw, value, b, (m - 1));

            MatR y_est;
            NumB flag = math21_geometry_project_non_homogeneous_with_jacobian(H, x, y_est, 1, pJ, type);
            math21_op_subtract(y_est, y, value);
            return flag;
        }
    };

    NumB math21_geometry_refine_transformation(const MatR &X, const MatR &Y, MatR &H, NumN type) {
        NumN m = H.nrows();
        NumN n = H.ncols();
        NumN dof;
        if (type == m21_flag_projection_affine) {
            dof = (m - 1) * n;
        } else {
            MATH21_ASSERT(type == m21_flag_projection_projective);
            dof = m * n - 1;
        }
        VecR h;
        math21_operator_share_vector_part_using_from_to(H, h, 1, dof);
        OptParasLevMarq parasLevMarq;
        ProjectionRefineCallback cb(X, Y, type);
        OptAlg optAlg;
        optAlg.set(OptUpdateType_LevMar, &parasLevMarq, &cb, 0);
        NumB flag = optAlg.run(h);
        return flag;
    }

    NumB math21_geometry_refine_affine(const MatR &X, const MatR &Y, MatR &H) {
        return math21_geometry_refine_transformation(X, Y, H, m21_flag_projection_affine);
    }

    NumB math21_geometry_refine_projectivity(const MatR &X, const MatR &Y, MatR &H) {
        return math21_geometry_refine_transformation(X, Y, H, m21_flag_projection_projective);
    }

    NumB math21_geometry_find_affine(const MatR &X, const MatR &Y, MatR &H) {
        NumB flag = math21_geometry_cal_affine_non_homogeneous(X, Y, H);
        if (!flag)return 0;
        flag = math21_geometry_refine_affine(X, Y, H);
        return flag;
    }

    NumB math21_geometry_find_projectivity(const MatR &X, const MatR &Y, MatR &H) {
        NumB flag = math21_geometry_cal_projectivity(X, Y, H);
        if (!flag)return 0;
        flag = math21_geometry_refine_projectivity(X, Y, H);
        return flag;
    }
}