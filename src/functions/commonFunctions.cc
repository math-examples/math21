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

#include "commonFunctions.h"

namespace math21 {

    LogOperator::LogOperator() {
    }

    LogOperator::~LogOperator() {}

    void LogOperator::valueAt(const TenR &x, TenR &y) {
        math21_operator_log(x, y);
    }

    ExpOperator::ExpOperator() {
    }

    ExpOperator::~ExpOperator() {}

    void ExpOperator::valueAt(const TenR &x, TenR &y) {
        math21_operator_exp(x, y);
    }


    SoftargmaxOperator::SoftargmaxOperator() {
    }

    SoftargmaxOperator::~SoftargmaxOperator() {}

    void SoftargmaxOperator::valueAt(const TenR &x, TenR &y) {
        ExpOperator::valueAt(x, y);
        NumR sum = math21_operator_norm(y, 1);
        MATH21_ASSERT(sum > 0,
                      "norm as denominator which should be larger than 0, but it is "
                              << sum << "\n"
                              << "\t" << x.log("x") << "\n"
                              << "\t" << y.log("y") << "\n"
        )
//            math21_operator_clip_not_less_than_eps(sum);
        math21_operator_linear_to(1 / sum, y);
        MATH21_ASSERT_CHECK_VALUE_TMP(math21_operator_is_not_less(y, 0))
    }


    PureLinearOperator::PureLinearOperator() {
    }

    PureLinearOperator::~PureLinearOperator() {}


    void PureLinearOperator::derivativeValueAt(const VecR &x, MatR &dH) {
        if (dH.nrows() != x.size() || dH.ncols() != x.size()) {
            dH.setSize(x.size(), x.size());
        }
        math21_operator_mat_eye(dH);
    }

    void PureLinearOperator::derivativeValueUsingf(const VecR &y, MatR &dH) {
        if (dH.isSameSize(y.size(), y.size()) == 0) {
            dH.setSize(y.size(), y.size());
        }
        math21_operator_mat_eye(dH);
    }


    CostFunctional_class::CostFunctional_class() {}

    CostFunctional_class::~CostFunctional_class() {}


    CostFunctional_mse_se_class::CostFunctional_mse_se_class() {
        isSet = 0;
    }

    CostFunctional_mse_se_class::~CostFunctional_mse_se_class() {}

    NumR CostFunctional_mse_se_class::valueAt(const VecR &x) {
        MATH21_ASSERT(isSet, "Please set parameters first!")
        NumR fx;
        VecR tmp;
        math21_operator_vec_linear(1, t, -1, x, tmp);
        fx = math21_operator_InnerProduct(1, tmp, tmp);
        return fx;
    }

    NumN CostFunctional_mse_se_class::getXDim() {
        MATH21_ASSERT(isSet, "Please set parameters first!")
        return t.size();
    }

    const VecR &CostFunctional_mse_se_class::derivativeValueAt(const VecR &x) {
        math21_operator_vec_linear(2, x, -2, t, g);
        return g;
    }

    void CostFunctional_mse_se_class::setParas(const VecR &_t) {
        if (t.isSameSize(_t.size()) == 0) {
            t.setSize(_t.size());
        }
        t.assign(_t);
        isSet = 1;
    }

    void CostFunctional_mse_se_class::clear() {
        isSet = 0;
    }


    CostFunctional_nll_CrossEntroy_softmax_class::CostFunctional_nll_CrossEntroy_softmax_class() {
        isSet = 0;
    }

    CostFunctional_nll_CrossEntroy_softmax_class::~CostFunctional_nll_CrossEntroy_softmax_class() {}

    NumR CostFunctional_nll_CrossEntroy_softmax_class::valueAt(const TenR &x) {
        MATH21_ASSERT(isSet, "Please set parameters first!")
        NumR fx;
        SoftargmaxOperator::valueAt(x, y_softmax);
        math21_operator_clip_not_less_than_eps(y_softmax);
        LogOperator::valueAt(y_softmax, tmp);
        MATH21_ASSERT_CHECK_VALUE_TMP(math21_operator_is_not_larger(tmp, 0))
        fx = math21_operator_InnerProduct(-1, tmp, t);
        return fx;
    }

    void CostFunctional_nll_CrossEntroy_softmax_class::setParas(const TenR &_t) {
        t.setSize(_t.shape());
        t.assign(_t);
        isSet = 1;
    }

    void CostFunctional_nll_CrossEntroy_softmax_class::clear() {
        isSet = 0;
    }

    const TenR &CostFunctional_nll_CrossEntroy_softmax_class::derivativeValueAt(const TenR &x) {
        valueAt(x);
        math21_operator_linear(1, y_softmax, -1, t, g);
        return g;
    }


    // Todo: maybe getXshape
    NumN CostFunctional_nll_CrossEntroy_softmax_class::getXDim() {
        MATH21_ASSERT(isSet, "Please set parameters first!")
        return t.size();
    }

    const ArrayN &CostFunctional_nll_CrossEntroy_softmax_class::get_x_shape() const {
        return t.shape();
    }

    dummy_class::dummy_class() {
    }

    dummy_class::~dummy_class() {}


    ////####################################


    Function_linear::Function_linear() {
    }

    Function_linear::~Function_linear() {}

    NumR Function_linear::valueAt(const NumR &x) {
        return x;
    }

    NumR Function_linear::derivativeValueAt(const NumR &x) {
        return 1;
    }

    NumR Function_linear::derivativeValue_using_y(const NumR &y) {
        return 1;
    }


    Function_tanh::Function_tanh() {
    }

    Function_tanh::~Function_tanh() {}

    NumR Function_tanh::valueAt(const NumR &x) {
        NumR y;
        NumR a, b;
        a = xjexp(x);
        b = xjexp(-x);
        y = (a - b) / (a + b);
        return y;
    }

    NumR Function_tanh::derivativeValueAt(const NumR &x) {
        NumR y;
        y = valueAt(x);
        return 1 - y * y;
    }

    NumR Function_tanh::derivativeValue_using_y(const NumR &y) {
        return 1 - y * y;
    }


    Function_LogSigmoid::Function_LogSigmoid() {
    }

    Function_LogSigmoid::~Function_LogSigmoid() {}

    NumR Function_LogSigmoid::valueAt(const NumR &x) {
        return 1.0 / (1 + xjexp(-x));
    }

    NumR Function_LogSigmoid::derivativeValueAt(const NumR &x) {
        NumR y = valueAt(x);
        return (1 - y) * y;
    }

    NumR Function_LogSigmoid::derivativeValue_using_y(const NumR &y) {
        return (1 - y) * y;
    }


    Function_LeakyReLU::Function_LeakyReLU() {
    }

    Function_LeakyReLU::~Function_LeakyReLU() {}

    NumR Function_LeakyReLU::valueAt(const NumR &x) {
        if (x >= 0) {
            return x;
        } else {
            return 0.01 * x;
        }
    }

    NumR Function_LeakyReLU::derivativeValueAt(const NumR &x) {
        if (x >= 0) {
            return 1;
        } else {
            return 0.01;
        }
    }

    NumR Function_LeakyReLU::derivativeValue_using_y(const NumR &y) {
        if (y >= 0) {
            return 1;
        } else {
            return 0.01;
        }
    }


    ////####################################

    LogSigmoid::LogSigmoid() {
    }

    LogSigmoid::~LogSigmoid() {}

    NumR LogSigmoid::valueAt(const NumR &x) {
        return 1.0 / (1 + xjexp(-x));
    }

    NumR LogSigmoid::derivativeValueAt(const NumR &x) {
        NumR y = valueAt(x);
        return (1 - y) * y;
    }

    NumR LogSigmoid::derivativeValueUsingf(const NumR &y) {
        return (1 - y) * y;
    }


    LeakyReLU::LeakyReLU() {
    }

    LeakyReLU::~LeakyReLU() {}

    NumR LeakyReLU::valueAt(const NumR &x) {
        if (x >= 0) {
            return x;
        } else {
            return 0.01 * x;
        }
    }

    NumR LeakyReLU::derivativeValueAt(const NumR &x) {
        if (x >= 0) {
            return 1;
        } else {
            return 0.01;
        }
    }

    NumR LeakyReLU::derivativeValueUsingf(const NumR &y) {
        if (y >= 0) {
            return 1;
        } else {
            return 0.01;
        }
    }

    LogSigmoidOperator::LogSigmoidOperator() {
    }

    LogSigmoidOperator::~LogSigmoidOperator() {}

    void LogSigmoidOperator::valueAt(const VecR &x, VecR &y) {
        LogSigmoid ls;
        y.setSize(x.size());
        for (NumN i = 1; i <= x.size(); i++) {
            y(i) = ls.valueAt(x(i));
        }
    }

    void LogSigmoidOperator::derivativeValueAt(const VecR &x, MatR &dH) {
        dH.setSize(x.size(), x.size());
        dH = 0;
        LogSigmoid ls;
        for (NumN i = 1; i <= x.size(); i++) {
            dH(i, i) = ls.derivativeValueAt(x(i));
        }
    }

    void LogSigmoidOperator::derivativeValueUsing_y(const VecR &y, MatR &dH) {
        dH.setSize(y.size(), y.size());
        dH = 0;
        for (NumN i = 1; i <= y.size(); i++) {
            dH(i, i) = LogSigmoid::derivativeValueUsingf(y(i));
        }
    }

    Operator_tanh::Operator_tanh() {
    }

    Operator_tanh::~Operator_tanh() {}

    void Operator_tanh::valueAt(const VecR &x, VecR &y) {
        y.setSize(x.size());
        for (NumN i = 1; i <= x.size(); i++) {
            y(i) = m21tanh(x(i));
        }
    }

    void Operator_tanh::derivativeValueAt(const VecR &x, MatR &dH) {
        MATH21_ASSERT_NOT_CALL(0, "dummy");
        dH.setSize(x.size(), x.size());
        dH = 0;
    }

    void Operator_tanh::derivativeValueUsing_y(const VecR &y, MatR &dH) {
        dH.setSize(y.size(), y.size());
        dH = 0;
        for (NumN i = 1; i <= y.size(); i++) {
            dH(i, i) = m21tanh_derivativeValueUsing_y(y(i));
        }
    }

    LeakyReLUOperator::LeakyReLUOperator() {
    }

    LeakyReLUOperator::~LeakyReLUOperator() {}

    VecR LeakyReLUOperator::valueAt(const VecR &x) {
        LeakyReLU ls;
        VecR y(x.size());
        for (NumN i = 1; i <= x.size(); i++) {
            y(i) = ls.valueAt(x(i));
        }
        return y;
    }

    void LeakyReLUOperator::derivativeValueAt(const VecR &x, MatR &dH) {
        dH.setSize(x.size(), x.size());
        dH = 0;
        LeakyReLU ls;
        for (NumN i = 1; i <= x.size(); i++) {
            dH(i, i) = ls.derivativeValueAt(x(i));
        }
    }

    void LeakyReLUOperator::derivativeValueUsing_y(const VecR &y, MatR &dH) {
        dH.setSize(y.size(), y.size());
        dH = 0;
        for (NumN i = 1; i <= y.size(); i++) {
            dH(i, i) = LeakyReLU::derivativeValueUsingf(y(i));
        }
    }


    FunctionSineEx1::FunctionSineEx1() {
    }

    FunctionSineEx1::~FunctionSineEx1() {}

    NumR FunctionSineEx1::valueAt(const NumR &x) {
        return 1 + xjsin((XJ_PI / 4.0) * x);
    }

    NumR FunctionSineEx1::derivativeValueAt(const NumR &x) {
        MATH21_ASSERT_NOT_CALL(0, "can't call.");
        return x;
    }
}