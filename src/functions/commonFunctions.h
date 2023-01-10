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

namespace math21 {

    class LogOperator : public think::Operator {
    private:
    public:
        LogOperator();

        virtual ~LogOperator();

        static void valueAt(const TenR &x, TenR &y);
    };

    class ExpOperator : public think::Operator {
    private:
    public:
        ExpOperator();

        virtual ~ExpOperator();

        static void valueAt(const TenR &x, TenR &y);
    };

    class SoftargmaxOperator : public think::Operator {
    public:
        SoftargmaxOperator();

        virtual ~SoftargmaxOperator();

        static void valueAt(const TenR &x, TenR &y);
    };

    class PureLinearOperator : public think::Operator {
    public:
        PureLinearOperator();

        virtual ~PureLinearOperator();

        void derivativeValueAt(const VecR &x, MatR &dH);

        static void derivativeValueUsingf(const VecR &y, MatR &dH);
    };

    class CostFunctional_class : public Functional {
    public:
        CostFunctional_class();

        virtual ~CostFunctional_class();

        virtual void setParas(const VecR &_t) = 0;


        virtual void clear() = 0;
    };

    //Todo: add tensor support.
    class CostFunctional_mse_se_class : public CostFunctional_class {
    private:
        NumB isSet;
        VecR t;
        VecR g;
    public:
        CostFunctional_mse_se_class();

        virtual ~CostFunctional_mse_se_class();

        NumR valueAt(const VecR &x) override;

        NumN getXDim() override;

        const VecR &derivativeValueAt(const VecR &x) override;

        void setParas(const VecR &_t) override;

        void clear() override;
    };

    class CostFunctional_nll_CrossEntroy_softmax_class : public CostFunctional_class {
    private:
        NumB isSet;
        TenR t;
        TenR g;
        TenR y_softmax;

        VecR tmp;
    public:
        CostFunctional_nll_CrossEntroy_softmax_class();

        virtual ~CostFunctional_nll_CrossEntroy_softmax_class();

        NumR valueAt(const TenR &x) override;

        void setParas(const TenR &_t) override;

        void clear() override;

        const TenR &derivativeValueAt(const TenR &x) override;


        // Todo: maybe getXshape
        NumN getXDim() override;

        const ArrayN &get_x_shape() const override;

    };

    class dummy_class {
    public:
        dummy_class();

        virtual ~dummy_class();
    };

    ////####################################

    class Function_linear : public Function {
    private:
    public:
        Function_linear();

        virtual ~Function_linear();

        NumR valueAt(const NumR &x) override;

        NumR derivativeValueAt(const NumR &x) override;

        NumR derivativeValue_using_y(const NumR &y) override;
    };

    class Function_tanh : public Function {
    private:
    public:
        Function_tanh();

        virtual ~Function_tanh();

        NumR valueAt(const NumR &x) override;

        NumR derivativeValueAt(const NumR &x) override;

        NumR derivativeValue_using_y(const NumR &y) override;
    };

    class Function_LogSigmoid : public Function {
    private:
    public:
        Function_LogSigmoid();

        virtual ~Function_LogSigmoid();

        NumR valueAt(const NumR &x) override;

        NumR derivativeValueAt(const NumR &x) override;

        NumR derivativeValue_using_y(const NumR &y) override;
    };

    class Function_LeakyReLU : public Function {
    private:
    public:
        Function_LeakyReLU();

        virtual ~Function_LeakyReLU();

        NumR valueAt(const NumR &x) override;

        NumR derivativeValueAt(const NumR &x) override;

        NumR derivativeValue_using_y(const NumR &y) override;
    };

    ////####################################

    ////!!!! deprecated
    class LogSigmoid : public Function {
    private:
    public:
        LogSigmoid();

        virtual ~LogSigmoid();

        NumR valueAt(const NumR &x) override;

        NumR derivativeValueAt(const NumR &x) override;

        static NumR derivativeValueUsingf(const NumR &y);
    };

    inline NumR m21tanh(const NumR &x) {
        NumR y;
        NumR a, b;
        a = xjexp(x);
        b = xjexp(-x);
        y = (a - b) / (a + b);
        return y;
    }

    inline NumR m21tanh_derivativeValueUsing_y(const NumR &y) {
        return 1 - y * y;
    }

    ////!!!! deprecated
    class LeakyReLU : public Function {
    private:
    public:
        LeakyReLU();

        virtual ~LeakyReLU();

        NumR valueAt(const NumR &x) override;

        NumR derivativeValueAt(const NumR &x) override;

        static NumR derivativeValueUsingf(const NumR &y);
    };

    class LogSigmoidOperator : public think::Operator {
    private:
    public:
        LogSigmoidOperator();

        virtual ~LogSigmoidOperator();

        void valueAt(const VecR &x, VecR &y);

        void derivativeValueAt(const VecR &x, MatR &dH);

        static void derivativeValueUsing_y(const VecR &y, MatR &dH);
    };

    class Operator_tanh : public think::Operator {
    private:
    public:
        Operator_tanh();

        virtual ~Operator_tanh();

        void valueAt(const VecR &x, VecR &y);

        void derivativeValueAt(const VecR &x, MatR &dH);

        static void derivativeValueUsing_y(const VecR &y, MatR &dH);
    };

    class LeakyReLUOperator : public think::Operator {
    public:
        LeakyReLUOperator();

        virtual ~LeakyReLUOperator();

        VecR valueAt(const VecR &x);

        void derivativeValueAt(const VecR &x, MatR &dH);

        static void derivativeValueUsing_y(const VecR &y, MatR &dH);
    };

    class FunctionSineEx1 : public Function {
    private:
    public:
        FunctionSineEx1();

        virtual ~FunctionSineEx1();

        NumR valueAt(const NumR &x) override;

        NumR derivativeValueAt(const NumR &x) override;
    };
}