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
    namespace ad {
        struct VariableMap;

        class Derivative;

        class VarAd {
        private:
            NumPtr id;
        public:
            VarAd();

            explicit VarAd(NumPtr id);

            NumPtr getId() const;

            Variable &getVariable();

            const Variable &getVariable() const;

            bool operator==(const VarAd &var) const;

            bool operator!=(const VarAd &var) const;

            bool operator<(const VarAd &var) const;

            bool operator>(const VarAd &var) const;

            NumB isEmpty() const;

            NumB log(const char *name = 0) const;

            NumB log(std::ostream &io, const char *name = 0) const;

        };

        std::ostream &operator<<(std::ostream &out, const VarAd &var);

        NumB math21_point_isEqual(const VarAd &x, const VarAd &y, NumR epsilon);

        void math21_io_serialize(std::ostream &out, const VarAd &m, SerializeNumInterface &sn);

        void math21_io_deserialize(std::istream &in, VarAd &m, DeserializeNumInterface &sn);

        typedef Map_<VarAd, NumN> MapVarN;
        typedef Map_<VarAd, VarAd> MapVarVar;
        typedef Tensor<VarAd> TenVar;
        typedef Set_<VarAd> SetVar;
        typedef Seqce<VarAd> SeqVar;

        struct Function {
        private:
//            static const NumB isSetSizeFlag;
            // deprecate
            static NumB isSetSizeFlag;
            // type 1: element-wise
            // type 2: not element-wise
            static NumB isElementWiseTestFlag;
            NumB isElementWiseFlag;
            NumB isGlobalFlag;

            virtual void cr_jvp(const SetVar &X, VarAd x, VarAd y, VarAd dy, SetVar &Y, VariableMap &data) const;

            // We decide to reshape dx at end.
            // todo: reshape dx at end, or reshape dy at start.
            // todo: if we reshape dx at end, y can be left out.
            // see autograd/numpy/numpy_vjps.py
            // 'return dx = 0' means differential being zero.
            // chain rule in define-by-run mode
            virtual VarAd cr_vjp_inner(const SetVar &X, VarAd x, VarAd y, VarAd dy, VariableMap &data) const;

            virtual void cr_jmp(const SetVar &X, VarAd x, VarAd y, VarAd dy, SetVar &Y, VariableMap &data) const;

            virtual void cr_mjp(const SetVar &X, VarAd x, VarAd y, VarAd dy, SetVar &Y, VariableMap &data) const;

        public:
            Function();

            virtual ~Function();

            // todo: maybe remove virtual
            virtual VarAd cr_vjp(const SetVar &X, VarAd x, VarAd y, VarAd dy, VariableMap &data) const;

            VarAd evaluate(VarAd x, VariableMap &data);

            VarAd evaluate(VarAd x1, VarAd x2, VariableMap &data);

            VarAd evaluate(VarAd x1, VarAd x2, VarAd x3, VariableMap &data);

            // todo: implement here instead of in every function.
            // define-by-run <=> f + fv
            virtual VarAd evaluate(const SetVar &X, VariableMap &data);

            void f(VarAd x, VarAd &y, VariableMap &data);

            void f(VarAd x1, VarAd x2, VarAd &y, VariableMap &data);

            void f(VarAd x1, VarAd x2, VarAd x3, VarAd &y, VariableMap &data);

            void compute(VarAd x, VarAd y, VariableMap &data, Derivative &derivative);

            void compute(VarAd x1, VarAd x2, VarAd y, VariableMap &data, Derivative &derivative);

            void compute(VarAd x1, VarAd x2, VarAd x3, VarAd y, VariableMap &data, Derivative &derivative);

            void forward(VarAd x, VarAd &y, VariableMap &data);

            void forward(VarAd x1, VarAd x2, VarAd &y, VariableMap &data);

            void forward(VarAd x1, VarAd x2, VarAd x3, VarAd &y, VariableMap &data);

            // may deprecate, use Derivative.cd() instead.
            virtual void df(const SetVar &X, VarAd x, VarAd y, VarAd &dydx, VariableMap &data) const;

            // df in define-by-run mode
            virtual void df_dbr(const SetVar &X, VarAd x, VarAd y, VarAd &dydx, VariableMap &data) const;

            // chain rule
            virtual void cr(const SetVar &X, VarAd x, VarAd y, VarAd dy, SetVar &Y, VariableMap &data) const;

            // deprecate, use cr_vjp
            // chain rule in define-by-run mode
            virtual void
            backward(const SetVar &X, VarAd x, VarAd y, VarAd dy, SetVar &output, VariableMap &data) const;

            // define function graph
            virtual void f(const SetVar &X, SetVar &Y, VariableMap &data);

            // run function graph
            // evaluate function value
            virtual void fv(const SetVar &X, const SetVar &Y, VariableMap &data) const;

            // run in partial define-by-run mode
            // evaluate function value, modify graph if necessary.
            virtual void compute(const SetVar &X, const SetVar &Y, VariableMap &data, Derivative &derivative);

            // define-by-run <=> f + fv
            virtual void forward(const SetVar &X, SetVar &Y, VariableMap &data);

            // deprecate
            virtual void setSize(const SetVar &X, const SetVar &Y, VariableMap &data) const {}

            virtual Function *clone() const = 0;

            virtual const char *getName() const = 0;

            // deprecated, use compute instead.
            static NumB isSetSize();

            static void setSetSizeFlag(NumB flag);

            // test
            static NumB isElementWiseTest();

            NumB isElementWise() const;

            void setElementWiseFlag(NumB flag);

            NumB isGlobal() const;

            void setGlobalFlag(NumB flag);

            static void broadcast_tensors(const SetVar &X, SetVar &Y, VariableMap &data);

            static void broadcast_num_to_vec(const SetVar &X, SetVar &Y, VariableMap &data);

            static void variable_set_device_type_using_variable(VarAd x, VarAd y, VariableMap &data);

            static void variable_set_device_type_gpu(VarAd y, VariableMap &data);

            static NumN variable_get_device_type(VarAd x, VariableMap &data);

            static NumB variable_is_cpu(VarAd x, VariableMap &data);

            static NumB variable_setSize_to_same_vspace_using_variable(VarAd x, VarAd y, VariableMap &data);

            static NumB variable_reshape_to_same_vspace_using_variable(VarAd x, VarAd y, VariableMap &data);

            static NumB variable_setSize_to_same_vspace_using_shape(const VecN &d, VarAd y, VariableMap &data);

            static NumB variable_reshape_to_same_vspace_using_shape(const VecN &d, VarAd y, VariableMap &data);
        };
    }
}