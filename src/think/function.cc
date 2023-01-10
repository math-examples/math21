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

#include "../ad/functions_01/op_get_shape.h"
#include "../ad/functions_01/op_share_reshape.h"
#include "../ad/functions_02/op_broadcast_tensor.h"
#include "../algebra/set.h"
#include "../ad/differential.h"
#include "function.h"

namespace math21 {
    namespace ad {
        VarAd::VarAd() : id(0) {

        }

        VarAd::VarAd(NumPtr id) : id(id) {
        }

        NumPtr VarAd::getId() const {
            return id;
        }

        Variable &VarAd::getVariable() {
            return *(Variable *) id;
        }

        const Variable &VarAd::getVariable() const {
            return *(const Variable *) id;
        }

        bool VarAd::operator==(const VarAd &var) const {
            return this->id == var.id;
        }

        bool VarAd::operator!=(const VarAd &var) const {
            return this->id != var.id;
        }

        bool VarAd::operator<(const VarAd &var) const {
            return this->id < var.id;
        }

        bool VarAd::operator>(const VarAd &var) const {
            return this->id > var.id;
        }

        NumB VarAd::isEmpty() const {
            if (id == 0)return 1;
            else return 0;
        }

        NumB VarAd::log(const char *name) const {
            log(std::cout, name);
            return 1;
        }

        NumB VarAd::log(std::ostream &io, const char *name) const {
            getVariable().log(io, name);
            return 1;
        }

        std::ostream &operator<<(std::ostream &out, const VarAd &var) {
            var.log(out);
            return out;
        }

        NumB math21_point_isEqual(const VarAd &x, const VarAd &y, NumR epsilon) {
            MATH21_ASSERT(0);
            if (x == y)return 1;
            else return 0;
        }

        void math21_io_serialize(std::ostream &out, const VarAd &m, SerializeNumInterface &sn) {
            MATH21_ASSERT(0);
        }

        void math21_io_deserialize(std::istream &in, VarAd &m, DeserializeNumInterface &sn) {
            MATH21_ASSERT(0);
        }

        //        const NumB Function::isSetSizeFlag = 0;
//        const NumB Function::isSetSizeFlag = 1;
        NumB Function::isSetSizeFlag = 0;
//        NumB Function::isElementWiseTestFlag = 1;
        NumB Function::isElementWiseTestFlag = 0;

        Function::Function() {
            isElementWiseFlag = 0;
            isGlobalFlag = 0;
        }

        Function::~Function() {
        }

        void Function::cr_jvp(const SetVar &X, VarAd x, VarAd y, VarAd dy, SetVar &Y, VariableMap &data) const {
            MATH21_ASSERT(0)
        }

        VarAd Function::cr_vjp_inner(const SetVar &X, VarAd x, VarAd y, VarAd dy, VariableMap &data) const {
            MATH21_ASSERT(0)
            return VarAd(0);
        }

        VarAd Function::cr_vjp(const SetVar &X, VarAd x, VarAd y, VarAd dy, VariableMap &data) const {
//            static NumN count = 0;
//            m21log(getName(), ++count);
            // todo: remove this
//            op_share_reshape a;
            if (variable_reshape_to_same_vspace_using_variable(y, dy, data) == 1) {
                MATH21_ASSERT(0, ">>>>>this shouldn't be called!");
            }

            VarAd dx = cr_vjp_inner(X, x, y, dy, data);
            // dxi_part can't be global contant, 'cause it will be used by others.
            if (data(dx).getType() == variable_type_constant) {
                if (data(dx).getValue().size() == 1) {
                    MATH21_ASSERT(0, "check here")
                }
            }

            // put here so as to avoid putting everywhere.
            op_get_shape _get_shape;
            Function &f_get_shape = _get_shape;
            VarAd d_x = f_get_shape.evaluate(x, data);
            op_share_reshape _op_share_reshape;
            Function &f_op_share_reshape = _op_share_reshape;
            dx = f_op_share_reshape.evaluate(dx, d_x, data);
            return dx;
        }

        void Function::cr_jmp(const SetVar &X, VarAd x, VarAd y, VarAd dy, SetVar &Y, VariableMap &data) const {
            MATH21_ASSERT(0)
        }

        void Function::cr_mjp(const SetVar &X, VarAd x, VarAd y, VarAd dy, SetVar &Y, VariableMap &data) const {
            MATH21_ASSERT(0)
        }

        VarAd Function::evaluate(VarAd x, VariableMap &data) {
            SetVar X;
            X.add(x);
            return evaluate(X, data);
        }

        VarAd Function::evaluate(VarAd x1, VarAd x2, VariableMap &data) {
            SetVar X;
            X.add(x1);
            X.add(x2);
            return evaluate(X, data);
        }

        VarAd Function::evaluate(VarAd x1, VarAd x2, VarAd x3, VariableMap &data) {
            SetVar X;
            X.add(x1);
            X.add(x2);
            X.add(x3);
            return evaluate(X, data);
        }

        VarAd Function::evaluate(const SetVar &X, VariableMap &data) {
            VarAd y = data.createV(math21_string_concatenate(getName(), "(x)").c_str());
            variable_set_device_type_using_variable(X(1), y, data);

            data.at(y).setf(this);
            data.at(y).setX(X);
            for (NumN i = 1; i <= X.size(); ++i) {
                data.at(X(i)).addy(y);
            }
            SetVar Y;
            Y.add(y);
            fv(X, Y, data);
            return y;
        }

        void Function::df(const SetVar &X, VarAd x, VarAd y, VarAd &dydx, VariableMap &data) const {
            MATH21_ASSERT(0)
        }

        void Function::df_dbr(const SetVar &X, VarAd x, VarAd y, VarAd &dydx, VariableMap &data) const {
            MATH21_ASSERT(0)
        }

        void Function::cr(const SetVar &X, VarAd x, VarAd y, VarAd dy, SetVar &Y, VariableMap &data) const {
            MATH21_ASSERT(0)
        }

        void Function::backward(const SetVar &X, VarAd x, VarAd y, VarAd dy, SetVar &Y, VariableMap &data) const {
            MATH21_ASSERT(0)
        }

        void Function::f(const SetVar &X, SetVar &Y, VariableMap &data) {
            MATH21_ASSERT(0)
        }

        void Function::fv(const SetVar &X, const SetVar &Y, VariableMap &data) const {
            MATH21_ASSERT(0)
        }

        void Function::f(VarAd x, VarAd &y, VariableMap &data) {
            SetVar X;
            X.add(x);
            SetVar Y;
            f(X, Y, data);
            y = Y(1);
        }

        void Function::f(VarAd x1, VarAd x2, VarAd &y, VariableMap &data) {
            SetVar X;
            X.add(x1);
            X.add(x2);
            SetVar Y;
            f(X, Y, data);
            y = Y(1);
        }

        void Function::f(VarAd x1, VarAd x2, VarAd x3, VarAd &y, VariableMap &data) {
            SetVar X;
            X.add(x1);
            X.add(x2);
            X.add(x3);
            SetVar Y;
            f(X, Y, data);
            y = Y(1);
        }

        void Function::compute(VarAd x, VarAd y, VariableMap &data, Derivative &derivative) {
            SetVar X;
            X.add(x);
            SetVar Y;
            Y.add(y);
            compute(X, Y, data, derivative);
        }

        void Function::compute(VarAd x1, VarAd x2, VarAd y, VariableMap &data, Derivative &derivative) {
            SetVar X;
            X.add(x1);
            X.add(x2);
            SetVar Y;
            Y.add(y);
            compute(X, Y, data, derivative);
        }

        void Function::compute(VarAd x1, VarAd x2, VarAd x3, VarAd y, VariableMap &data, Derivative &derivative) {
            SetVar X;
            X.add(x1);
            X.add(x2);
            X.add(x3);
            SetVar Y;
            Y.add(y);
            compute(X, Y, data, derivative);
        }

        void Function::compute(const SetVar &X, const SetVar &Y, VariableMap &data, Derivative &derivative) {
            fv(X, Y, data);
        }

        void Function::forward(VarAd x, VarAd &y, VariableMap &data) {
            SetVar X;
            X.add(x);
            SetVar Y;
            forward(X, Y, data);
            y = Y(1);
        }

        void Function::forward(VarAd x1, VarAd x2, VarAd &y, VariableMap &data) {
            SetVar X;
            X.add(x1);
            X.add(x2);
            SetVar Y;
            forward(X, Y, data);
            y = Y(1);
        }

        void Function::forward(VarAd x1, VarAd x2, VarAd x3, VarAd &y, VariableMap &data) {
            SetVar X;
            X.add(x1);
            X.add(x2);
            X.add(x3);
            SetVar Y;
            forward(X, Y, data);
            y = Y(1);
        }

        void Function::forward(const SetVar &X, SetVar &Y, VariableMap &data) {
            f(X, Y, data);
            Derivative derivative(data);
            compute(X, Y, data, derivative);
        }

        NumB Function::isSetSize() {
            return isSetSizeFlag;
        }

        void Function::setSetSizeFlag(NumB flag) {
            isSetSizeFlag = flag;
        }

        NumB Function::isElementWiseTest() {
            return isElementWiseTestFlag;
        }

        NumB Function::isElementWise() const {
            return isElementWiseFlag;
        }

        void Function::setElementWiseFlag(NumB flag) {
            isElementWiseFlag = flag;
        }

        NumB Function::isGlobal() const {
            return isGlobalFlag;
        }

        void Function::setGlobalFlag(NumB flag) {
            isGlobalFlag = flag;
        }

        // see TensorBroadcast error?
        // see np.broadcast_arrays
        void Function::broadcast_tensors(const SetVar &X, SetVar &Y, VariableMap &data) {
            math21_tool_assert(X.size() > 0);
            Seqce<VecN> shapes(X.size());
            for (NumN i = 1; i <= X.size(); ++i) {
                VarAd x = X(i);
                const auto &x_value = data(x).getValue();
                VecN d;
                shapes.at(i) = x_value.shape(d);
            }
            VecN d;
            NumB flag = math21_broadcast_is_compatible_in_ele_op(shapes, d);
            MATH21_ASSERT(flag, "shape not compatible when broadcasting\n"
                    << X.log("X") << shapes.log("shapes") << data.log("data"));

            VarAd k = data.createC("shape");
            data.setValue(k, 1);
            data.at(k).getValue() = d;
            Y.clear();
            for (NumN i = 1; i <= X.size(); ++i) {
                VarAd x = X(i);
                VarAd x_new = x;
                const auto &x_value = data(x).getValue();
                // todo: not broadcast same shape tensors
                if (!x_value.isSameSize(d)) {
                    op_broadcast_tensor bc;
                    Function &function = bc;
                    x_new = function.evaluate(x, k, data);
                }
                Y.add(x_new);
            }
        }

        // todo: remove this, and use broadcast_tensors instead.
        void Function::broadcast_num_to_vec(const SetVar &X, SetVar &Y, VariableMap &data) {
            broadcast_tensors(X, Y, data);
        }

        void Function::variable_set_device_type_using_variable(VarAd x, VarAd y, VariableMap &data) {
            math21_tool_assert(!y.isEmpty());
            const auto &x_value = data(x).getValue();
            auto &y_value = data.at(y).getValue();
            y_value.setDeviceType(x_value.getDeviceType());
        }

        void Function::variable_set_device_type_gpu(VarAd y, VariableMap &data) {
            math21_tool_assert(!y.isEmpty());
            auto &y_value = data.at(y).getValue();
            y_value.setDeviceType(m21_device_type_gpu);
        }

        NumN Function::variable_get_device_type(VarAd x, VariableMap &data) {
            return data(x).getValue().getDeviceType();
        }

        NumB Function::variable_is_cpu(VarAd x, VariableMap &data) {
            return data(x).getValue().is_cpu();
        }

        NumB Function::variable_setSize_to_same_vspace_using_variable(VarAd x, VarAd y, VariableMap &data) {
            if (y.isEmpty()) {
                return 0;
            }
            const auto &x_value = data.at(x).getValue();
            auto &y_value = data.at(y).getValue();
            return math21_operator_tensor_setSize_to_same_vspace_using_value(x_value, y_value);
        }

        NumB Function::variable_setSize_to_same_vspace_using_shape(const VecN &d, VarAd y, VariableMap &data) {
            if (y.isEmpty()) {
                return 0;
            }
            auto &y_value = data.at(y).getValue();
            return math21_operator_tensor_setSize_to_same_vspace_using_shape(d, y_value);
        }

        NumB Function::variable_reshape_to_same_vspace_using_variable(VarAd x, VarAd y, VariableMap &data) {
            if (y.isEmpty()) {
                return 0;
            }
            const auto &x_value = data.at(x).getValue();
            auto &y_value = data.at(y).getValue();
            return math21_operator_tensor_reshape_to_same_vspace_using_value(x_value, y_value);
        }

        NumB Function::variable_reshape_to_same_vspace_using_shape(const VecN &d, VarAd y, VariableMap &data) {
            if (y.isEmpty()) {
                return 0;
            }
            auto &y_value = data.at(y).getValue();
            return math21_operator_tensor_reshape_to_same_vspace_using_shape(d, y_value);
        }
    }
}