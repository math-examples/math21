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

#include "../ad/functions_01/ad_global.h"
#include "Variable.h"
#include "VariableMap.h"

namespace math21 {
    namespace ad {
        VarAd VariableMap::_createSharedC(NumR x, const char *name) {
            varData.push(Variable::create());
            Variable &vk = varData.peek().getVariable();
            vk.setName(name);
            vk.setType(variable_type_constant);
            vk.setf(ad_global_get_op_num_constant());
            vk.getValue().setSize(1);
            vk.getValue() = x;

            VarAd sharedId = varData.peek();
            return sharedId;
        }

        void VariableMap::_createSomeSharedC() {
            constant_0 = _createSharedC(0, "0");
            constant_1 = _createSharedC(1, "1");
            constant_m1 = _createSharedC(-1, "-1");
        }

        VarAd VariableMap::_createV(const char *name, NumB isShared, VarAd sharedId) {
            VarAd id;
            if (isShared) {
                id = sharedId;
            } else {
                varData.push(Variable::create());

                Variable &x = varData.peek().getVariable();
                x.setName(name);

                id = varData.peek();
            }
            V.add(id);
//            if (id.getVariable().getId() == math21_global_ad_debug_var_id()) {
            if (id.getId() == math21_global_ad_debug_var_id()) {
                MATH21_ASSERT(0);
            }
//            m21log("variable id", id);
            return id;
        }

        void VariableMap::clear() {
            // It had better be called at last, but can still put here.
            ad_global_destroy();

            NumN n = varData.size();
            for (NumN i = 1; i <= n; ++i) {
                Variable::destroy(varData.pop());
            }

            V.clear();

            constant_0 = VarAd(0);
            constant_1 = VarAd(0);
            constant_m1 = VarAd(0);

            V_backup.clear();
            v_size_backup = 0;
        }

        void VariableMap::init() {
            ad_global_create(*this);
            _createSomeSharedC();
            v_size_backup = 0;
        }

        // todo
        void VariableMap::backup() {
            return;
            MATH21_ASSERT(!Variable::isHavingY(), "not implement")
            // no need to backup V when cd_inc
            // Because the time cost is little, we don't optimize it.
            if (math21_global_ad_log_time()) {
                MATH21_PRINT_TIME_ELAPSED(V_backup.set(V));
            } else {
                V_backup.set(V);
            }
            v_size_backup = varData.size();
        }

        // todo
        void VariableMap::restore() {
//            m21log("dataSize", varData.size());
//            m21log("VSize", V.size());
            return;
            if (isEmpty()) {
                return;
            }
            MATH21_ASSERT(!Variable::isHavingY(), "not implement")
            MATH21_ASSERT(V.size() >= V_backup.size());
//            MATH21_ASSERT(map.size() >= v_size_backup);
            V.set(V_backup);
            {
                MATH21_ASSERT(0, "DEBUG, use map instead of v");
//                math21_operator_container_sub_from_start(v, v_size_backup);
//                math21_operator_container_sub_from_start(map, v_size_backup);
            }
        }

        void VariableMap::reset() {
            clear();
            init();
        }

        VarAd VariableMap::get_constant_0() {
            return _createV("", 1, constant_0);
        }

        VarAd VariableMap::get_constant_1() {
            return _createV("", 1, constant_1);
        }

        VarAd VariableMap::get_constant_m1() {
            return _createV("", 1, constant_m1);
        }

        VariableMap::VariableMap(SetVar &V) : V(V) {
            init();
        }

        VariableMap::~VariableMap() {
            clear();
        }

        NumN VariableMap::size() const {
            return varData.size();
        }

        data_structure::Stack<VarAd> &VariableMap::getVarData() {
            return varData;
        }

        NumB VariableMap::isEmpty() const {
            if (size() == 0)return 1;
            else return 0;
        }

        NumB VariableMap::log(const char *s) const {
            varData.log(s);
            return 1;
        }

        NumB VariableMap::log(std::ostream &io, const char *s) const {
            varData.log(io, s);
            return 1;
        }

        // this will make previous references to pointer content invalid.
        // So use references to data.
        VarAd VariableMap::createV(const char *name) {
            VarAd k = _createV(name, 0, VarAd(0));
            Variable &vk = at(k);
            return k;
        }

        // create constant.
        VarAd VariableMap::createC(const char *name) {
            VarAd k = _createV(name, 0, VarAd(0));
            Variable &vk = at(k);
            vk.setType(variable_type_constant);
            return k;
        }

        void VariableMap::setDeviceType(VarAd id, NumN deviceType) {
            Variable &vk = at(id);
            vk.getValue().setDeviceType(deviceType);
        }

        void VariableMap::setValue(VarAd id, NumR x) {
            Variable &vk = at(id);
            vk.getValue().setSize(1);
            vk.getValue() = x;
        }

        Variable &VariableMap::at(VarAd i) {
            return i.getVariable();
        }

        const Variable &VariableMap::operator()(VarAd i) const {
            return i.getVariable();
        }

        SetVar &VariableMap::getV() {
            return V;
        }

        // constant X by variable X
        void setSizeCXUXByX(const SetVar &X, VariableMap &data) {
            VarAd x;
            for (NumN i = 1; i <= X.size(); ++i) {
                NumN type = data.at(X(i)).getType();
                if (type != variable_type_constant) {
                    x = X(i);
                    // x is used
                    if (!data.at(x).getValue().isEmpty()) {
                        break;
                    }
                }
            }
            if (!x.isEmpty()) {
                for (NumN i = 1; i <= X.size(); ++i) {
                    NumN type = data.at(X(i)).getType();
                    if (type == variable_type_constant) {
                        Variable &vk = data.at(X(i));
                        if (vk.getValue().isScalarInMath()) {
                            if (data.at(x).getValue().isScalarInMath()) {
                                continue;
                            }
                            NumR c = vk.getValue()(1);
                            vk.getValue().setSize(data.at(x).getValue().shape());
                            vk.getValue() = c;
                        }
                    } else {
                        if (data.at(X(i)).getValue().isEmpty()) {
                            data.at(X(i)).getValue().setSize(data.at(x).getValue().shape());
                            data.at(X(i)).getValue() = 0;
                        }
                    }
                }
            }
        }

        // constant Y by X
        void setSizeYByX(const SetVar &X, const SetVar &Y, VariableMap &data) {
            const ArrayN &d = data.at(X(1)).getValue().shape();
            for (NumN i = 1; i <= Y.size(); ++i) {
                VarAd y = Y(i);
                data.at(y).getValue().setSize(d);
            }
        }

        // set size of y by x
        void _setSizeyByx(VarAd x, VarAd y, VariableMap &data) {
            const ArrayN &d = data.at(x).getValue().shape();
            if (!data.at(y).getValue().isSameSize(d)) {
                data.at(y).getValue().setSize(d);
            }
        }

        // set size of y by x
        void _setSizeCyByx(VarAd x, VarAd y, VariableMap &data) {
            const ArrayN &d = data.at(x).getValue().shape();
            if (!data.at(y).getValue().isSameSize(d)) {
                Variable &vk = data.at(y);
                NumR c = 0;
                if (vk.getValue().isScalarInMath()) {
                    c = vk.getValue()(1);
                }
                vk.getValue().setSize(d);
                vk.getValue() = c;
            }
        }

        // set size of y by x
        void setSizeyByx(VarAd x, VarAd y, VariableMap &data) {
            NumN type = data.at(y).getType();
            if (type == variable_type_constant) {
                _setSizeCyByx(x, y, data);
            } else {
                _setSizeyByx(x, y, data);
            }
        }
    }
}