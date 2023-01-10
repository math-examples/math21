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

#include "../functions_01/files.h"
#include "../functions_02/files.h"
#include "../differential.h"
#include "01.h"
#include "point.h"

namespace math21 {
    namespace ad {
        SetVar V;
        VariableMap data(V);
        Derivative derivative(data);

        VariableMap &ad_global_get_data() {
            return data;
        }

        Derivative &ad_global_get_derivative() {
            return derivative;
        }

        void ad_clear_graph() {
            data.reset();
        }

        PointAd::PointAd() {
            var = VarAd(0);
        }

        PointAd::PointAd(NumR x) {
            *this = ad_num_const(x);
        }

        PointAd::PointAd(VarAd var) : var(var) {
        }

        PointAd::PointAd(const PointAd &p) {
            var = p.var;
        }

        PointAd::~PointAd() {}

        PointAd &PointAd::operator=(const PointAd &p) {
            var = p.var;
            return *this;
        }

        NumB PointAd::isEmpty() const {
            return var.isEmpty();
        }

        VarAd PointAd::getVarAd() const {
            return var;
        }

        void PointAd::clear() {
            var = VarAd(0);
        }

        void PointAd::log(const char *name, NumN precision) const {
            log(std::cout, name, precision);
        }

        void PointAd::log(std::ostream &io, const char *name, NumN precision) const {
            ad_get_variable(*this).getValue().log(io, name, 0, 0, precision);
        }

        bool operator==(const PointAd &p1, const PointAd &p2) {
            return p1.getVarAd() == p2.getVarAd();
        }

        VariableMap &ad_get_data() {
            return data;
        }

        TenR &ad_get_value(const PointAd &p) {
            MATH21_ASSERT(!p.isEmpty())
            return data.at(p.getVarAd()).getValue();
        }

        NumN ad_get_dim_i(const PointAd &x, NumN i) {
            return ad_get_value(x).dim(i);
        }

        Variable &ad_get_variable(const PointAd &p) {
            MATH21_ASSERT(!p.isEmpty())
            return data.at(p.getVarAd());
        }

        NumB ad_is_const_variable(const PointAd &p) {
            if (ad_get_variable(p).getType() == variable_type_constant) {
                return 1;
            } else {
                return 0;
            }
        }

        void ad_point_set_device_type(const PointAd &p, NumN deviceType) {
            data.setDeviceType(p.getVarAd(), deviceType);
        }

        NumN ad_get_device_type(const PointAd &p) {
            return ad_get_variable(p).getValue().getDeviceType();
        }

        NumB ad_point_is_cpu(const PointAd &p) {
            if (ad_get_variable(p).getValue().getDeviceType() == m21_device_type_gpu) {
                return 0;
            } else {
                return 1;
            }
        }

        PointAd ad_create_point_var(const char *name) {
            return PointAd(data.createV(name));
        }

        PointAd ad_create_point_const(const char *name) {
            return PointAd(data.createC(name));
        }

        PointAd ad_num_const(NumR x) {
//            m21warn("error-prone ad_num_const");
            VarAd id;
            if (x == 0) {
                id = ad_global_get_constant_0();
            } else if (x == 1) {
                id = ad_global_get_constant_1();
            } else if (x == -1) {
                id = ad_global_get_constant_m1();
            } else {
                id = data.createC(math21_string_to_string(x).c_str());
                data.setValue(id, x);
            }
            return PointAd(id);
        }
    }


    std::ostream &operator<<(std::ostream &io, const ad::PointAd &m) {
        m.log(io);
        return io;
    }

    void math21_io_serialize(std::ostream &out, const ad::PointAd &m, SerializeNumInterface &sn) {
        math21_io_serialize(out, m.getVarAd(), sn);
    }

    void math21_io_deserialize(std::istream &in, ad::PointAd &m, DeserializeNumInterface &sn) {
        ad::VarAd var;
        math21_io_deserialize(in, var, sn);
        m = ad::PointAd(var);
    }

    void math21_ad_destroy() {
        ad::data.clear();
    }
}