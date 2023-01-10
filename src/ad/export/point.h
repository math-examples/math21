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
        VariableMap &ad_global_get_data();

        Derivative &ad_global_get_derivative();

        // used only when have fresh start.
        void ad_clear_graph();

        struct PointAd {
        private:
            VarAd var;
        public:

            PointAd();

            // create constant num x
            PointAd(NumR x);

            // Todo: error-prone, consider removing '=0', and use two args at least.
            template<typename T>
            PointAd(const Tensor <T> &m, NumB isInput = 0, NumN deviceType = m21_device_type_default);

            explicit PointAd(VarAd var);

            PointAd(const PointAd &p);

            virtual ~PointAd();

            PointAd &operator=(const PointAd &p);

            NumB isEmpty() const;

            VarAd getVarAd() const;

            void clear();

            void log(const char *name = 0, NumN precision = 3) const;

            void log(std::ostream &io, const char *name = 0, NumN precision = 3) const;
        };

        bool operator==(const PointAd &p1, const PointAd &p2);

        VariableMap &ad_get_data();

        TenR &ad_get_value(const PointAd &p);

        NumN ad_get_dim_i(const PointAd &x, NumN i);

        Variable &ad_get_variable(const PointAd &p);

        NumB ad_is_const_variable(const PointAd &p);

        template<typename T>
        void ad_point_setValue(const PointAd &p, const Tensor <T> &value) {
            Variable &vx = ad_global_get_data().at(p.getVarAd());
            vx.getValue() = value;
        }

        void ad_point_set_device_type(const PointAd &p, NumN deviceType);

        NumN ad_get_device_type(const PointAd &p);

        NumB ad_point_is_cpu(const PointAd &p);

        PointAd ad_create_point_var(const char *name = 0);

        PointAd ad_create_point_const(const char *name = 0);

        template<typename T>
        PointAd ad_create_point_input_with_value(const Tensor <T> &value, const char *name = 0,
                                                 NumN deviceType = m21_device_type_default) {
            auto p = ad_create_point_var(name);
            VarAd x = p.getVarAd();
            Variable &vx = ad_global_get_data().at(x);
            vx.setType(variable_type_input);
            ad_point_set_device_type(p, deviceType);
            ad_point_setValue(p, value);
            return p;
        }

        template<typename T>
        PointAd ad_create_point_const_with_value(const Tensor <T> &value, const char *name = 0,
                                                 NumN deviceType = m21_device_type_default) {
            auto p = ad_create_point_const(name);
            ad_point_set_device_type(p, deviceType);
            ad_point_setValue(p, value);
            return p;
        }

        PointAd ad_num_const(NumR x);

        template<typename T>
        PointAd::PointAd(const Tensor <T> &m, NumB isInput, NumN deviceType) {
            if (!isInput) {
                *this = ad_create_point_const_with_value(m, "", deviceType);
            } else {
                *this = ad_create_point_input_with_value(m, 0, deviceType);
            }
        }
    }

    std::ostream &operator<<(std::ostream &io, const ad::PointAd &m);

    void math21_io_serialize(std::ostream &out, const ad::PointAd &m, SerializeNumInterface &sn);

    void math21_io_deserialize(std::istream &in, ad::PointAd &m, DeserializeNumInterface &sn);

    void math21_ad_destroy();

    // yf: yield function
    template<typename T>
    NumN yf_get_dim_i(const T &x, NumN i) {
        return ad::ad_get_dim_i(x, i);
    }

    template<typename T>
    TenR &yf_get_value(const T &x) {
        return ad::ad_get_value(x);
    }

    template<typename T>
    NumB yf_is_value_empty(const T &x) {
        return ad::ad_get_value(x).isEmpty();
    }
}