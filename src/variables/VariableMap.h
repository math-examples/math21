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

        // VariableMap
        struct VariableMap {
        private:
            VarAd constant_0;
            VarAd constant_1;
            VarAd constant_m1;

            data_structure::Stack<VarAd> varData; // data
            // todo: optimize Set
            SetVar &V; // variable ids

            // backup
            SetVar V_backup;
            NumN v_size_backup;

            void init();

            VarAd _createSharedC(NumR x, const char *name);

            void _createSomeSharedC();

            VarAd _createV(const char *name, NumB isShared, VarAd sharedId);

        public:
            VariableMap(SetVar &V);

            virtual ~VariableMap();

            void clear();

            // todo: error, should make it different.
            VarAd get_constant_0();

            VarAd get_constant_1();

            // -1
            VarAd get_constant_m1();

            NumN size() const;

            NumB isEmpty() const;

            NumB log(const char *s = 0) const;

            NumB log(std::ostream &io, const char *s = 0) const;

            // this will make previous references to pointer content invalid.
            // So use references to data.
            VarAd createV(const char *name = 0);

            // create constant.
            VarAd createC(const char *name);

            void setDeviceType(VarAd id, NumN deviceType);

            void setValue(VarAd id, NumR x);

            Variable &at(VarAd i);

            data_structure::Stack<VarAd> &getVarData();

            const Variable &operator()(VarAd i) const;

            SetVar &getV();

            void backup();

            void restore();

            void reset();

        };

        // constant X by variable X
        void setSizeCXUXByX(const SetVar &X, VariableMap &data);

        // constant Y by X
        void setSizeYByX(const SetVar &X, const SetVar &Y, VariableMap &data);

        void setSizeyByx(VarAd x, VarAd y, VariableMap &data);
    }
}