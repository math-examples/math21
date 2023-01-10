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

#include "point_cc.h"

namespace math21 {
    // RAII (Resource Acquisition Is Initialization): m21point -> Uobj
    // Uniform Object, a wrapper over the actual data
    class Uobj {
    private:
        static NumZ count;
        std::string name;
        m21point point;
    public:
        Uobj();

        ~Uobj();

        Uobj(const Uobj &o);

        Uobj &operator=(const Uobj &o);

        Uobj &create(NumN type);

        template<typename T>
        Uobj(const T &x) {
            point = math21_cast_to_point(x);
        }

        void clear();

        NumB isEmpty() const;

        NumB isContentEmpty() const;

        template<typename T>
        T &get() {
            return math21_cast_to_T<T>(point);
        }

        template<typename T>
        const T &get() const {
            return math21_cast_to_T<T>(point);
        }

        NumB isTenN() const;

        NumB isTenN8() const;

        NumB isTenZ() const;

        NumB isTenR() const;

        NumB log(const char *name = 0) const;

        void setSize(const VecN &d);

        void setValue(NumR k);

        void letters(NumZ start_letter);

        NumB log(std::ostream &io, const char *name = 0) const;

        NumN type() const;

        m21point getPoint() const;

        static NumPtr math21_object_create();

        static void math21_object_destroy(NumPtr ptr);
    };

    std::ostream &operator<<(std::ostream &out, const Uobj &m);

    void math21_io_serialize(std::ostream &out, const Uobj &v, SerializeNumInterface &sn);

    void math21_io_deserialize(std::istream &in, Uobj &v, DeserializeNumInterface &sn);

    Uobj math21_object_read(const char *path);

    NumB math21_object_write(const Uobj &object, const char *path, NumB binary);
}