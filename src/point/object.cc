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

#include "point_c.h"
#include "object.h"
#include "inner_cc.h"

namespace math21 {

    NumZ Uobj::count = 0;

    Uobj::Uobj() {
        point = math21_point_init(point);
    }

    Uobj::~Uobj() {
        clear();
    }

    Uobj::Uobj(const Uobj &o) {
        point = math21_point_share_assign(o.point);
    }

    Uobj &Uobj::operator=(const Uobj &o) {
        if (this != &o) {
            if (!isEmpty())clear();
            point = math21_point_share_assign(o.point);
        }
        return *this;
    }

    Uobj &Uobj::create(NumN type) {
        if (!isEmpty())clear();
        point = math21_point_create_by_type(type);
        return *this;
    }

    void Uobj::clear() {
        point = math21_point_destroy(point);
    }

    NumB Uobj::isEmpty() const {
        return math21_point_is_empty(point);
    }

    NumB Uobj::isContentEmpty() const {
        if(isEmpty())return 1;
        return math21_point_is_content_empty(point);
    }

    NumB Uobj::isTenN() const {
        if (point.type == m21_type_TenN) {
            return 1;
        }
        return 0;
    }

    NumB Uobj::isTenN8() const {
        if (point.type == m21_type_TenN8) {
            return 1;
        }
        return 0;
    }

    NumB Uobj::isTenZ() const {
        if (point.type == m21_type_TenZ) {
            return 1;
        }
        return 0;
    }

    NumB Uobj::isTenR() const {
        if (point.type == m21_type_TenR) {
            return 1;
        }
        return 0;
    }

    NumB Uobj::log(const char *name) const {
        math21_point_log_cc(point, name);
        return 1;
    }

    NumB Uobj::log(std::ostream &io, const char *name) const {
        math21_point_log_cc(io, point, name);
        return 1;
    }

    void Uobj::setSize(const VecN &d) {
        math21_point_tensor_set_size(point, d);
    }

    void Uobj::setValue(NumR k) {
        math21_point_tensor_set_value(point, k);
    }

    void Uobj::letters(NumZ start_letter) {
        math21_point_tensor_set_letters(point, start_letter);
    }

    NumN Uobj::type() const {
        return point.type;
    }

    m21point Uobj::getPoint() const {
        return point;
    }

    std::ostream &operator<<(std::ostream &out, const Uobj &m) {
        m.log(out);
        return out;
    }

    void math21_io_serialize(std::ostream &out, const Uobj &v, SerializeNumInterface &sn) {
        math21_io_serialize(out, v.getPoint(), sn);
    }

    void math21_io_deserialize(std::istream &in, Uobj &v, DeserializeNumInterface &sn) {
        m21point point;
        math21_io_deserialize(in, point, sn);
        v = Uobj(point);
//        math21_point_destroy(point);
    }

    Uobj math21_object_read(const char *path) {
        return Uobj(math21_point_read(path));
    }

    NumB math21_object_write(const Uobj &object, const char *path, NumB binary) {
        return math21_point_write(object.getPoint(), path, binary);
    }

    NumPtr Uobj::math21_object_create() {
        ++count;
        m21log("Uobj create count", count);
        void *ptr = new Uobj();
        MATH21_ASSERT(sizeof(void *) <= sizeof(NumPtr));
        return (NumPtr) ptr;
    }

    void Uobj::math21_object_destroy(NumPtr ptr) {
        --count;
        m21log("Uobj destroy count", count);
        delete (Uobj *) ptr;
    }
}