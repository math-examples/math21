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

#include <memory>

#include "inner.h"

namespace math21 {

    template<typename T>
    class PtrShared {
    private:
        std::shared_ptr<T> p_;
    public:
        PtrShared() = default;

        ~PtrShared() = default;

        void set(T *p) {
            p_.reset(p);
        }

        template<typename... Args>
        void make(Args... args) {
            p_ = std::make_shared<T>(args...);
        }

        T *operator->() const { return p_.operator->(); }

        T *get() const { return p_.get(); }

        T &operator*() { return *p_; }

        const T &operator*() const { return *p_; }

        operator bool() const { return (bool) p_.operator bool(); }

        void reset() {
            p_.reset();
        }

        NumB log(const char *s = 0) const {
            return log(std::cout, s);
        }

        NumB log(std::ostream &io, const char *s = 0) const {
            if (s == 0) {
                s = "";
            }
            io << "PtrShared " << s << ":\n";
            io << "pointer " << *p_ << "\n";
            return 1;
        }
    };

    namespace short_name {
        template<typename T, typename ... Args>
        static inline PtrShared<T> makePtrShared(const Args &... args) {
            PtrShared<T> ptrShared;
            ptrShared.make(args...);
            return ptrShared;
        }

        template<typename T>
        static inline PtrShared<T> setPtrShared(T *p) {
            PtrShared<T> ptrShared;
            ptrShared.set(p);
            return ptrShared;
        }
    }

    template<typename T>
    void math21_io_serialize(std::ostream &io, const PtrShared<T> &m, SerializeNumInterface &sn) {
        math21_io_serialize(io, *m, sn);
    }

    template<typename T>
    void math21_io_deserialize(std::istream &io, PtrShared<T> &m, DeserializeNumInterface &sn) {
        m = short_name::makePtrShared<T>();
        math21_io_deserialize(io, *m, sn);
    }
}