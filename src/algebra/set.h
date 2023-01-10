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

    template<typename T>
    struct Set_ {
    private:
        Seqce<T> v;
    public:
        Set_() {
            clear();
        }

        virtual ~Set_() {
        }

        NumN size() const {
            return v.size();
        }

        NumB isEmpty() const {
            if (size() == 0) {
                return 1;
            }
            return 0;
        }

        const T &operator()(NumN j) const {
            return v.operator()(j);
        }

        void add(const T &j) {
            if (!contains(j)) {
                v.push(j);
            }
        }

        void set(const T &j) {
            clear();
            v.push(j);
        }

        void set(const Set_<T> &S) {
            clear();
            add(S);
        }

        // can deprecate
        void add(const Set_<T> &S) {
            for (NumN i = 1; i <= S.size(); ++i) {
                add(S(i));
            }
        }

        template<template<typename> class Container>
        void add(const Container<T> &S) {
            for (NumN i = 1; i <= S.size(); ++i) {
                add(S(i));
            }
        }

        void copyToNox(Set_<T> &S, const T &x) const {
            S.clear();
            for (NumN i = 1; i <= size(); ++i) {
                const T &s = (*this)(i);
                if (s != x)
                    S.add(s);
            }
        }

        void copyTo(Set_<T> &S) const {
            S.clear();
            for (NumN i = 1; i <= size(); ++i) {
                S.add((*this)(i));
            }
        }

        T &at(NumN i) {
            return v.at(i);
        }

        NumB contains(const T &x) const {
            for (NumN i = 1; i <= size(); ++i) {
                if (v.operator()(i) == x) {
                    return 1;
                }
            }
            return 0;
        }

        void clear() {
            v.clear();
        }

        NumB log(const char *s = 0) const {
            return log(std::cout, s);
        }

        NumB log(std::ostream &io, const char *s = 0) const {
            if (s == 0) {
                s = "";
            }
            io << "Set " << s << ":\n";
            v.log(io, 0, 1);
            return 1;
        }

        void intersect(const Set_<T> &X, Set_<T> &Y) const {
            Y.clear();
            for (NumN i = 1; i <= size(); ++i) {
                if (X.contains((*this)(i))) {
                    Y.add((*this)(i));
                }
            }
        }

        void difference(const Set_<T> &X, Set_<T> &Y) const {
            Y.clear();
            for (NumN i = 1; i <= size(); ++i) {
                if (!X.contains((*this)(i))) {
                    Y.add((*this)(i));
                }
            }
        }

        void difference(const T &x, Set_<T> &Y) const {
            Set_<T> X;
            X.add(x);
            difference(X, Y);
        }

        void difference(const T &x) {
            Set_<T> Y;
            difference(x, Y);
            swap(Y);
        }

        void swap(Set_<T> &X) {
            v.swap(X.v);
        }

        void sort() {
            v.sort();
        }

        // using name 'getMax' instead of using name 'max' to avoid some macro problems.
        // max elements
        NumN getMax() const {
            MATH21_ASSERT(!isEmpty())
            SetN B;
            Seqce<T> v2;
            v.copyTo(v2);
            v2.sort();
            return v2(v2.size());
        }

        NumB isEqual(const Set_<T> &S) const{
            Set_<T> X;
            X.set(*this);
            Set_<T> Y;
            Y.set(S);
            X.sort();
            Y.sort();
            return X.v.isEqual(Y.v);
        }

    };

    template<typename T>
    NumB math21_operator_set_isEqual(const Set_<T> &a, const Set_<T> &b) {
        if (a.size() != b.size()) {
            return 0;
        }
        SetN c;
        a.intersect(b, c);
        if (c.size() == a.size()) {
            return 1;
        }
        return 0;
    }

    template<typename T>
    std::ostream &operator<<(std::ostream &out, const Set_<T> &m) {
        m.log(out);
        return out;
    }

    // argwhere from index k = 1
    template<template<typename> class Container, typename T>
    void math21_operator_container_argwhere(const Container<T> &m,
                                            const T &x, SetN &set, NumR epsilon = 0, NumN k = 1) {
        NumN i;
        NumN n = m.size();
        set.clear();
        MATH21_ASSERT(k >= 1 && k <= n);
        for (i = k; i <= n; ++i) {
            if (math21_point_isEqual(m(i), x, epsilon)) {
                set.add(i);
            }
        }
    }

    template<template<typename> class Container, typename T>
    void math21_convert_container_to_set(const Container<T> &A, Set_<T> &S) {
        S.clear();
        S.add(A);
    }

    template<template<typename> class Container, typename T>
    void math21_convert_set_to_container(const Set_<T> &S, Container<T> &A, NumN n = 0, NumN offset = 0) {
        n = math21_number_get_n_and_check_offset(S.size(), n, offset);
        A.setSize(n);
        for (NumN i = 1; i <= n; ++i) {
            A(i) = S(i + offset);
        }
    }

}