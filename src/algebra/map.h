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
#include <map>

namespace math21 {

    template<typename T, typename S>
    struct Map_ {
    private:
        S b_dummy;
        std::map<T, S> data;

        void init();

    public:

        Map_();

        virtual ~Map_();

        NumN size() const;

        NumB isEmpty() const;

//        const Seqce <T> &getX() const;

        std::map<T, S> &getData();

        const std::map<T, S> &getData() const;

        void getX(Set_ <T> &keys) const;

        void getY(Seqce <S> &vs) const;

//        const Seqce <S> &getY() const;

        NumB get(const T &x, S &y) const;

        S &valueAt(const T &x);

        S &valueAt(const T &x, NumB &flag);

        const S &valueAt(const T &x) const;

        const S &valueAt(const T &x, NumB &flag) const;

        NumN has(const T &x) const;

        NumB remove(const T &x);

        void add(const T &x, const S &y);

        void add(const Set_ <T> &X, const S &y);

        void add(const Relation <T, S> &r);

        void clear();

        void restrictTo(const Set_ <T> &X, Map_<T, S> &dst);

        void log(const char *s = 0) const;

        void log(std::ostream &io, const char *s = 0) const;

        void serialize(std::ostream &io, SerializeNumInterface &sn) const;

        void deserialize(std::istream &io, DeserializeNumInterface &sn);
    };

    template<typename T, typename S>
    void Map_<T, S>::init() {
        clear();
    }

    template<typename T, typename S>
    Map_<T, S>::Map_() {
        init();
    }

    template<typename T, typename S>
    Map_<T, S>::~Map_() {
        clear();
    }

    template<typename T, typename S>
    NumN Map_<T, S>::size() const {
        MATH21_ASSERT(data.size()<NumN32_MAX);
        return (NumN) data.size();
    }

    template<typename T, typename S>
    NumB Map_<T, S>::isEmpty() const {
        if (size() == 0) {
            return 1;
        } else {
            return 0;
        }
    }

//    template<typename T, typename S>
//    const Seqce <T> &Map_<T, S>::getX() const {
//        math21_tool_assert(0);
//    }

    template<typename T, typename S>
    std::map<T, S> &Map_<T, S>::getData() {
        return data;
    }

    template<typename T, typename S>
    const std::map<T, S> &Map_<T, S>::getData() const {
        return data;
    }

    template<typename T, typename S>
    void Map_<T, S>::getX(Set_ <T> &keys) const {
        keys.clear();
        if (isEmpty()) {
            return;
        }
        for (auto itr = data.begin(); itr != data.end(); ++itr) {
            keys.add(itr->first);
        }
    }

    template<typename T, typename S>
    void Map_<T, S>::getY(Seqce <S> &vs) const {
        vs.clear();
        if (isEmpty()) {
            return;
        }
        vs.setSize(size());
        NumN i = 1;
        for (auto itr = data.begin(); itr != data.end(); ++itr) {
            vs.at(i) = itr->second;
            ++i;
        }
    }

//    template<typename T, typename S>
//    const Seqce <S> &Map_<T, S>::getY() const {
//        math21_tool_assert(0);
//    }

    template<typename T, typename S>
    NumB Map_<T, S>::get(const T &x, S &y) const {
        auto itr = data.find(x);
        if (itr != data.end()) {
            y = itr->second;
            return 1;
        } else {
            return 0;
        }
    }

    template<typename T, typename S>
    S &Map_<T, S>::valueAt(const T &x) {
        auto itr = data.find(x);
        if (itr == data.end()) {
            MATH21_ASSERT(0);
        }
        return itr->second;
    }

    template<typename T, typename S>
    S &Map_<T, S>::valueAt(const T &x, NumB &flag) {
        auto itr = data.find(x);
        if (itr != data.end()) {
            flag = 1;
            return itr->second;
        }
        flag = 0;
        return b_dummy;
    }

    template<typename T, typename S>
    const S &Map_<T, S>::valueAt(const T &x) const {
        auto itr = data.find(x);
        if (itr != data.end()) {
            return itr->second;
        }
        MATH21_ASSERT(0)
        return b_dummy;
    }

    template<typename T, typename S>
    const S &Map_<T, S>::valueAt(const T &x, NumB &flag) const {
        auto itr = data.find(x);
        if (itr != data.end()) {
            flag = 1;
            return itr->second;
        }
        flag = 0;
        return b_dummy;
    }

    // todo: use NumB as return type
    template<typename T, typename S>
    NumN Map_<T, S>::has(const T &x) const {
        auto itr = data.find(x);
        if (itr != data.end()) {
            return 1;
        }
        return 0;
    }

    template<typename T, typename S>
    NumB Map_<T, S>::remove(const T &x) {
        if (data.erase(x) == 1) {
            return 1;
        } else {
            return 0;
        }
    }

    // It will fail if (x, *) exists
    template<typename T, typename S>
    void Map_<T, S>::add(const T &x, const S &y) {
        NumB flag = (NumB) data.insert(std::pair<T, S>(x, y)).second;
        if (!flag) {
            m21warn("add failed!");
            std::cout << "x = " << x << ", y = " << y << std::endl;
        }
    }

    template<typename T, typename S>
    void Map_<T, S>::add(const Set_ <T> &X, const S &y) {
        for (NumN i = 1; i <= X.size(); ++i) {
            add(X(i), y);
        }
    }

    template<typename T, typename S>
    void Map_<T, S>::add(const Relation <T, S> &r) {
        for (NumN i = 1; i <= r.size(); ++i) {
            add(r.keyAtIndex(i), r.valueAtIndex(i));
        }
    }

    template<typename T, typename S>
    void Map_<T, S>::clear() {
        data.clear();
    }

    // restrict f: X -> Y to X0 to get f: Xs -> Y, with Xs = intersect(X, X0)
    template<typename T, typename S>
    void Map_<T, S>::restrictTo(const Set_ <T> &X0, Map_<T, S> &dst) {
        dst.clear();
        for (auto itr = data.begin(); itr != data.end(); ++itr) {
            if (X0.contains(itr->first)) {
                dst.add(itr->first, itr->second);
            }
        }
    }

    template<typename T, typename S>
    void Map_<T, S>::log(const char *s) const {
        log(std::cout, s);
    }

    template<typename T, typename S>
    void Map_<T, S>::log(std::ostream &io, const char *s) const {
        if (s == 0) {
            s = "";
        }
        io << "Map_ " << s << ":\n";
        for (auto itr = data.begin(); itr != data.end(); ++itr) {
            io << "(" << itr->first << ", " << itr->second << ")\n";
        }
    }

    template<typename T, typename S>
    void Map_<T, S>::serialize(std::ostream &io, SerializeNumInterface &sn) const {
        NumN n = size();
        math21_io_serialize(io, n, sn);
        for (auto itr = data.begin(); itr != data.end(); ++itr) {
            math21_io_serialize(io, itr->first, sn);
            math21_io_serialize(io, itr->second, sn);
        }
    }

    template<typename T, typename S>
    void Map_<T, S>::deserialize(std::istream &io, DeserializeNumInterface &sn) {
        clear();
        NumN n;
        math21_io_deserialize(io, n, sn);
        T a;
        S b;
        for (NumN i = 1; i <= n; ++i) {
            math21_io_deserialize(io, a, sn);
            math21_io_deserialize(io, b, sn);
            add(a, b);
        }
    }

    template<typename T, typename S>
    void math21_io_serialize(std::ostream &io, const Map_<T, S> &m, SerializeNumInterface &sn) {
        m.serialize(io, sn);
    }

    template<typename T, typename S>
    void math21_io_deserialize(std::istream &io, Map_<T, S> &m, DeserializeNumInterface &sn) {
        m.deserialize(io, sn);
    }
}