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

#include "number.h"
#include <cmath>
#include <sstream>

namespace math21 {

    template<typename T>
    std::string math21_string_to_string(const T &x) {
        std::ostringstream oss;
        oss << x;
        return oss.str();
    }

    template<typename T, typename T2>
    std::string math21_string_to_string(const T &x, const T2 &x2) {
        std::ostringstream oss;
        oss << x << x2;
        return oss.str();
    }

    template<typename T, typename T2, typename T3>
    std::string math21_string_to_string(const T &x, const T2 &x2, const T3 &x3) {
        std::ostringstream oss;
        oss << x << x2 << x3;
        return oss.str();
    }

    template<typename T, typename T2, typename T3, typename T4>
    std::string math21_string_to_string(const T &x, const T2 &x2, const T3 &x3, const T4 &x4) {
        std::ostringstream oss;
        oss << x << x2 << x3 << x4;
        return oss.str();
    }

    template<typename T, typename T2, typename T3, typename T4, typename T5>
    std::string math21_string_to_string(const T &x, const T2 &x2, const T3 &x3,
            const T4 &x4, const T5 &x5) {
        std::ostringstream oss;
        oss << x << x2 << x3 << x4 << x5;
        return oss.str();
    }

    template<typename T, template<typename> class Container>
    std::string math21_string_to_string(const Container<T> &x, const std::string &separator = ", ") {
        std::ostringstream oss;
        for (NumN i = 1; i <= x.size(); ++i) {
            oss << x(i);
            if (i < x.size())oss << separator;
        }
        return oss.str();
    }

    template<typename T>
    void math21_string_to_type_generic(const std::string &s, T &num) {
        std::istringstream iss(s);
        iss >> num;
    }

    template<>
    void math21_string_to_type_generic(const std::string &s, std::string &num);

    NumR math21_string_to_NumR(const std::string &s);

    NumN math21_string_to_NumN(const std::string &s);

    NumZ math21_string_to_NumZ(const std::string &s);

    std::string math21_string_NumZ_to_string(NumZ x, NumN width = 0);

    std::string math21_string_replicate_n(NumN n, const std::string &s);

    std::string math21_string_concatenate(const std::string &s1, const std::string &s2);

    std::string
    math21_string_concatenate(const std::string &s1, const std::string &s2, const std::string &s3);

    std::string
    math21_string_concatenate(const std::string &s1, const std::string &s2, const std::string &s3,
                              const std::string &s4);

    std::string
    math21_string_concatenate(const std::string &s1, const std::string &s2, const std::string &s3,
                              const std::string &s4, const std::string &s5);

    NumB math21_string_is_equal(const char *str1, const char *str2);

    NumB math21_string_is_equal(const std::string &str1, const std::string &str2);

    template<typename LogType>
    void math21_string_log_2_string(const LogType &A, std::string &s) {
        std::ostringstream oss;
        A.log(oss);
        s = oss.str();
    }


    NumB math21_point_isEqual(const NumR &x, const NumR &y, NumR epsilon = 0);

    NumB math21_point_isEqual(const NumN &x, const NumN &y, NumR epsilon = 0);

    NumB math21_point_isEqual(const NumZ &x, const NumZ &y, NumR epsilon = 0);

    NumB math21_type_size_t_is_4_bytes();

    NumB math21_type_NumSize_is_4_bytes();
}