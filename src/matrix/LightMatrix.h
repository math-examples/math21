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

    // LightMatrix is a wrapper of c array.
    // So we can easily access element using operator() with index starting from 1 instead of 0.
    //
    template<typename T>
    class LightMatrix {
    private:
        NumN d1, d2;
        T *data; // data may be constant if isDataExternal = 1.
        NumB isDataExternal;

        void init() {
            data = 0;
            d1 = 0;
            d2 = 0;
            isDataExternal = 0;
        }

    public:
        LightMatrix() {
            init();
        }

        LightMatrix(const LightMatrix &vector) {
            init();
            copyFrom(vector);
        }

        LightMatrix(NumN d1) {
            init();
            setSize(d1, 1);
        }

        LightMatrix(NumN d1, NumN d2) {
            init();
            setSize(d1, d2);
        }

        LightMatrix(NumN d1, NumN d2, T *data) {
            set(d1, d2, data);
        }

        LightMatrix(NumN d1, NumN d2, const T *data) {
            set(d1, d2, (T *) data);
        }

        LightMatrix(Tensor <T> &m) {
            set(m.nrows(), m.ncols(), m.getDataAddress());
        }

        LightMatrix(const Tensor <T> &m) {
            set(m.nrows(), m.ncols(), (T *) m.getDataAddress());
        }

        void setSize(NumN d1, NumN d2) {
            MATH21_ASSERT(!isDataExternal);
            if (d1 * d2) {
                if (d1 * d2 != this->d1 * this->d2) {
                    if (data)data += 1;
                    data = (T *) math21_vector_setSize_buffer_cpu(data, d1 * d2, sizeof(T));
                    if (data)data -= 1;
                }
            } else {
                clear();
            }
            this->d1 = d1;
            this->d2 = d2;
            if (d1 * d2) {
                MATH21_ASSERT(data);
            } else {
                MATH21_ASSERT(!data);
            }
        }

        void set(NumN d1, NumN d2, T *data) {
            if (data)data -= 1;
            this->data = data;
            this->d1 = d1;
            this->d2 = d2;
            this->isDataExternal = 1;
            if (d1 * d2) {
                MATH21_ASSERT(data);
            } else {
                MATH21_ASSERT(!data);
            }
        }

        void set(Tensor <T> &m) {
            set(m.nrows(), m.ncols(), m.getDataAddress());
        }

        void set(const Tensor <T> &m) {
            set(m.nrows(), m.ncols(), (T *) m.getDataAddress());
        }

        void copyFrom(NumN d1, NumN d2, const T *data) {
            setSize(d1, d2);
            if (d1 * d2) {
                MATH21_ASSERT(data);
                math21_vector_copy_buffer_cpu(this->data + 1, data, d1 * d2, sizeof(T));
            }
        }

        void copyFrom(const Tensor <T> &m) {
            copyFrom(m.nrows(), m.ncols(), m.getDataAddress());
        }

        void copyTo(Tensor <T> &m) const {
            if (!isEmpty()) {
                m.setSize(d1, d2);
                math21_memory_tensor_data_copy_to_tensor_cpu(m, data + 1);
            } else {
                m.clear();
            }
        }

        void copyFrom(const LightMatrix &v) {
            if (!v.isEmpty())copyFrom(v.d1, v.d2, v.data + 1);
            else clear();
        }

        void clear() {
            if (!isEmpty()) {
                if (!isDataExternal) {
                    math21_vector_free_cpu(data + 1);
                } else {
                    isDataExternal = 0;
                }
                d1 = 0;
                d2 = 0;
                data = 0;
            }
        }

        virtual ~LightMatrix() {
            clear();
        }

        // j1 >= 1
        T &at(NumN j1) {
            return data[j1];
        }

        // j1 >= 1
        T &operator()(NumN j1) {
            return data[j1];
        }

        // j1 >= 1
        const T &operator()(NumN j1) const {
            return data[j1];
        }

        // j1 >= 1
        T &operator()(NumN j1, NumN j2) {
            return data[(j1 - 1) * d2 + j2];
        }

        // j1 >= 1
        const T &operator()(NumN j1, NumN j2) const {
            return data[(j1 - 1) * d2 + j2];
        }

        NumB isEmpty() const {
            if (data == 0) {
                MATH21_ASSERT(size() == 0);
                return (NumB) 1;
            } else {
                MATH21_ASSERT(size() != 0);
                return (NumB) 0;
            }
        }

        NumN size() const {
            return d1 * d2;
        }

        NumN nrows() const {
            return d1;
        }

        NumN ncols() const {
            return d2;
        }

        NumB log(const char *name = 0, const char *gap = 0, NumN precision = 3) const {
            return math21_operator_container_print(*this, std::cout, name, gap, precision);
        }
    };

    typedef LightMatrix<NumN> LightMatN;
    typedef LightMatrix<NumZ> LightMatZ;
    typedef LightMatrix<NumR> LightMatR;
}