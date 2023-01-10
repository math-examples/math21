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

#include "Queue.h"

namespace math21 {

    namespace data_structure {
        template<typename Item>
        class BigQueue {
        private:
            Queue <Item> cur;
            Queue <std::string> files;
            Queue <Item> next;
            NumN n;
            NumN maxMemSize;
            NumN iname;
            static NumN id;
        public:
            BigQueue() {
                MATH21_ASSERT(id < NumN_MAX);
                ++id;

                n = 0;
                maxMemSize = 10000;
                iname = 0;
            }

            virtual ~BigQueue() {
                while (!isEmpty()) {
                    dequeue();
                }
            }

            void setMaxMemSize(NumN maxMemSize_) {
                maxMemSize = maxMemSize_;
            }

            bool isEmpty() const {
                return n == 0;
            }

            NumN size() const {
                return n;
            }

            void enqueue(Item item) {
                if (next.size() >= maxMemSize) {
                    std::string name = math21_string_to_string(
                            "BigQueue_", id, "_", iname, ".bin");
                    ++iname;
                    math21_io_generic_type_write_to_file(next, name.c_str());
                    files.enqueue(name);
                    next.clear();
                }
                next.enqueue(item);
                MATH21_ASSERT(n < NumN_MAX);
                ++n;
            }

            Item dequeue() {
                MATH21_ASSERT(!isEmpty(), "BigQueue underflow");
                if (cur.isEmpty()) {
                    if (!files.isEmpty()) {
                        auto name = files.dequeue();
                        math21_io_generic_type_read_from_file(cur, name.c_str());
                        // the file may be used by serialization, so either copy it, or keep it.
                        // For now, just clear content. todo: remove file from disk.
//                        math21_io_generic_type_write_to_file(0, name.c_str(), 0, 0);
                    } else {
                        cur.swap(next);
                    }
                }
                --n;
                return cur.dequeue();
            }

            void log(const char *name = 0) const {
                log(std::cout, name);
            }

            void log(std::ostream &io, const char *name = 0) const {
                if (name) {
                    io << "BigQueue: " << name << "\n";
                }
                cur.log(io, "cur");
                files.log(io, "files");
                next.log(io, "next");
            }

            void serialize(std::ostream &io, SerializeNumInterface &sn) const {
                math21_io_serialize(io, cur, sn);
                math21_io_serialize(io, files, sn);
                math21_io_serialize(io, next, sn);
                math21_io_serialize(io, n, sn);
                math21_io_serialize(io, maxMemSize, sn);
                math21_io_serialize(io, iname, sn);
            }

            void deserialize(std::istream &io, DeserializeNumInterface &sn) {
                math21_io_deserialize(io, cur, sn);
                math21_io_deserialize(io, files, sn);
                math21_io_deserialize(io, next, sn);
                math21_io_deserialize(io, n, sn);
                math21_io_deserialize(io, maxMemSize, sn);
                math21_io_deserialize(io, iname, sn);
            }

        };

        template<typename Item>
        NumN BigQueue<Item>::id = 0;
    }

    template<typename T>
    void math21_io_serialize(std::ostream &io, const data_structure::BigQueue<T> &m, SerializeNumInterface &sn) {
        m.serialize(io, sn);
    }

    template<typename T>
    void math21_io_deserialize(std::istream &io, data_structure::BigQueue<T> &m, DeserializeNumInterface &sn) {
        m.deserialize(io, sn);
    }
}