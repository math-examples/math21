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
#include "Node.h"

namespace math21 {

    namespace data_structure {

        template<typename Item>
        class Queue {
        private:
            Node<Item> *first;    // beginning of queue
            Node<Item> *last;     // end of queue
            NumN n;               // number of elements on queue

            void _enqueue(Node<Item> *node) {
                if (isEmpty()) {
                    first = node;
                } else {
                    last->next = node;
                }
                last = node;
                MATH21_ASSERT(n < NumN_MAX);
                ++n;
            }

        public:
            Queue() {
                first = 0;
                last = 0;
                n = 0;
            }

            void clear() {
                while (!isEmpty()) {
                    dequeue();
                }
            }

            virtual ~Queue() {
                clear();
            }

            void swap(Queue<Item> &queue) {
                xjswap(first, queue.first);
                xjswap(last, queue.last);
                xjswap(n, queue.n);
            }

            NumB isEmpty() const {
                return first == 0 ? 1 : 0;
            }

            NumN size() const {
                return n;
            }

            Item peek() const {
                MATH21_ASSERT(!isEmpty(), "Queue underflow")
                return first->item;
            }

            // std::queue push
            void enqueue(Item item) {
                auto node = new Node<Item>();
                node->item = item;
                _enqueue(node);
            }

            // item = queue.front(); queue.pop();
            Item dequeue() {
                MATH21_ASSERT(!isEmpty(), "Queue underflow")
                Node<Item> *oldfirst = first;
                Item item = first->item;
                first = first->next;
                delete oldfirst;
                --n;
                if (isEmpty()) last = 0;   // to avoid loitering
                return item;
            }

            void log(const char *name = 0) const {
                log(std::cout, name);
            }

            void log(std::ostream &io, const char *name = 0) const {
                if (name) {
                    io << "Queue: " << name << "\n";
                }
                auto iterator = getIterator();
                while (iterator.hasNext()) {
                    io << iterator.next() << " ";
                }
                io << "\n";
            }

            ListIterator<Item> getIterator() const {
                ListIterator<Item> iterator(first);
                return iterator;
            }

            void serialize(std::ostream &io, SerializeNumInterface &sn) const {
                math21_io_serialize(io, n, sn);
                auto iterator = getIterator();
                while (iterator.hasNext()) {
                    math21_io_serialize(io, iterator.next(), sn);
                }
            }

            void deserialize(std::istream &io, DeserializeNumInterface &sn) {
                clear();
                NumN n;
                math21_io_deserialize(io, n, sn);
                for (NumN i = 1; i <= n; ++i) {
                    auto node = new Node<Item>();
                    math21_io_deserialize(io, node->item, sn);
                    _enqueue(node);
                }
            }
        };
    }

    template<typename T>
    void math21_io_serialize(std::ostream &io, const data_structure::Queue<T> &m, SerializeNumInterface &sn) {
        m.serialize(io, sn);
    }

    template<typename T>
    void math21_io_deserialize(std::istream &io, data_structure::Queue<T> &m, DeserializeNumInterface &sn) {
        m.deserialize(io, sn);
    }
}