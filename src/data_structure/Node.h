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

    namespace data_structure {

// helper linked list class
        template<typename Item>
        class Node {
        public:
            Node() {
                next = 0;
            }

            Item item;
            Node<Item> *next;
        };

        template<typename Item>
        class ListIterator {
        private:
            Node<Item> *current;
        public:
            ListIterator() {
                current = 0;
            }

            ListIterator(const ListIterator &i) {
                assign(i);
            }

            ListIterator &operator=(const ListIterator &i) {
                assign(i);
                return *this;
            }

            ListIterator &assign(const ListIterator &i) {
                current = i.current;
                return *this;
            }

            ListIterator(Node<Item> *first) {
                current = first;
            }

            virtual ~ListIterator() {
                current = 0;
            }

            bool isEmpty() const {
                return current == 0;
            }

            bool hasNext() const {
                return current != 0;
            }

            void remove() {
                MATH21_ASSERT(0, "UnsupportedOperationException")
            }

            Item next() {
                MATH21_ASSERT(hasNext(), "NoSuchElementException")
                Item item = current->item;
                current = current->next;
                return item;
            }

            void log(const char *name = 0) const {
                log(std::cout, name);
            }

            void log(std::ostream &io, const char *name = 0) const {
                if (name) {
                    io << "ListIterator: " << name << "\n";
                    if (current) {
                        io << current << "\n";
                    } else {
                        io << "null pointer\n";
                    }
                }
                ListIterator<Item> i(*this);
                while (i.hasNext()) {
                    auto w = i.next();
                    io << w << " ";
                }
            }
        };
    }
}