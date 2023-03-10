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
    template<typename StackType, typename VecType>
    void math21_data_structure_stack_type_convert_to_vector_type(const StackType &s, VecType &v) {
        if (s.isEmpty()) {
            v.clear();
            return;
        }
        v.setSize(s.size());
        NumN k = 1;
        auto iterator = s.getIterator();
        while (iterator.hasNext()) {
            v(k) = iterator.next();
            ++k;
        }
    }

    namespace data_structure {

        template<typename Item>
        class Bag {
        private:
            Node<Item> *first;    // beginning of bag
            NumN n;               // number of elements in bag

            Item pop() {
                MATH21_ASSERT(!isEmpty(), "Bag underflow")
                Item item = first->item;        // save item to return
                delete first;
                first = first->next;            // delete first node
                n--;
                return item;                   // return the saved item
            }

        public:
            Bag() {
                first = 0;
                n = 0;
            }

            virtual ~Bag() {
                while (!isEmpty()) {
                    pop();
                }
            }

            bool isEmpty() const {
                return first == 0;
            }

            NumN size() const {
                return n;
            }

            void add(Item item) {
                Node<Item> *oldfirst = first;
                first = new Node<Item>();
                first->item = item;
                first->next = oldfirst;
                n++;
            }

            void log(const char *name = 0) const {
                log(std::cout, name);
            }

            void log(std::ostream &io, const char *name = 0) const {
                if (name) {
                    io << "Bag: " << name << "\n";
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
        };

        template<typename Item>
        class Stack {
        private:
            Node<Item> *first;     // top of stack
            NumN n;                // size of the stack
        public:
            Stack() {
                first = 0;
                n = 0;
            }

            virtual ~Stack() {
                while (!isEmpty()) {
                    pop();
                }
            }

            bool isEmpty() const {
                return first == 0;
            }

            NumN size() const {
                return n;
            }

            void push(Item item) {
                Node<Item> *oldfirst = first;
                first = new Node<Item>();
                first->item = item;
                first->next = oldfirst;
                ++n;
            }

            Item pop() {
                MATH21_ASSERT(!isEmpty(), "Stack underflow");
                Item item = first->item;        // save item to return
                auto oldfirst = first->next;
                delete first;            // delete first node
                first = oldfirst;
                --n;
                return item;                   // return the saved item
            }

            Item peek() const {
                MATH21_ASSERT(!isEmpty(), "Stack underflow")
                return first->item;
            }

            void log(const char *name = 0) const {
                log(std::cout, name);
            }

            void log(std::ostream &io, const char *name = 0) const {
                if (name) {
                    io << "Stack: " << name << "\n";
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
        };
    }
}