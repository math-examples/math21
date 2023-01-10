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

#include "inner.h"
#include "State.h"
#include "tic_tac_toe.h"

namespace math21 {
    namespace rl {
        namespace tic_tac_toe {

            void State::init() {
                NumN nr, nc;
                getParasShape(nr, nc);
                mMat.setSize(nr, nc);

                math21_operator_container_set_num(mMat, 0);

                h = -1;
                w = 0;
                end = -1;
            }

            void State::copyTo(State &B) const {
                B.mMat.setSize(mMat.nrows(), mMat.ncols());
                math21_operator_container_set_partially(mMat, B.mMat);

                B.h = h;
                B.w = w;
                B.end = end;
            }

            State::State() {
                init();
            }

            // Copy constructor
            State::State(const State &B) {
//                MATH21_ASSERT(0, "State Copy constructor.");
                B.copyTo(*this);
            }

            State &State::operator=(const State &B) {
                if (this != &B) {
                    B.copyTo(*this);
                }
                return *this;
            }

#ifdef MATH21_FLAG_USE_RVALUE_REF
            State::State(State &&B) {
                MATH21_ASSERT(0, "State move constructor.");
                M.swap(B.M);
                h = B.h;
                w = B.w;
                end = B.end;
                B.clear();
            }
#endif

            State::~State() {
                clear();
            }

            void State::clear() {
            }

            // reset
            void State::reset() {
                init();
            }

            NumZ State::calHashValue() {
                if (h != -1) {
                    return h;
                }
                NumN x = 0;
                for (NumN i = 1; i <= mMat.nrows(); ++i) {
                    for (NumN j = 1; j <= mMat.ncols(); ++j) {
                        x = x * 3 + (mMat(i, j) + 1);
                    }
                }
                h = x;
                return h;
            }

            void State::calEnd() {
                if (end != -1) {
                    return;
                }
                MatZ M;
                mMat.copyTo(M); // todo: share data

                SequenceZ re;
                for (NumN i = 1; i <= M.nrows(); ++i) {
                    re.add(math21_operator_matrix_sum_row_i(M, i));
                }
                for (NumN i = 1; i <= M.ncols(); ++i) {
                    re.add(math21_operator_matrix_sum_col_i(M, i));
                }

                // check diagonal if M is square
                if (M.nrows() == M.ncols()) {
                    re.add(math21_operator_matrix_trace(M));
                    re.add(math21_operator_matrix_reverse_trace(M));
                }

                NumZ x0;
                for (NumN i = 1; i <= re.size(); ++i) {
                    NumZ x = re(i);
                    if (i <= M.nrows()) {
                        x0 = M.ncols();
                    } else {
                        x0 = M.nrows();
                    }
                    if (x == x0) {
                        w = 1;
                        end = 1;
                        return;
                    }
                    if (x == -x0) {
                        w = -1;
                        end = 1;
                        return;
                    }
                }

                //# whether it's a tie
                NumZ sv = (NumZ) math21_operator_norm(M, 1);
                if (sv == M.size()) {
                    w = 0;
                    end = 1;
                    return;
                }

                //# game is still going on
                end = 0;
            }

            NumZ State::hashValue() const {
                MATH21_ASSERT(h != -1, "call calHashValue first!");
                return h;
            }

            NumB State::isEnd() const {
                MATH21_ASSERT(end != -1, "call calEnd first!");
                return (NumB) end;
            }

            NumZ State::getWinner() const {
                return w;
            }

            // symbol sb.
            void State::nextState(NumN i, NumN j, NumZ sb, State &s_prime) const {
                s_prime.reset();
                math21_operator_container_set_partially(mMat, s_prime.mMat);
                MATH21_ASSERT(s_prime.mMat(i, j) == 0, "input error!")
                s_prime.mMat(i, j) = sb;
            }

            void State::log(const char *name) const {
                log(std::cout, name);
            }

            void State::log(std::ostream &io, const char *name) const {
                if (name) {
                    io << "State " << name << "\n";
                    io << "isEnd: " << end << "\n";
                    io << "h: " << h << "\n";
                }
                for (NumN i = 1; i <= mMat.nrows(); ++i) {
                    io << getParasBorder() << "\n";
                    std::string out = "| ";
                    for (NumN j = 1; j <= mMat.ncols(); ++j) {
                        std::string token;
                        if (this->operator()(i, j) == 1) {
                            token = "O";
                        } else if (this->operator()(i, j) == -1) {
                            token = "X";
                        } else {
                            token = " ";
                        }
                        out += token + " | ";
                    }
                    io << out << "\n";
                }
                io << getParasBorder() << "\n";
            }

            void State::serialize(std::ostream &io, SerializeNumInterface &sn) const {
                math21_io_serialize(io, mMat, sn);
                math21_io_serialize(io, h, sn);
                math21_io_serialize(io, w, sn);
                math21_io_serialize(io, end, sn);
            }

            void State::deserialize(std::istream &io, DeserializeNumInterface &sn) {
                clear();
                math21_io_deserialize(io, mMat, sn);
                math21_io_deserialize(io, h, sn);
                math21_io_deserialize(io, w, sn);
                math21_io_deserialize(io, end, sn);
            }

            std::ostream &operator<<(std::ostream &io, const State &s) {
                s.log(io);
                return io;
            }
        }
    }
}