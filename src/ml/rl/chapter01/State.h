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

#include "inner_header.h"

namespace math21 {
    namespace rl {
        namespace tic_tac_toe {

            struct State {
            private:
                LightMatZ mMat;
                NumZ h; // hash value
                NumZ w; // winner
                NumZ end;

                void init();

            public:

                const NumZ &operator()(NumN j1, NumN j2) const {
                    return mMat.operator()(j1, j2);
                }

                void copyTo(State &B) const;

                State();

                // Copy constructor
                State(const State &B);

                State &operator=(const State &B);

#ifdef MATH21_FLAG_USE_RVALUE_REF
                State(State &&B);
#endif

                virtual ~State();

                // clear is used in program level.
                void clear();

                // reset is used in algorithm level.
                void reset();

                NumZ calHashValue();

                void calEnd();

                NumZ hashValue() const;

                NumB isEnd() const;

                NumZ getWinner() const;

                // symbol sb.
                void nextState(NumN i, NumN j, NumZ sb, State &s_prime) const;

                void log(const char *name = 0) const;

                void log(std::ostream &io, const char *name = 0) const;

                void serialize(std::ostream &io, SerializeNumInterface &sn) const;

                void deserialize(std::istream &io, DeserializeNumInterface &sn);
            };

            std::ostream &operator<<(std::ostream &io, const State &s);
        }
    }
}