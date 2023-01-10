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

#include "Fn.h"

namespace math21 {
    class FnRoute {
    public:
        const char *name;
        int mini_batch_size;
        VecN inputLayers;
        FnMatType y; // Y, with batch
        FnMatType dy; // dL/dY
    public:
        FnRoute();

        virtual ~FnRoute();

        void init();

        void create(const mlfunction_net *net,
                    int mini_batch_size, const VecN &inputLayers, NumN deviceType);

        void resize(const mlfunction_net *net);

        void log(const char *varName) const;

        void forward(mlfunction_net *net);

        void backward(mlfunction_net *net);

        void saveState(FILE *file) const;
    };

}