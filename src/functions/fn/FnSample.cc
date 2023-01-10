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

#include "inner_cc.h"
#include "FnSample.h"

namespace math21 {

    FnSample::FnSample() {
        init();
    }

    FnSample::~FnSample() {}

    void FnSample::init() {
        name = 0;
        batch = 0;
        stride = 0;
        h = 0, w = 0, c = 0;
        out_h = 0, out_w = 0, out_c = 0;
        inputs = 0;
        outputs = 0;
        delta = 0;
        output = 0;
        reverse = 0;
        scale = 0;
    }

    void FnSample::create(int mini_batch_size, int nc, int nr, int nch, int stride) {
        auto *f = this;
        f->batch = mini_batch_size;
        f->w = nc;
        f->h = nr;
        f->c = nch;
        f->out_w = nc * stride;
        f->out_h = nr * stride;
        f->out_c = nch;
        if (stride < 0) {
            stride = -stride;
            f->reverse = 1;
            f->out_w = nc / stride;
            f->out_h = nr / stride;
        }
        f->stride = stride;
        f->outputs = f->out_w * f->out_h * f->out_c;
        f->inputs = f->w * f->h * f->c;
        f->output = math21_vector_create_with_default_value_wrapper(mini_batch_size * f->outputs, 0);
        f->delta = math21_vector_create_with_default_value_wrapper(mini_batch_size * f->outputs, 0);
        if (f->reverse) {
            f->name = math21_string_create_from_string("sumdownsample");
        } else {
            f->name = math21_string_create_from_string("upsample");
        }
    }

    void FnSample::resize(int nr, int nc) {
        auto *f = this;
        f->w = nc;
        f->h = nr;
        f->out_w = nc * f->stride;
        f->out_h = nr * f->stride;
        if (f->reverse) {
            f->out_w = nc / f->stride;
            f->out_h = nr / f->stride;
        }
        f->outputs = f->out_w * f->out_h * f->out_c;
        f->inputs = f->h * f->w * f->c;
        f->output = math21_vector_resize_with_default_value_wrapper(f->output, f->outputs * f->batch, 0);
        f->delta = math21_vector_resize_with_default_value_wrapper(f->delta, f->outputs * f->batch, 0);
    }

    void FnSample::log(const char *varName) const {
        auto *f = this;
        fprintf(stdout, "%s                %2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", f->name, f->stride, f->w, f->h,
                f->c,
                f->out_w, f->out_h, f->out_c);
    }

    void FnSample::forward(mlfunction_node *finput, NumB is_train) {
        auto *f = this;
        if (is_train) {
            math21_vector_set_wrapper(f->batch * f->outputs, 0, f->delta, 1);
        }
        math21_vector_set_wrapper(f->outputs * f->batch, 0, f->output, 1);
        if (f->reverse) {
            math21_vector_feature2d_sample_wrapper(
                    f->batch, f->output, f->c, f->out_h,
                    f->out_w, f->stride, 0, f->scale, finput->y);
        } else {
            math21_vector_feature2d_sample_wrapper(f->batch, finput->y, f->c, f->h, f->w, f->stride, 1, f->scale,
                                                   f->output);
        }
    }

    void FnSample::backward(mlfunction_node *finput) {
        auto *f = this;
        if (f->reverse) {
            math21_vector_feature2d_sample_wrapper(f->batch, f->delta, f->c, f->out_h, f->out_w, f->stride, 1, f->scale,
                                                   finput->dy);
        } else {
            math21_vector_feature2d_sample_wrapper(f->batch, finput->dy, f->c, f->h, f->w, f->stride, 0, f->scale,
                                                   f->delta);
        }
    }

    void FnSample::saveState(FILE *file) const {
        auto *f = this;
        math21_vector_serialize_c_wrapper(file, f->output, f->batch * f->outputs);
        math21_vector_serialize_c_wrapper(file, f->delta, f->batch * f->outputs);
    }

}