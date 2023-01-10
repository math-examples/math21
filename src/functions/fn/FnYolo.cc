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
#include "box.h"
#include "FnYolo.h"

namespace math21 {

    FnYolo::FnYolo() {
        init();
    }

    FnYolo::~FnYolo() {
        auto f = this;
        if (f->map) {
            math21_vector_free_cpu(f->map);
        }
        math21_vector_free_cpu(f->cost);
        math21_vector_free_cpu(f->mask);
        math21_vector_free_cpu(f->biases);
        math21_vector_free_cpu(f->output);
        math21_vector_free_cpu(f->delta);
#ifndef MATH21_FLAG_USE_CPU
        math21_vector_free_wrapper(f->output_gpu);
        math21_vector_free_wrapper(f->delta_gpu);
#endif
    }

    void FnYolo::init() {
        name = 0;
        is_this_type = 0;
        batch = 0;
        h = 0;
        w = 0;
        c = 0;
        out_h = 0;
        out_w = 0;
        out_c = 0;
        n = 0;
        total = 0;
        classes = 0;
        cost = 0;
        mask = 0;
        biases = 0;
        outputs = 0;
        inputs = 0;
        truths = 0;
        output = 0;
        output_gpu = 0;
        delta = 0;
        delta_gpu = 0;
        max_boxes = 0;
        ignore_thresh = 0;
        truth_thresh = 0;
        onlyforward = 0;
        map = 0;
        net_train = 0;
        net_truth = 0;
        net_h = 0;
        net_w = 0;
        net_index = 0;
        jitter = 0;
        random = 0;
    }

    void FnYolo::create(int mini_batch_size, int nc_grids, int nr_grids, int num_box,
                        int total_prior, int *prior_mask, int num_class, int max_boxes) {
        int i;
        auto f = this;
        f->is_this_type = 1;
        f->batch = mini_batch_size;
        f->c = num_box * (num_class + 4 + 1);
        f->h = nr_grids;
        f->w = nc_grids;
        f->out_c = f->c;
        f->out_h = f->h;
        f->out_w = f->w;
        f->n = num_box;
        f->total = total_prior;
        f->classes = num_class;
        f->cost = (float *) math21_vector_calloc_cpu(1, sizeof(float));
        if (prior_mask) f->mask = prior_mask;
        else {
            f->mask = (int *) math21_vector_calloc_cpu(num_box, sizeof(int));
            for (i = 0; i < num_box; ++i) {
                f->mask[i] = i;
            }
        }
        // num_box used only, not total_prior
        f->biases = (float *) math21_vector_calloc_cpu(total_prior * 2, sizeof(float));
        f->outputs = f->out_h * f->out_w * f->out_c;
        f->inputs = f->outputs;
        f->max_boxes = max_boxes;
        f->truths = f->max_boxes * (4 + 1);
        for (i = 0; i < total_prior * 2; ++i) {
            f->biases[i] = .5;
        }

        f->output = math21_vector_create_with_default_value_cpu(mini_batch_size * f->outputs, 0);
        f->delta = math21_vector_create_with_default_value_cpu(mini_batch_size * f->outputs, 0);

#ifndef MATH21_FLAG_USE_CPU
        f->output_gpu = math21_vector_create_with_default_value_wrapper(mini_batch_size * f->outputs, 0);
        f->delta_gpu = math21_vector_create_with_default_value_wrapper(mini_batch_size * f->outputs, 0);
#endif

        math21_c_seed_random_generator(212121);

        f->name = math21_string_create_from_string("yolo");
    }

    void FnYolo::resize(int h, int w) {
        auto f = this;
        f->h = h;
        f->w = w;
        f->out_h = f->h;
        f->out_w = f->w;

        f->outputs = f->w * f->h * f->c;
        f->inputs = f->outputs;

        f->output = math21_vector_resize_with_default_value_cpu(f->output, f->batch * f->outputs, 0);
        f->delta = math21_vector_resize_with_default_value_cpu(f->delta, f->batch * f->outputs, 0);

#ifndef MATH21_FLAG_USE_CPU
        f->output_gpu = math21_vector_resize_with_default_value_wrapper(f->output_gpu, f->batch * f->outputs, 0);
        f->delta_gpu = math21_vector_resize_with_default_value_wrapper(f->delta_gpu, f->batch * f->outputs, 0);
#endif
    }

    void FnYolo::log(const char *varName) const {
        fprintf(stdout, "yolo\n");
    }

    // x_ratio = (x_pred + ix_grids)/nx_grids
// y_ratio = (y_pred + iy_grids)/ny_grids
// nr_ratio = exp(nr_pred + logb) / nr_input
// nc_ratio = exp(nc_pred + logb) / nc_input
    mlbox
    _math21_ml_function_yolo_get_box(const float *x, const float *b, int ib, int index, int ic_grids, int ir_grids,
                                     int nc_grids, int nr_grids, int nc_input, int nr_input, int stride) {
        mlbox mbox;
        mbox.x = (x[index + 0 * stride] + ic_grids) / nc_grids;
        mbox.y = (x[index + 1 * stride] + ir_grids) / nr_grids;
        mbox.w = expf(x[index + 2 * stride]) * b[2 * ib] / nc_input;
        mbox.h = expf(x[index + 3 * stride]) * b[2 * ib + 1] / nr_input;
        return mbox;
    }

    float _math21_ml_function_yolo_box_backward(mlbox box_true, float *x, float *b, int ib,
                                                int index, int ic_grids, int ir_grids, int nc_grids, int nr_grids,
                                                int nc_input, int nr_input,
                                                float *dx, float scale, int stride) {
        mlbox box_pred = _math21_ml_function_yolo_get_box(x, b, ib, index, ic_grids, ir_grids, nc_grids, nr_grids,
                                                          nc_input,
                                                          nr_input, stride);
        float iou = math21_ml_box_iou(box_pred, box_true);

        float tx = (box_true.x * nc_grids - ic_grids);
        float ty = (box_true.y * nr_grids - ir_grids);
        float tw = logf(box_true.w * nc_input / b[2 * ib]);
        float th = logf(box_true.h * nr_input / b[2 * ib + 1]);

        dx[index + 0 * stride] = scale * (tx - x[index + 0 * stride]);
        dx[index + 1 * stride] = scale * (ty - x[index + 1 * stride]);
        dx[index + 2 * stride] = scale * (tw - x[index + 2 * stride]);
        dx[index + 3 * stride] = scale * (th - x[index + 3 * stride]);
        return iou;
    }

// class_true in [0, num_class)
    void
    _math21_ml_function_yolo_class_backward(const float *Y, float *dY, int index, int class_true, int num_class,
                                            int stride,
                                            float *avg_class_pr) {
        int i;
        // if dy(0) != 0, i.e., the predicted box has been responsible for some actual boxes.
        if (dY[index]) {
            dY[index + class_true * stride] = 1 - Y[index + class_true * stride];
            if (avg_class_pr) *avg_class_pr += Y[index + class_true * stride];
            return;
        }
        // if dy(0) = 0, i.e., the predicted box has never been responsible for any actual boxes.
        for (i = 0; i < num_class; ++i) {
            dY[index + i * stride] = ((i == class_true) ? 1 : 0) - Y[index + i * stride];
            if (i == class_true && avg_class_pr) *avg_class_pr += Y[index + class_true * stride];
        }
    }

// X has shape nb * np * (4 + 1 + num_class) * nr * nc
// get        (imb, ibox, offset, ir_grids, ic_grids)
    int _math21_ml_function_yolo_get_box_related_index(FnYolo *l, int imb, int ibox, int offset, int igrid) {
//    int ibox = location / (l->w * l->h);
//    int igrid = location % (l->w * l->h);
        return imb * l->outputs + ibox * (4 + l->classes + 1) * l->w * l->h + offset * l->w * l->h + igrid;
    }

    // get negative derivative, -dY
    void FnYolo::forwardDetailCpu() {
        auto l = this;
        int ir_grids, ic_grids, imb, t, ibox;

        // dY = 0
        memset(l->delta, 0, l->batch * l->outputs * sizeof(float));
        if (!l->net_train) return;
        float avg_iou = 0;
        float recall = 0;
        float recall75 = 0;
        float avg_class_pr = 0;
        float avg_obj = 0;
        float avg_anyobj = 0;
        int count = 0;
        *(l->cost) = 0;
        for (imb = 0; imb < l->batch; ++imb) {
            // for every predicted box get negative derivative
            for (ir_grids = 0; ir_grids < l->h; ++ir_grids) {
                for (ic_grids = 0; ic_grids < l->w; ++ic_grids) {
                    for (ibox = 0; ibox < l->n; ++ibox) {
                        // get box index
                        int box_index = _math21_ml_function_yolo_get_box_related_index(l, imb, ibox, 0,
                                                                                       ir_grids * l->w + ic_grids);
                        // get predicted box
                        mlbox box_pred = _math21_ml_function_yolo_get_box(l->output, l->biases, l->mask[ibox],
                                                                          box_index,
                                                                          ic_grids, ir_grids, l->w, l->h, l->net_w,
                                                                          l->net_h, l->w * l->h);
                        float best_iou = 0;
                        int best_t = 0;
                        for (t = 0; t < l->max_boxes; ++t) {
                            mlbox box_true = math21_ml_box_vector_to_box(l->net_truth + imb * l->truths + t * (4 + 1),
                                                                         1);
                            if (!box_true.x) break;
                            float iou = math21_ml_box_iou(box_pred, box_true);
                            if (iou > best_iou) {
                                best_iou = iou;
                                best_t = t;
                            }
                        }
                        // get obj index
                        int obj_index = _math21_ml_function_yolo_get_box_related_index(l, imb, ibox, 4,
                                                                                       ir_grids * l->w + ic_grids);
                        avg_anyobj += l->output[obj_index];
                        if (best_iou > l->ignore_thresh) {
                            // outlier point, no derivative
                            l->delta[obj_index] = 0;
                        } else {
                            // negative point
                            l->delta[obj_index] = 0 - l->output[obj_index];
                        }
                        // positive point
                        if (best_iou > l->truth_thresh) {
                            l->delta[obj_index] = 1 - l->output[obj_index];
                            int class_true = (int) l->net_truth[imb * l->truths + best_t * (4 + 1) + 4];
                            if (l->map) class_true = l->map[class_true];
                            int class_index = _math21_ml_function_yolo_get_box_related_index(l, imb, ibox, 4 + 1,
                                                                                             ir_grids * l->w +
                                                                                             ic_grids);
                            _math21_ml_function_yolo_class_backward(l->output, l->delta, class_index, class_true,
                                                                    l->classes, l->w * l->h, 0);
                            mlbox box_true = math21_ml_box_vector_to_box(
                                    l->net_truth + imb * l->truths + best_t * (4 + 1),
                                    1);
                            _math21_ml_function_yolo_box_backward(box_true, l->output, l->biases, l->mask[ibox],
                                                                  box_index,
                                                                  ic_grids, ir_grids, l->w, l->h, l->net_w, l->net_h,
                                                                  l->delta, (2 - box_true.w * box_true.h), l->w * l->h);
                        }
                    }
                }
            }

            // for every actual box find globally one and only one predicted box responsible for it.
            // get negative derivative
            for (t = 0; t < l->max_boxes; ++t) {
                mlbox box_true = math21_ml_box_vector_to_box(l->net_truth + imb * l->truths + t * (4 + 1), 1);

                if (!box_true.x) break;
                float best_iou = 0;
                int best_n = 0;
                ic_grids = (int) (box_true.x * l->w);
                ir_grids = (int) (box_true.y * l->h);
                mlbox box_true_shift = box_true;
                box_true_shift.x = box_true_shift.y = 0;
                // for all priors globally
                for (ibox = 0; ibox < l->total; ++ibox) {
                    mlbox box_prior = {0};
                    box_prior.w = l->biases[2 * ibox] / l->net_w;
                    box_prior.h = l->biases[2 * ibox + 1] / l->net_h;
                    float iou = math21_ml_box_iou(box_prior, box_true_shift);
                    if (iou > best_iou) {
                        best_iou = iou;
                        best_n = ibox;
                    }
                }

                int mask_n = math21_vector_argequal_int(l->mask, best_n, l->n);
                if (mask_n >= 0) {
                    int box_index = _math21_ml_function_yolo_get_box_related_index(l, imb, mask_n, 0,
                                                                                   ir_grids * l->w + ic_grids);
                    float iou = _math21_ml_function_yolo_box_backward(box_true, l->output, l->biases, best_n, box_index,
                                                                      ic_grids, ir_grids, l->w, l->h, l->net_w,
                                                                      l->net_h,
                                                                      l->delta, (2 - box_true.w * box_true.h),
                                                                      l->w * l->h);

                    int obj_index = _math21_ml_function_yolo_get_box_related_index(l, imb, mask_n, 4,
                                                                                   ir_grids * l->w + ic_grids);
                    avg_obj += l->output[obj_index];
                    l->delta[obj_index] = 1 - l->output[obj_index];

                    int class_true = (int) l->net_truth[imb * l->truths + t * (4 + 1) + 4];
                    if (l->map) class_true = l->map[class_true];
                    int class_index = _math21_ml_function_yolo_get_box_related_index(l, imb, mask_n, 4 + 1,
                                                                                     ir_grids * l->w + ic_grids);
                    _math21_ml_function_yolo_class_backward(l->output, l->delta, class_index, class_true, l->classes,
                                                            l->w * l->h, &avg_class_pr);

                    ++count;
                    if (iou > .5) recall += 1;
                    if (iou > .75) recall75 += 1;
                    avg_iou += iou;
                }
            }

        }
        *(l->cost) = powf(math21_vector_norm_2_float(l->delta, l->batch * l->outputs), 2);
        if (0) {
            printf("layer %d, avg iou: %f, avg_class_pr: %f, avg_obj: %f, avg_anyobj: %f, 0.5R: %f, 0.75R: %f, count: %d\n",
                   l->net_index,
                   avg_iou / count, avg_class_pr / count, avg_obj / count, avg_anyobj / (l->batch * l->w * l->h * l->n),
                   recall / count, recall75 / count, count);
        }
    }

    void math21_ml_function_yolo_correct_boxes(mldetection *dets, int n, int data_nc, int data_nr, int netw, int neth,
                                               int relative) {
        int i;
        int new_w = 0;
        int new_h = 0;
        if (((float) netw / data_nc) < ((float) neth / data_nr)) {
            new_w = netw;
            new_h = (data_nr * netw) / data_nc;
        } else {
            new_h = neth;
            new_w = (data_nc * neth) / data_nr;
        }
        for (i = 0; i < n; ++i) {
            mlbox b = dets[i].bbox;
            b.x = (b.x - (netw - new_w) / 2. / netw) / ((float) new_w / netw);
            b.y = (b.y - (neth - new_h) / 2. / neth) / ((float) new_h / neth);
            b.w *= (float) netw / new_w;
            b.h *= (float) neth / new_h;
            if (!relative) {
                b.x *= data_nc;
                b.w *= data_nc;
                b.y *= data_nr;
                b.h *= data_nr;
            }
            dets[i].bbox = b;
        }
    }

    int FnYolo::getDetectionNum(float thresh) {
        auto l = this;
        int i, n;
        int count = 0;
        for (i = 0; i < l->w * l->h; ++i) {
            for (n = 0; n < l->n; ++n) {
                int obj_index = _math21_ml_function_yolo_get_box_related_index(l, 0, n, 4, i);
                if (l->output[obj_index] > thresh) {
                    ++count;
                }
            }
        }
        return count;
    }

    void _math21_ml_function_yolo_avg_flipped(FnYolo *l) {
        int i, j, n, z;
        float *flip = l->output + l->outputs;
        for (j = 0; j < l->h; ++j) {
            for (i = 0; i < l->w / 2; ++i) {
                for (n = 0; n < l->n; ++n) {
                    for (z = 0; z < l->classes + 4 + 1; ++z) {
                        int i1 = z * l->w * l->h * l->n + n * l->w * l->h + j * l->w + i;
                        int i2 = z * l->w * l->h * l->n + n * l->w * l->h + j * l->w + (l->w - i - 1);
                        float swap = flip[i1];
                        flip[i1] = flip[i2];
                        flip[i2] = swap;
                        if (z == 0) {
                            flip[i1] = -flip[i1];
                            flip[i2] = -flip[i2];
                        }
                    }
                }
            }
        }
        for (i = 0; i < l->outputs; ++i) {
            l->output[i] = (l->output[i] + flip[i]) / 2.;
        }
    }

    int FnYolo::getDetections(
            int data_nc, int data_nr, int netw, int neth, float thresh,
            int relative, mldetection *dets) {
        auto l = this;
        int i, j, n;
        float *predictions = l->output;
        if (l->batch == 2) _math21_ml_function_yolo_avg_flipped(l);
        int count = 0;
        for (i = 0; i < l->w * l->h; ++i) {
            int row = i / l->w;
            int col = i % l->w;
            for (n = 0; n < l->n; ++n) {
                int obj_index = _math21_ml_function_yolo_get_box_related_index(l, 0, n, 4, i);
                float objectness = predictions[obj_index];
                if (xjisnan(objectness)) {
                    m21warn("objectness is nan");
                }
                if (objectness <= thresh) continue;
                int box_index = _math21_ml_function_yolo_get_box_related_index(l, 0, n, 0, i);
                dets[count].bbox = _math21_ml_function_yolo_get_box(predictions, l->biases, l->mask[n], box_index, col,
                                                                    row,
                                                                    l->w, l->h, netw, neth,
                                                                    l->w * l->h);
                dets[count].objectness = objectness;
                dets[count].classes = l->classes;
                for (j = 0; j < l->classes; ++j) {
                    int class_index = _math21_ml_function_yolo_get_box_related_index(l, 0, n, 4 + 1 + j, i);
                    float prob = objectness * predictions[class_index];
                    dets[count].prob[j] = (prob > thresh) ? prob : 0;
                }
                ++count;
            }
        }
        math21_ml_function_yolo_correct_boxes(dets, count, data_nc, data_nr, netw, neth, relative);
        return count;
    }

#ifdef MATH21_FLAG_USE_CPU
    void FnYolo::forwardCpu(mlfunction_node *finput) {
        auto l = this;
    int ir_grids, ic_grids, imb, t, ibox;

    // Y <- X
    memcpy(l->output, finput->y, l->batch * l->outputs * sizeof(float));

    // Y <- logistic(Y), excluding nr, nc ingredient
    for (imb = 0; imb < l->batch; ++imb) {
        for (ibox = 0; ibox < l->n; ++ibox) {
            int index = _math21_ml_function_yolo_get_box_related_index(l, imb, ibox, 0, 0);
            math21_function_activation_vector_wrapper(l->output + index, 2 * l->w * l->h,
                                                      MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC);
            index = _math21_ml_function_yolo_get_box_related_index(l, imb, ibox, 4, 0);
            math21_function_activation_vector_wrapper(l->output + index, (1 + l->classes) * l->w * l->h,
                                                      MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC);
        }
    }

    l->forwardDetailCpu();
}

void FnYolo::backwardCpu(mlfunction_node *finput) {
        auto l = this;
    math21_vector_kx_add_y_wrapper(l->batch * l->inputs, 1, l->delta, 1, finput->dy, 1);
}
#endif


#ifndef MATH21_FLAG_USE_CPU

    void FnYolo::forwardGpu(mlfunction_node *finput) {
        auto l = this;
        // Y <- X
        math21_vector_assign_from_vector_wrapper(l->batch * l->inputs, finput->y, 1, l->output_gpu, 1);

        // Y <- logistic(Y), excluding nr, nc ingredient
        int imb, ibox;
        for (imb = 0; imb < l->batch; ++imb) {
            for (ibox = 0; ibox < l->n; ++ibox) {
                int index = _math21_ml_function_yolo_get_box_related_index(l, imb, ibox, 0, 0);
                math21_function_activation_vector_wrapper(l->output_gpu + index, 2 * l->w * l->h,
                                                          MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC);
                index = _math21_ml_function_yolo_get_box_related_index(l, imb, ibox, 4, 0);
                math21_function_activation_vector_wrapper(l->output_gpu + index, (1 + l->classes) * l->w * l->h,
                                                          MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC);
            }
        }

        if (!l->net_train || l->onlyforward) {
            math21_vector_pull_wrapper(l->output_gpu, l->output, l->batch * l->outputs);
            return;
        }

        // X <- Y
        math21_vector_pull_wrapper(l->output_gpu, l->output, l->batch * l->inputs);

        l->forwardDetailCpu();

        math21_vector_push_wrapper(l->delta_gpu, l->delta, l->batch * l->outputs);
    }

    void FnYolo::backwardGpu(mlfunction_node *finput) {
        auto l = this;
        math21_vector_kx_add_y_wrapper(l->batch * l->inputs, 1, l->delta_gpu, 1, finput->dy, 1);
    }

#endif

    void FnYolo::forward(mlfunction_node *finput) {
        auto l = this;
        if (l->net_train) {
#if defined(MATH21_FLAG_USE_CPU)
            math21_vector_set_wrapper(l->batch * l->outputs, 0, l->delta, 1);
#else
            math21_vector_set_wrapper(l->batch * l->outputs, 0, l->delta_gpu, 1);
#endif
        }
#if defined(MATH21_FLAG_USE_CPU)
        l->forwardCpu(finput);
#else
        l->forwardGpu(finput);
#endif
    }

    void FnYolo::backward(mlfunction_node *finput) {
        auto l = this;
#if defined(MATH21_FLAG_USE_CPU)
        l->backwardCpu(finput);
#else
        l->backwardGpu(finput);
#endif
    }

    void FnYolo::saveState(FILE *file) const {
        auto f = this;
#if defined(MATH21_FLAG_USE_CPU)
        math21_vector_serialize_c_wrapper(file, f->output, f->batch * f->outputs);
    math21_vector_serialize_c_wrapper(file, f->delta, f->batch * f->outputs);
#else
        math21_vector_serialize_c_wrapper(file, f->output_gpu, f->batch * f->outputs);
        math21_vector_serialize_c_wrapper(file, f->delta_gpu, f->batch * f->outputs);
#endif
    }
}
