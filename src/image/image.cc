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

#include "../matrix_op/files.h"
#include "../functions/files.h"
#include "inner_src.h"
#include "image_c.h"
#include "image.h"

namespace math21 {
    //r, c are dst size. e.x., (0, 100] -> (0, 1000].
    void math21_img_resize_method_sampling(const MatR &src, MatR &dst) {
        MATH21_ASSERT(!src.isEmpty(), "src matrix is empty!");
        MATH21_ASSERT(!dst.isEmpty(), "dst matrix is empty!");
        MATH21_ASSERT((src.dims() == 2 && dst.dims() == 2) ||
                      (src.dims() == 3 && dst.dims() == 3 && src.dim(1) == dst.dim(1)),
                      "" << src.shape().log("src")
                         << dst.shape().log("dst")
        );
        NumN r, c;
        NumN src_r, src_c;
        NumN nch;
        if (src.dims() == 2) {
            r = dst.dim(1);
            c = dst.dim(2);
            src_r = src.dim(1);
            src_c = src.dim(2);
        } else {
            r = dst.dim(2);
            c = dst.dim(3);
            src_r = src.dim(2);
            src_c = src.dim(3);
            nch = src.dim(1);
        }
        NumN i, j, k;
        NumN src_i, src_j;

        NumR rs, cs;
        rs = src_r / (NumR) r;
        cs = src_c / (NumR) c;
        if (src.dims() == 2) {
            for (i = 1; i <= r; i++) {
                src_i = (NumN) (i * rs);
                math21_clip_not_less(src_i, 1);
                for (j = 1; j <= c; j++) {
                    src_j = (NumN) (j * cs);
                    math21_clip_not_less(src_j, 1);
                    dst(i, j) = src(src_i, src_j);
                }
            }
        } else {
            for (i = 1; i <= r; i++) {
                src_i = (NumN) (i * rs);
                math21_clip_not_less(src_i, 1);
                for (j = 1; j <= c; j++) {
                    src_j = (NumN) (j * cs);
                    math21_clip_not_less(src_j, 1);
                    for (k = 1; k <= nch; ++k) {
                        dst(k, i, j) = src(k, src_i, src_j);
                    }
                }
            }
        }
    }

    // we use max pooling
    void math21_img_resize_method_pooling(const MatR &src, MatR &dst) {
        MATH21_ASSERT(!src.isEmpty(), "src matrix is empty!");
        MATH21_ASSERT(!dst.isEmpty(), "dst matrix is empty!");
        MATH21_ASSERT((src.dims() == 2 && dst.dims() == 2) ||
                      (src.dims() == 3 && dst.dims() == 3 && src.dim(1) == dst.dim(1)),
                      "" << src.shape().log("src")
                         << dst.shape().log("dst"));
        NumN r, c;
        NumN src_r, src_c;
        if (src.dims() == 2) {
            r = dst.dim(1);
            c = dst.dim(2);
            src_r = src.dim(1);
            src_c = src.dim(2);
        } else {
            r = dst.dim(2);
            c = dst.dim(3);
            src_r = src.dim(2);
            src_c = src.dim(3);
        }

        NumN mk, nk, ms, ns;
        math21_operator_ml_pooling_get_mk_ms(src_r, src_c, r, c, mk, nk, ms, ns);
        if (src.dims() == 2) {
            TenR src_share, dst_share;
            math21_operator_share_2d_to_3d(src, src_share);
            math21_operator_share_2d_to_3d(dst, dst_share);
            math21_operator_ml_pooling_valueAt(src_share, dst_share, cnn_type_pooling_max,
                                               mk, nk, ms, ns);

        } else {
            math21_operator_ml_pooling_valueAt(src, dst, cnn_type_pooling_max,
                                               mk, nk, ms, ns);

        }
    }

    void math21_img_resize_method_interpolation_bilinear_kernel(
            NumN n, const NumR *a, NumR *b,
            NumN a_nr, NumN a_nc,
            NumN b_nr, NumN b_nc, NumN b_nch, NumN id) {
        if (id > n) return;
        NumN ir, ic, ich;
        math21_device_index_1d_to_3d_fast(&ich, &ir, &ic, id, b_nr, b_nc);

        NumR ir_o = ((NumR) (ir - 1) / b_nr) * a_nr;
        NumR ic_o = ((NumR) (ic - 1) / b_nc) * a_nc;
        NumR val;
        NumB flag = math21_device_image_get_pixel_bilinear_interpolate(
                a, &val, ich, ir_o + 1, ic_o + 1,
                b_nch, a_nr, a_nc, 0);
        if (!flag) {
            val = 0;
        }
        math21_device_image_pixel_value_set(b, ir, ic, ich, b_nr, b_nc, b_nch, val);
    }

    void math21_img_resize_method_interpolation_bilinear(const MatR &A, MatR &B) {
        MATH21_ASSERT(!A.isEmpty(), "A matrix is empty!");
        MATH21_ASSERT(!B.isEmpty(), "B matrix is empty!");
        MATH21_ASSERT((A.dims() == 2 && B.dims() == 2) ||
                      (A.dims() == 3 && B.dims() == 3 && A.dim(1) == B.dim(1)),
                      "" << A.shape().log("A")
                         << B.shape().log("B"));
        NumN nr, nc, nch;
        NumN nr_A, nc_A;
        if (A.dims() == 2) {
            nch = 1;
            nr = B.dim(1);
            nc = B.dim(2);
            nr_A = A.dim(1);
            nc_A = A.dim(2);
        } else {
            nch = B.dim(1);
            nr = B.dim(2);
            nc = B.dim(3);
            nr_A = A.dim(2);
            nc_A = A.dim(3);
        }
        auto data_a = A.getDataAddress();
        auto data_b = B.getDataAddress();
        data_a -= 1;
        data_b -= 1;
        NumN n = nch * nr * nc;
        NumN id;
#pragma omp parallel for
        for (id = 1; id <= n; ++id) {
            math21_img_resize_method_interpolation_bilinear_kernel(n, data_a, data_b, nr_A, nc_A, nr, nc, nch, id);
        }
    }

    void math21_img_resize(const MatR &src, MatR &dst, NumN img_resize_method) {
        MATH21_ASSERT(!src.isEmpty(), "src matrix is empty!");
        MATH21_ASSERT(!dst.isEmpty(), "dst matrix is empty!");
        MATH21_ASSERT((src.dims() == 2 && dst.dims() == 2) ||
                      (src.dims() == 3 && dst.dims() == 3 && src.dim(1) == dst.dim(1)),
                      "" << src.shape().log("src")
                         << dst.shape().log("dst"));
        if (src.isSameSize(dst.shape())) {
            dst = src;
            return;
        }
        if (img_resize_method == m21_flag_resize_method_default ||
            img_resize_method == m21_flag_resize_method_interpolation_bilinear) {
            math21_img_resize_method_interpolation_bilinear(src, dst);
        } else if (img_resize_method == m21_flag_resize_method_sampling) {
            math21_img_resize_method_sampling(src, dst);
        } else if (img_resize_method == m21_flag_resize_method_pooling) {
            math21_img_resize_method_pooling(src, dst);
        }
    }

    void math21_img_resize(const Seqce<TenR> &srcs, Seqce<TenR> &dsts, NumN img_resize_method) {
        MATH21_ASSERT(srcs.size() == dsts.size())
        for (NumN i = 1; i <= srcs.size(); i++) {
            const TenR &src = srcs(i);
            TenR &dst = dsts.at(i);
            math21_img_resize(src, dst, img_resize_method);
        }
    }

    // just return if image already has size d.
    void math21_img_resize(Seqce<TenR> &images, const VecN &d, NumN img_resize_method) {
        for (NumN i = 1; i <= images.size(); i++) {
            TenR &x = images.at(i);
            math21_img_resize(x, d, img_resize_method);
        }
    }

    void math21_img_resize(TenR &image, const VecN &d, NumN img_resize_method) {
        if (image.isSameSize(d)) {
            return;
        }
        TenR tmp;
        tmp.setSize(d);
        math21_img_resize(image, tmp, img_resize_method);
        image.swap(tmp);
    }

    // convert all color images to gray images. Just return if image is already gray.
    void math21_img_rgb_to_gray(Seqce<TenR> &images) {
        for (NumN i = 1; i <= images.size(); i++) {
            TenR &x = images.at(i);
            math21_img_rgb_to_gray(x);
        }
    }

    // argb, rgb, etc.
    void math21_img_rgb_to_gray(MatR &image) {
        if (image.dims() == 2) {
            return;
        }
        MATH21_ASSERT(image.dims() == 3)
        TenR tmp;
        tmp.setSize(image.dim(2), image.dim(3));
        math21_img_rgb_to_gray(image, tmp);
        image.swap(tmp);
    }

    void math21_img_gray_to_binary(MatR &image) {
        MATH21_ASSERT(image.dims() == 2)
        TenR tmp;
        tmp.setSize(image.shape());
        math21_img_gray_to_binary(image, tmp);
        image.swap(tmp);
    }

    void math21_img_rgb_to_gray(const MatR &src, MatR &dst) {
        MATH21_ASSERT(!src.isEmpty(), "src matrix is empty!");
        MATH21_ASSERT(!dst.isEmpty(), "dst matrix is empty!");
        MATH21_ASSERT(src.dims() == 3)
        MATH21_ASSERT(dst.dims() == 2 && dst.isSameSize(src.dim(2), src.dim(3)));
        NumN r, c;
        NumN nch;
        r = dst.dim(1);
        c = dst.dim(2);
        nch = src.dim(1);

        NumN i, j, k;
        NumR sum;
        for (i = 1; i <= r; i++) {
            for (j = 1; j <= c; j++) {
                sum = 0;
                for (k = 1; k <= nch; ++k) {
                    sum += src(k, i, j);
                }
                dst(i, j) = sum / nch;
            }
        }
    }

    // histogram has shape: nch*256
    void math21_img_histogram(const MatR &src, MatR &histogram) {
        MATH21_ASSERT(!src.isEmpty(), "src matrix is empty!");
        MATH21_ASSERT(src.dims() == 3 || src.dims() == 2)
        NumN r, c;
        NumN nch;
        if (src.dims() == 3) {
            r = src.dim(2);
            c = src.dim(3);
            nch = src.dim(1);
        } else {
            nch = 1;
            r = src.dim(1);
            c = src.dim(2);
        }
        if (histogram.isEmpty()) {
            histogram.setSize(nch, 256);
        }
        if (!histogram.isEmpty()) {
            MATH21_ASSERT(histogram.isSameSize(nch, 256));
        }

        NumN i, j, k;
        NumN index;
        if (nch > 1) {
            for (i = 1; i <= r; i++) {
                for (j = 1; j <= c; j++) {
                    for (k = 1; k <= nch; ++k) {
                        index = (NumN) src(k, i, j) + 1;
                        ++histogram(k, index);
                    }
                }
            }
        } else {
            for (i = 1; i <= r; i++) {
                for (j = 1; j <= c; j++) {
                    index = (NumN) src(i, j) + 1;
                    ++histogram(1, index);
                }
            }
        }

    }

    void math21_img_cluster_by_value(const MatR &src, MatN &mask) {

    }

    // dst is mask with values in {1, 2, ...}
    void math21_img_gray_cluster_by_value(const MatR &src, MatN &dst, NumN K, VecN &num_in_clusters) {
        MATH21_ASSERT(!src.isEmpty(), "src matrix is empty!");
        MATH21_ASSERT(!dst.isEmpty(), "dst matrix is empty!");
        MATH21_ASSERT(src.dims() == 2)
        MATH21_ASSERT(dst.dims() == 2 && dst.isSameSize(src.shape()));
        NumN total_points = src.volume();

        VecN labels;
        VecR tmp_src;
        math21_operator_share_to_vector(dst, labels);
        math21_operator_share_to_vector(src, tmp_src);

        Seqce<TenR> data;
        math21_tool_vec_2_seqce(tmp_src, data);
#ifdef MATH21_ENABLE_ML
        ml_kmeans_config config(K, total_points, 1, 100);
        ml_kmeans(data, labels, num_in_clusters, config);
#endif
    }

    // background is 0, foreground is 255
    void math21_img_gray_to_binary(const MatR &src, MatR &dst) {
        MATH21_ASSERT(!src.isEmpty(), "src matrix is empty!");
        MATH21_ASSERT(!dst.isEmpty(), "dst matrix is empty!");
        MATH21_ASSERT(src.dims() == 2)
        MATH21_ASSERT(dst.dims() == 2 && dst.isSameSize(src.shape()));

        MatN mask;
        mask.setSize(src.shape());
        VecN num_in_clusters;
        math21_img_gray_cluster_by_value(src, mask, 2, num_in_clusters);
        NumN r, c;
        r = dst.dim(1);
        c = dst.dim(2);

        NumN bg_index;
        if (num_in_clusters(1) > num_in_clusters(2)) {
            bg_index = 1;
        } else {
            bg_index = 2;
        }
        NumN i, j;
        for (i = 1; i <= r; i++) {
            for (j = 1; j <= c; j++) {
                if (mask(i, j) == bg_index) {
                    dst(i, j) = 0;
                } else {
                    dst(i, j) = 255;
                }
            }
        }
    }

    void math21_img_gray_to_binary(Seqce<TenR> &images) {
        for (NumN i = 1; i <= images.size(); i++) {
            TenR &x = images.at(i);
            math21_img_gray_to_binary(x);
        }
    }

    namespace detail {
        // todo: optimize
        void math21_image_argb8888_to_tensor_cpu(const NumN32 *data, TenR &m, NumN nr, NumN nc, NumN nch) {
            MATH21_ASSERT(nch == 3 || nch == 4)
            if (m.isSameSize(nch, nr, nc) == 0) {
                m.setSize(nch, nr, nc);
            }
            NumN i2, i3;
            NumN32 a, r, g, b;
            NumN32 value;

            for (i2 = 1; i2 <= nr; ++i2) {
                for (i3 = 1; i3 <= nc; ++i3) {
                    value = data[(i2 - 1) * nc + (i3 - 1)];
                    a = (value >> 24) & 0xff;
                    r = (value >> 16) & 0xff;
                    g = (value >> 8) & 0xff;
                    b = (value) & 0xff;
                    m(1, i2, i3) = r;
                    m(2, i2, i3) = g;
                    m(3, i2, i3) = b;
                    if (nch == 4) {
                        m(4, i2, i3) = a;
                    }
                }
            }
        }
    }


    // read from NumN32
    void math21_image_argb8888_to_tensor(const NumN32 *data, TenR &m, NumN nr, NumN nc, NumN nch) {
        MATH21_ASSERT(m.is_cpu());
#if defined(MATH21_FLAG_USE_CUDA)
        detail::math21_image_argb8888_to_tensor_cuda(data, m, nr, nc, nch);
#else
        detail::math21_image_argb8888_to_tensor_cpu(data, m, nr, nc, nch);
#endif
    }

    namespace detail {
        void math21_image_tensor_to_argb8888(NumN32 *data, const TenR &m) {
            MATH21_ASSERT(m.dims() == 3 &&
                          (m.dim(1) == 3 || m.dim(1) == 4))

            NumN nr, nc, nch;
            nr = m.dim(2);
            nc = m.dim(3);
            nch = m.dim(1);

            NumN i2, i3;
            NumN32 a, r, g, b;
            NumN32 value;

            for (i2 = 1; i2 <= nr; ++i2) {
                for (i3 = 1; i3 <= nc; ++i3) {
                    r = (NumN32) math21_operator_number_clip(m(1, i2, i3), 0, 255);
                    g = (NumN32) math21_operator_number_clip(m(2, i2, i3), 0, 255);
                    b = (NumN32) math21_operator_number_clip(m(3, i2, i3), 0, 255);
                    if (nch == 3) {
                        value = 0xff000000 | (r << 16) | (g << 8) | b;
                    } else {
                        a = (NumN32) math21_operator_number_clip(m(4, i2, i3), 0, 255);
                        value = (a << 24) | (r << 16) | (g << 8) | b;
                    }
                    data[(i2 - 1) * nc + (i3 - 1)] = value;
                }
            }
        }
    }

    // maybe to do next: to read all argb data we saved, we store them as bgra no matter which endian is.
    // Note argb8888 is stored as bgra in little endian, as argb in big endian.
    // rgba to argb8888
    void math21_image_tensor_to_argb8888(NumN32 *data, const TenR &m) {
#if defined(MATH21_FLAG_USE_CPU)
        detail::math21_image_tensor_to_argb8888(data, m);
#elif defined(MATH21_FLAG_USE_CUDA)
        detail::math21_image_tensor_to_argb8888_cuda(data, m);
#elif defined(MATH21_FLAG_USE_OPENCL)
        MATH21_ASSERT(0)
#endif
    }

    // argb8888 to rgba
    NumB math21_io_argb8888_read(TenR &m, NumN nr, NumN nc, NumN nch, const char *path) {
        NumN32 *output_data;
        NumN8 *data;
        size_t width;
        size_t height;

        width = nc;
        height = nr;
        size_t size = width * height * sizeof(NumN32);

        data = (NumN8 *) calloc(size, 1);

        NumB flag;
        flag = math21_io_read_file(path, data, size);
        if (!flag) {
            free(data);
            return 0;
        }
        output_data = (NumN32 *) data;
        math21_image_argb8888_to_tensor(output_data, m, nr, nc, nch);
        free(data);
        return 1;
    }

    // rgba to argb8888
    NumB math21_io_argb8888_write(const TenR &m, const char *path) {
        NumN32 *output_data;
        NumN8 *data;

        MATH21_ASSERT(m.dims() == 3)
        NumN nr, nc;
        nr = m.dim(2);
        nc = m.dim(3);
        size_t size = nr * nc * sizeof(NumN32);
        data = (NumN8 *) calloc(size, 1);

        output_data = (NumN32 *) data;
        math21_image_tensor_to_argb8888(output_data, m);
        NumB flag;
        flag = math21_io_write_file(path, data, size);
        free(data);
        if (!flag) {
            return 0;
        }
        return 1;
    }

    // tensor rgba to tensor argb8888
    void math21_image_convert_to_argb8888(const TenR &m, Tensor<NumN32> &B) {
        NumN32 *output_data;
        MATH21_ASSERT(m.dims() == 3)
        NumN nr, nc;
        nr = m.dim(2);
        nc = m.dim(3);

        if (B.isSameSize(nr, nc) == 0) {
            B.setSize(nr, nc);
        }
        output_data = math21_memory_tensor_data_address(B);
        math21_image_tensor_to_argb8888(output_data, m);
    }

    void math21_image_size_to_Interval2D(NumN nr, NumN nc, Interval2D &I) {
        I(1).set(0, nr, 0, 1);
        I(2).set(0, nc, 0, 1);
    }

    void math21_image_size_to_Interval2D(const TenR &A, Interval2D &I) {
        math21_image_size_to_Interval2D(math21_image_get_nr(A), math21_image_get_nc(A), I);
    }

    void math21_image_rotate_90(const TenR &x, TenR &y, NumB isClockwise) {
        MATH21_ASSERT(x.dims() == 3)
        y.setSize(x.dim(1), x.dim(3), x.dim(2));
        Seqce<TenR> xs, ys;
        math21_operator_share_remove_dim_1(x, xs);
        math21_operator_share_remove_dim_1(y, ys);
        NumB isXAxis = isClockwise ? 0 : 1;
        for (NumN i = 1; i <= x.dim(1); ++i) {
            math21_op_matrix_trans_reverse_axis(xs(i), ys(i), isXAxis);
        }
    }
}

using namespace math21;

m21image math21_image_create_empty(NumN nr, NumN nc, NumN nch) {
    m21image image;
    image.nr = nr;
    image.nc = nc;
    image.nch = nch;
    image.data = 0;
    return image;
}

m21image math21_image_create_image(NumN nr, NumN nc, NumN nch) {
    m21image image = math21_image_create_empty(nr, nc, nch);
    image.data = math21_vector_create_with_default_value_cpu(nr * nc * nch, 0);
    return image;
}

m21image math21_image_create_image_int_input(int nr, int nc, int nch) {
    return math21_image_create_image(nr, nc, nch);
}

m21image math21_image_clone_image(m21image image0) {
    m21image image = math21_image_create_empty(image0.nr, image0.nc, image0.nch);
    image.data = math21_vector_create_from_cpuvector_cpu(image.nr * image.nc * image.nch, image0.data, 1);
    return image;
}

void math21_image_destroy_image(m21image *image) {
    if (image->data) {
        math21_vector_free_cpu(image->data);
        image->data = 0;
    }
}

void math21_image_destroy_image_no_pointer_pass(m21image image) {
    math21_image_destroy_image(&image);
}

void math21_image_set_image(m21image m, float s) {
    math21_vector_set_cpu(m.nr * m.nc * m.nch, s, m.data, 1);
}

void math21_image_pixel_value_set_ignore_error_32(m21image *m, NumN ir, NumN ic, NumN ich, NumR32 val) {
    if (ir >= m->nr || ic >= m->nc || ich >= m->nch) {
        return;
    }
    m->data[ich * m->nr * m->nc + ir * m->nc + ic] = val;
}

// see math21_device_image_pixel_value_set
void math21_image_pixel_value_set_32(m21image *m, NumN ir, NumN ic, NumN ich, NumR32 val) {
    math21_tool_assert(ir < m->nr && ic < m->nc && ich < m->nch);
    m->data[ich * m->nr * m->nc + ir * m->nc + ic] = val;
}

NumR32 math21_image_pixel_value_get_32(const m21image *m, NumN ir, NumN ic, NumN ich) {
    math21_tool_assert(ir < m->nr && ic < m->nc && ich < m->nch);
    return m->data[ich * m->nr * m->nc + ir * m->nc + ic];
}

void math21_image_pixel_add_to(m21image *m, NumN ir, NumN ic, NumN ich, NumR32 val) {
    math21_tool_assert(ir < m->nr && ic < m->nc && ich < m->nch);
    m->data[ich * m->nr * m->nc + ir * m->nc + ic] += val;
}

void math21_image_embed_image(m21image src, m21image dst, NumN dst_offset_r, NumN dst_offset_c) {
    NumN ir, ic, ich;
    for (ich = 0; ich < src.nch; ++ich) {
        for (ir = 0; ir < src.nr; ++ir) {
            for (ic = 0; ic < src.nc; ++ic) {
                NumR32 val = math21_image_pixel_value_get_32(&src, ir, ic, ich);
                math21_image_pixel_value_set_32(&dst, dst_offset_r + ir, dst_offset_c + ic, ich, val);
            }
        }
    }
}

// todo: use math21_img_resize instead
// todo: check
m21image math21_image_resize_image(m21image m, NumN nr_dst, NumN nc_dst) {
    m21image resized = math21_image_create_image(nr_dst, nc_dst, m.nch);
    m21image part = math21_image_create_image(m.nr, nc_dst, m.nch);
    NumN ir, ic, ich;
    NumR32 w_scale = (NumR32) (m.nc - 1) / (nc_dst - 1);
    NumR32 h_scale = (NumR32) (m.nr - 1) / (nr_dst - 1);
    for (ich = 0; ich < m.nch; ++ich) {
        for (ir = 0; ir < m.nr; ++ir) {
            for (ic = 0; ic < nc_dst; ++ic) {
                NumR32 val = 0;
                if (ic == nc_dst - 1 || m.nc == 1) {
                    val = math21_image_pixel_value_get_32(&m, ir, m.nc - 1, ich);
                } else {
                    NumR32 sx = ic * w_scale;
                    NumN ix = (NumN) sx;
                    NumR32 dx = sx - ix;
                    val = (1 - dx) * math21_image_pixel_value_get_32(&m, ir, ix, ich)
                          + dx * math21_image_pixel_value_get_32(&m, ir, ix + 1, ich);
                }
                math21_image_pixel_value_set_32(&part, ir, ic, ich, val);
            }
        }
    }
    for (ich = 0; ich < m.nch; ++ich) {
        for (ir = 0; ir < nr_dst; ++ir) {
            NumR32 sy = ir * h_scale;
            NumN iy = (NumN) sy;
            NumR32 dy = sy - iy;
            for (ic = 0; ic < nc_dst; ++ic) {
                NumR32 val = (1 - dy) * math21_image_pixel_value_get_32(&part, iy, ic, ich);
                math21_image_pixel_value_set_32(&resized, ir, ic, ich, val);
            }
            if (ir == nr_dst - 1 || m.nr == 1) continue;
            for (ic = 0; ic < nc_dst; ++ic) {
                NumR32 val = dy * math21_image_pixel_value_get_32(&part, iy + 1, ic, ich);
                math21_image_pixel_add_to(&resized, ir, ic, ich, val);
            }
        }
    }

    math21_image_destroy_image(&part);
    return resized;
}

// resize image while keeping aspect ratio, and set background to 0
m21image math21_image_resize_with_padding(m21image m, NumN nr_dst, NumN nc_dst) {
    NumN nc_new;
    NumN nr_new;
    NumR nr_new0, nc_new0;;
    math21_number_rectangle_resize_just_put_into_box(m.nr, m.nc, nr_dst, nc_dst, &nr_new0, &nc_new0);
    nr_new = (NumN) nr_new0;
    nc_new = (NumN) nc_new0;

    m21image resized = math21_image_resize_image(m, nr_new, nc_new);
    m21image dst = math21_image_create_image(nr_dst, nc_dst, m.nch);
    math21_image_set_image(dst, 0);
    math21_image_embed_image(resized, dst, (nr_dst - nr_new) / 2, (nc_dst - nc_new) / 2);
    math21_image_destroy_image(&resized);
    return dst;
}

void math21_image_resize_and_embed(m21image src, int nr_resized, int nc_resized, int offset_r, int offset_c,
                                   m21image canvas) {
    NumN ir, ic, ich;
    for (ich = 0; ich < src.nch; ++ich) {
        for (ir = 0; ir < nr_resized; ++ir) {
            for (ic = 0; ic < nc_resized; ++ic) {
                NumR32 ir_o = ((NumR32) ir / nr_resized) * src.nr;
                NumR32 ic_o = ((NumR32) ic / nc_resized) * src.nc;
                NumR val;
                NumB flag = math21_device_image_get_pixel_bilinear_interpolate_32(
                        src.data - 1, &val, ich + 1, ir_o + 1, ic_o + 1,
                        src.nch, src.nr, src.nc, 0);
                if (!flag) {
                    continue;
//                    math21_image_pixel_value_set_32(&canvas, ir + offset_r, ic + offset_c, ich, 0);
                }
                math21_image_pixel_value_set_32(&canvas, ir + offset_r, ic + offset_c, ich, (NumR32) val);
            }
        }
    }
}

void math21_image_flip_horizontally(m21image src) {
    math21_generic_tensor_reverse_axis_3_in_d3_cpu(src.data, src.nch, src.nr, src.nc, m21_type_NumR32);
}

// [start, end]
void math21_image_draw_line_horizontal(m21image image, int ir, int start, int end,
                                       float r, float g, float b) {
    int i;
    for (i = start; i <= end; ++i) {
        image.data[0 * image.nr * image.nc + ir * image.nc + i] = r;
        image.data[1 * image.nr * image.nc + ir * image.nc + i] = g;
        image.data[2 * image.nr * image.nc + ir * image.nc + i] = b;
    }
}

// [start, end]
void math21_image_draw_line_vertical(m21image image, int ic, int start, int end,
                                     float r, float g, float b) {
    int i;
    for (i = start; i <= end; ++i) {
        image.data[0 * image.nr * image.nc + i * image.nc + ic] = r;
        image.data[1 * image.nr * image.nc + i * image.nc + ic] = g;
        image.data[2 * image.nr * image.nc + i * image.nc + ic] = b;
    }
}

void
math21_image_draw_box_one(m21image image, int r_start, int r_end, int c_start, int c_end, float r, float g, float b) {
    math21_number_interval_intersect_to_int(0, image.nr - 1, &r_start, &r_end);
    math21_number_interval_intersect_to_int(0, image.nc - 1, &c_start, &c_end);

    math21_image_draw_line_horizontal(image, r_start, c_start, c_end, r, g, b);
    math21_image_draw_line_horizontal(image, r_end, c_start, c_end, r, g, b);
    math21_image_draw_line_vertical(image, c_start, r_start, r_end, r, g, b);
    math21_image_draw_line_vertical(image, c_end, r_start, r_end, r, g, b);
}

void math21_image_draw_box_with_width(m21image image, int r_start, int r_end, int c_start, int c_end,
                                      int line_width, float r, float g, float b) {
    int i;
    for (i = 0; i < line_width; ++i) {
        math21_image_draw_box_one(image, r_start + i, r_end - i, c_start + i, c_end - i, r, g, b);
    }
}

void math21_image_convert_ch_r_c_to_r_c_ch_NumR_2_NumN8(const NumR *src, NumN8 *dst, NumN nr, NumN nc, NumN nch) {
    math21_image_convert_ch_r_c_to_r_c_ch(src, dst, nr, nc, nch);
}

void math21_image_convert_r_c_ch_to_ch_r_c_NumN8_2_NumR(const NumN8 *src, NumR *dst, NumN nr, NumN nc, NumN nch) {
    math21_image_convert_r_c_ch_to_ch_r_c(src, dst, nr, nc, nch);
}

static const NumN m21_colors_basic[6][3] =
        {{1, 0, 1},
         {1, 0, 0},
         {1, 1, 0},
         {0, 1, 0},
         {0, 1, 1},
         {0, 0, 1}};

// i: {1, ..., n}
NumR math21_image_get_ith_color_normalized(NumN ich, NumN i, NumN n) {
    i = i * 123157 % n;
    NumR ratio = ((NumR) i / n) * 5;
    NumN a = (NumN) xjfloor(ratio);
    NumN b = (NumN) xjceil(ratio);
    ratio -= a;
    NumR v = (1 - ratio) * m21_colors_basic[a][ich - 1] + ratio * m21_colors_basic[b][ich - 1];
    return v;
}

NumN math21_image_color_get_red(NumN i, NumN n) {
    return (NumN) (math21_image_get_ith_color_normalized(1, i, n) * 255);
}

NumN math21_image_color_get_green(NumN i, NumN n) {
    return (NumN) (math21_image_get_ith_color_normalized(2, i, n) * 255);
}

NumN math21_image_color_get_blue(NumN i, NumN n) {
    return (NumN) (math21_image_get_ith_color_normalized(3, i, n) * 255);
}

namespace math21 {
    template<typename T>
    void _math21_image_color_generate(Tensor<T> &color, NumN i, NumN n) {
        NumN red = math21_image_color_get_red(i, n);
        NumN green = math21_image_color_get_green(i, n);
        NumN blue = math21_image_color_get_blue(i, n);
        color.setSize(3);
        color = red, green, blue;
    }

    void math21_image_color_generate(VecN &color, NumN i, NumN n) {
        _math21_image_color_generate(color, i, n);
    }

    void math21_image_color_generate(VecR &color, NumN i, NumN n) {
        _math21_image_color_generate(color, i, n);
    }

    NumR math21_image_measure_similarity(const TenR &x0, const TenR &y0, NumN scale, NumR pixelEpsilon) {
        TenR x, y, z;
        VecN d(3);
        x0.shape(d);
        if (scale == 0) {
            d = d(1), 100, 100;
        } else {
            d = d(1), d(2) / scale, d(3) / scale;
        }
        x.setSize(d);
        y.setSize(d);
        math21_img_resize(x0, x);
        math21_img_resize(y0, y);
        math21_op_subtract(x, y, z);
        math21_op_abs_onto(z);
        math21_op_ele_is_less_than_onto(pixelEpsilon, z);
        NumN count = (NumN) math21_op_vector_sum(z);
        return 1 - count / (NumR) z.size();
    }
}

