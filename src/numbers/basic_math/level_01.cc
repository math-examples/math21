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
#include "level_01.h"
#include "level_01_c.h"
#include "../../gpu/files.h" // for opencl x+n, see math21_number_pointer_input_increase

#if !defined(MATH21_FLAG_USE_CUDA)

#include "level_01"

#endif

namespace math21 {
    void math21_number_get_from_and_to(NumN n0, NumZ &from, NumZ &to) {
        NumZ n = n0;
        if (from == 0) {
            from = 1;
        } else if (from < 0) {
            from = n + from + 1;
        }
        if (to == 0) {
            to = n;
        } else if (to < 0) {
            to = n + to + 1;
        }
    }

    // 1 <= from <= to <= n
    NumB math21_number_check_from_and_to(NumN n, NumZ from, NumZ to) {
        if (from < 1 || to > n || from > to) {
            return 0;
        } else {
            return 1;
        }
    }

    void math21_number_get_from_only_with_check(NumN n0, NumZ &from) {
        NumZ n = n0;
        if (from == 0) {
            from = 1;
        } else if (from < 0) {
            from = n + from + 1;
        } else {
            MATH21_ASSERT(from <= n)
        }
    }

    void math21_number_get_from_and_num_with_check(NumN n0, NumZ &from, NumN &num) {
        NumZ n = n0;
        if (from == 0) {
            from = 1;
        } else if (from < 0) {
            from = n + from + 1;
        } else {
            MATH21_ASSERT(from <= n)
        }
        if (num == 0) {
            num = (NumN) n + 1 - from;
        } else {
            MATH21_ASSERT(from + num <= n + 1)
        }
    }

    /** Returns true if `ch` is in `[ \f\n\r\t\v]`. */
    NumB math21_number_is_white_space(NumN8 x) {
        return isspace(x);
    }

    NumB math21_number_is_comma(NumN8 x) {
        if (x == ',')return 1;
        return 0;
    }

    NumB math21_number_is_letter(NumN8 x) {
        if ((x >= 'A' && x <= 'Z') || (x >= 'a' && x <= 'z')) {
            return 1;
        }
        return 0;
    }

    NumB math21_number_is_digit(NumN8 x) {
        if (x >= '0' && x <= '9') {
            return 1;
        }
        return 0;
    }

    NumN8 math21_number_hex_to_letter(NumN8 y1) {
        return y1 < 10 ? '0' + y1 : 'A' + y1 - 10;
    }

    NumN8 math21_number_hex_to_num(NumN8 x1) {
        return (x1 >= '0' && x1 <= '9') ? x1 - '0' :
               (x1 >= 'A' && x1 <= 'F') ? x1 - 'A' + 10 : x1 - 'a' + 10;
    }

    void math21_number_char2hex(NumN8 x, NumN8 &y1, NumN8 &y2) {
        y1 = (x >> 4) & 0x0F;
        y2 = x & 0x0f;
        y1 = math21_number_hex_to_letter(y1);
        y2 = math21_number_hex_to_letter(y2);
    }

    void math21_number_hex2char(NumN8 x1, NumN8 x2, NumN8 &y) {
        x1 = math21_number_hex_to_num(x1);
        x2 = math21_number_hex_to_num(x2);
        y = x1 << 4 | x2;
    }

    // calculate minimum offset of x, y respectively such that nx - offset_x = ny - offset_y
    void math21_number_get_offset_from_right(NumN nx, NumN ny, NumN &offset_x, NumN &offset_y) {
        NumN offset = xjmax(nx, ny) - xjmin(nx, ny);
        offset_x = nx >= ny ? offset : 0;
        offset_y = ny >= nx ? offset : 0;
    }

    NumN math21_number_get_n_and_check_offset(NumN size, NumN n, NumN offset) {
        if (size == 0)return 0;
        MATH21_ASSERT(offset < size, "offset not in [0, size)");
        NumN n_max = size - offset;
        if (n == 0 || n > n_max) {
            n = n_max;
        }
        return n;
    }
}

using namespace math21;

int math21_number_min_2_int(int x1, int x2) {
    return xjmin(x1, x2);
}

int math21_number_min_3_int(int x1, int x2, int x3) {
    return xjmin(x1, x2, x3);
}

// keep aspect ratio
void math21_number_rectangle_resize_just_put_into_box(NumR src_nr, NumR src_nc,
                                                      NumR box_nr, NumR box_nc,
                                                      NumR *dst_nr, NumR *dst_nc) {
    math21_tool_assert(dst_nr && dst_nc);
    NumR ratio = xjmin(box_nc / src_nc, box_nr / src_nr);
    *dst_nr = src_nr * ratio;
    *dst_nc = src_nc * ratio;
}

// [c, d] <- [a, b] intersect [c, d]
template<typename T>
void _math21_number_interval_intersect_to(T a, T b, T *c0, T *d0) {
    int c = *c0;
    int d = *d0;
    if (c < a) c = a;
    if (c > b) c = b;
    if (d < a) d = a;
    if (d > b) d = b;
    *c0 = c;
    *d0 = d;
}

// [c, d] <- [a, b] intersect [c, d]
void math21_number_interval_intersect_to_int(int a, int b, int *c0, int *d0) {
    _math21_number_interval_intersect_to(a, b, c0, d0);
}

NumN math21_number_axis_insert_pos_check(NumN dims, NumZ pos) {
    MATH21_ASSERT(xjIsIn(xjabs(pos), 1, dims + 1),
                  "abs|pos| not in [1, n+1], n = " << dims << ", pos = " << pos);
    return pos < 0 ? dims + 2 + pos : pos;
}

NumN math21_number_container_pos_check(NumN size, NumZ pos) {
    MATH21_ASSERT(xjIsIn(xjabs(pos), 1, size),
                  "|pos| not in [1, n], n = " << size << ", pos = " << pos);
    return pos < 0 ? size + 1 + pos : (NumN) pos;
}

NumN math21_number_container_pos_correct(NumN size, NumZ pos) {
    if (xjIsIn(xjabs(pos), 1, size)) return pos < 0 ? size + 1 + pos : (NumN) pos;
    else return 0;
}

NumN math21_number_container_stride_get_n(NumN n, NumN stride, NumN offset) {
    return (NumN) xjceil((n - offset) / (NumR) stride);
}

NumN math21_number_container_get_n(NumN n, NumN n_x, NumN stride_x, NumN offset_x) {
    NumN n_max_x;
    n_max_x = math21_number_container_stride_get_n(n_x, stride_x, offset_x);
    if (n == 0)n = n_max_x;
    return xjmin(n_max_x, n);
}

NumN math21_number_container_assign_get_n(NumN n,
                                          NumN n_x, NumN stride1_x, NumN offset1_x,
                                          NumN n_y, NumN stride1_y, NumN offset1_y) {
    NumN n_max_x;
    n_max_x = math21_number_container_stride_get_n(n_x, stride1_x, offset1_x);
    NumN n_max_y;
    n_max_y = math21_number_container_stride_get_n(n_y, stride1_y, offset1_y);
    NumN n_max = xjmin(n_max_x, n_max_y);
    if (n == 0)n = n_max;
    return xjmin(n_max, n);
}

PtrVoidWrapper math21_number_pointer_increase(PtrVoidWrapper x, int n, NumN type) {
#if !defined(MATH21_FLAG_USE_OPENCL)
    if (type == m21_type_NumN8) {
        n *= 1;
    } else if (type == m21_type_NumN) {
        n *= sizeof(NumN);
    } else if (type == m21_type_NumR) {
        n *= sizeof(NumR);
    } else if (type == m21_type_NumR32) {
        n *= 4;
    } else if (type == m21_type_NumR64) {
        n *= 8;
    } else {
        math21_tool_assert(0);
    }
#endif
    return (PtrN8Wrapper) x + n;
}

PtrVoidInWrapper
math21_number_pointer_input_increase(PtrVoidInWrapper x, int n, NumN type) {
#if !defined(MATH21_FLAG_USE_OPENCL)
    if (type == m21_type_NumN8) {
        n *= 1;
    } else if (type == m21_type_NumN) {
        n *= sizeof(NumN);
    } else if (type == m21_type_NumR) {
        n *= sizeof(NumR);
    } else if (type == m21_type_NumR32) {
        n *= 4;
    } else if (type == m21_type_NumR64) {
        n *= 8;
    } else {
        math21_tool_assert(0);
    }
#endif
    return (PtrN8InWrapper) x + n;
}
