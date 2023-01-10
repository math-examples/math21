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

#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <ctime>
//#include "../ad/files.h" // for math21_global_ad_log_data
#include "inner.h"
#include "util_c.h"
#include "../print/files.h"

using namespace math21;

namespace math21 {
    static int m21timeIndex = 0;
}

void math21_time_index_add() {
    ++m21timeIndex;
    int curTime = m21timeIndex;
    if (math21_time_index_get() == math21_time_index_get_debug_time()) {
//        math21_tool_assert(0);
    }
}

int math21_time_index_get() {
    return m21timeIndex;
}

int math21_time_index_get_start_time() {
//    return 261000;
//    return 10272;
//    return 1000;
    return 0;
}

int math21_time_index_get_end_time() {
    return 980;
//    return 1;
//    return 10390;

//    return 1100;
//    return 1500;
}

int math21_time_index_get_debug_time() {
//    return 1820;
//    return 1;
//    return 964;
//    return 10379;
    return 28;
}

namespace math21 {
    int m21debugLevel = 0;
}

NumB math21_time_is_debug() {
    return m21debugLevel ? 1 : 0;
}

void math21_time_set_debug(NumN debugLevel) {
    m21debugLevel = debugLevel;
}

void math21_error(const char *s) {
//    perror(s);
    MATH21_ASSERT(0, s);
    exit(-1);
}

// todo: use m21log when android
void math21_file_error(const char *s) {
    fprintf(stderr, "Couldn't open file: %s\n", s);
//    m21log(math21_string_to_string("Couldn't open file: ", s, "\n"));
    MATH21_ASSERT(0)
    exit(-1);
}

void math21_file_warn(const char *s) {
    fprintf(stderr, "Couldn't open file: %s\n", s);
}

void math21_file_property_warn(const char *s, const char *content) {
    fprintf(stderr, "%s: %s\n", content, s);
}

// replace time(), use less
time_t math21_c_tim(time_t *timer) {
    return time(timer);
}

// replace szzrand()
void math21_c_seed_random_generator(unsigned int seed) {
    srand(seed);
//    math21_c_seed_random_generator_by_current_time();
}

namespace math21 {
    static unsigned int myTime = 0;
}

// replace srand(time(0));
void math21_c_seed_random_generator_by_current_time() {
//    srazznd(time(0));
    srand(myTime);
    ++myTime;
//    printf("\nmyTime: %d\n", myTime);
}

unsigned int math21_c_seed_get() {
    ++myTime;
    return myTime;
//    printf("\nmyTime: %d\n", myTime);
}

namespace math21 {
    NumB is_log_auto_buffer = 0;
}

NumB math21_global_tensor_is_log_auto_buffer() {
    return is_log_auto_buffer;
}

void math21_global_tensor_enable_log_auto_buffer() {
    is_log_auto_buffer = 1;
}

void math21_global_tensor_disable_log_auto_buffer() {
    is_log_auto_buffer = 0;
}

namespace math21 {
    NumB is_log_all_elements = 0;
}

NumB math21_global_tensor_is_log_all_elements() {
    return is_log_all_elements;
}

void math21_global_tensor_enable_log_all_elements() {
    is_log_all_elements = 1;
}

void math21_global_tensor_disable_log_all_elements() {
    is_log_all_elements = 0;
}

namespace math21 {
    NumB is_log_matlab_style = 0;
}

NumB math21_global_tensor_is_log_matlab_style() {
    return is_log_matlab_style;
}

void math21_global_tensor_enable_log_matlab_style() {
    is_log_matlab_style = 1;
}

namespace math21 {
    NumB is_log_no_last_new_line = 0;
}

NumB math21_global_tensor_is_log_no_last_new_line() {
    return is_log_no_last_new_line;
}

// matlab style
void math21_global_tensor_enable_log_no_last_new_line() {
    is_log_no_last_new_line = 1;
}

namespace math21 {
//NumN ad_debug_var_id = 16;
    NumN ad_debug_var_id = 0;
}

// todo: read from file
NumN math21_global_ad_debug_var_id() {
    return ad_debug_var_id;
}

void math21_global_set_ad_debug_var_id(NumN id) {
    ad_debug_var_id = id;
}

NumB math21_global_ad_log_data() {
//    ad::ad_get_data().log();
    return 1;
}

NumB math21_global_ad_log_time() {
    return 0;
}

namespace math21 {
    NumB haveError = 0;
}

NumB math21_global_have_error() {
    return haveError;
}

void math21_global_error_set() {
    haveError = 1;
}

void math21_global_error_clear() {
    haveError = 0;
}

namespace math21 {
    NumB isDebug = 0;
}

NumB math21_global_is_debug() {
    return isDebug;
}

void math21_global_enable_debug() {
    isDebug = 1;
}

void math21_global_disable_debug() {
    isDebug = 0;
}

namespace math21 {
//    NumB isAligned = 1;
    NumB isAligned = 0;
}

NumB math21_global_is_data_aligned() {
    return isAligned;
}

void math21_global_enable_alignment() {
    isAligned = 1;
}

void math21_global_disable_alignment() {
    isAligned = 0;
}

namespace math21 {
    NumB ad_is_check_nan = 0;
}

NumB math21_global_ad_is_check_nan() {
    return ad_is_check_nan;
}

void math21_global_ad_enable_check_nan() {
    ad_is_check_nan = 1;
}

NumB math21_global_is_log_todo() {
    return 0;
}

std::string math21_global_version() {
    std::string version = MATH21_LIB_VERSION;
    return version;
}

void math21_global_version_log() {
    std::string version = MATH21_LIB_VERSION;
    m21log("math21 version", version);
    NumB flag = math21_io_generic_type_write_to_file(version, "./math21_version", 0, 0);
    MATH21_ASSERT(flag);
}
