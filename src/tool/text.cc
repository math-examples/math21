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

#include "string.h"
#include "text.h"
#include "file_c.h"
#include "file_c.h"

namespace math21 {

    NumB math21_file_text_string_save(const char *filename, const std::string &s, NumB append) {
        std::ofstream io;
        if(!append){
            io.open(filename, std::ofstream::out);
        }else{
            io.open(filename, std::ofstream::app);
        }
        if (!io.is_open()) {
            math21_file_warn(filename);
            return 0;
        }
        io << s;
        io.close();
        return 1;
    }

    std::string math21_file_text_string_load(const char *filename) {
        std::string s;
        std::ifstream io;
        io.open(filename, std::ofstream::in);
//        if (!io.is_open()) {
        if (!io) {
            math21_file_warn(filename);
            return s;
        }
        std::ostringstream ss;
        ss << io.rdbuf(); // reading data
        s = ss.str();
        io.close();
        return s;
    }

    // no '\0'
    NumB math21_file_text_read_lines(const char *filename, Seqce<TenN8> &lines) {
        lines.clear();
        FILE *fp = fopen(filename, "r");
        if (!fp) {
            math21_file_warn(filename);
            return 0;
        }
        char *line;
        while ((line = math21_file_get_line_c(fp)) != 0) {
            NumN n = (NumN) strlen(line);
            TenN8 A;
            math21_operator_tensor_set_string(A, n, (const NumN8 *) line);
            lines.push(A);
        }
        return 1;
    }
}
