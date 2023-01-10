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

#include <assert.h>
#include "config/config.h"
#include "export.h"

#ifdef __cplusplus
extern "C" {
#endif

// breakpoints
M21_EXPORT static inline void math21_assert_breakpoint(
) {}
/*!
    ensures
        - this function does nothing
          It exists just so you can put breakpoints on it in a debugging tool.
          It is called only when an MATH21_ASSERT or MATH21_CASSERT fails and is about to
          throw an exception.
!*/

void math21_global_error_set();

#ifdef MATH21_FLAG_DISABLE_EXIT
#define xjassert() math21_global_error_set()
#else
#define xjassert() assert(0)
#endif

#ifdef __cplusplus
}
#endif
