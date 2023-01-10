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
#include "inner_cc.h"
#include "../matrix_analysis/files.h"
#include "vspace.h"

namespace math21 {

    // vector space V = <v1, ..., vn> = <u1, ..., un>, where dim(V) = n
    // {vi} is a set of basis vectors, {ui} is an orthonormal basis.
    // A = (v1, ..., vn), A: m*n, m>=n,  B = (u1, ..., un)
    void math21_la_vspace_get_orthonormal_basis(const MatR &A, MatR &B){
        MatR C, D;
        math21_operator_svd_real(A, B, D, C);
    }

    // Let V be an n-dim vector space, I, J two bases and E standard basis,
    // then the transition matrix P(J<-I) from I to J is
    // P(J<-I) = P(J<-E)P(E<-I) = P(E<-J).inv * P(E<-I)
    // v(J) = P(J<-I) * v(I) for any vector v in V, here v(J) is coordinates of v in basis J.
    // The columns of P(J<-I) are coordinates of I in basis J
    // see  https://pi.math.cornell.edu/~andreim/Lec26.pdf
    //      https://mathworld.wolfram.com/ChangeofCoordinatesMatrix.html
    void math21_la_vspace_get_transition_matrix(const MatR &I, const MatR &J, MatR &P){
        MatR J_inv;
        math21_operator_inverse(J, J_inv);
        math21_op_mat_mul(J_inv, I, P);
    }
}