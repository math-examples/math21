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

#include "inner.h"

namespace math21 {
/*
Properties about A'A:
(1) A'A in M(F, n, n) is self-adjoint and positive semi-definite, where F = R or C.
(2) Exist W unitary, D diagonal, s.t., A'A = WD*DW', here D := diag(lambda1, ..., lambdan)
(3) P:=sqrt(A'A) := W*D*W'
(4) P is self-adjoint and psd
(5) ||Av|| = ||Pv||
(6) kernel(A) = kernel(P) (space)
    rank(A) = rank(P)
(7) im(A).perp < kernel(A*A'), where S.perp means orthogonal complement of subspace S
 * */

/*
(right) polar decomposition
A in M(F, m, n), m>=n =>
exist U, P, with U in M(F, m, n), P in M(F, n, n), U having orthonormal columns
and P positive semi-definite, s.t., A = UP, where F = R or C.
proof:
(1) P:=sqrt(A'A) := W*D*W'
(2) let r = rank(P) => r <= n <= m
    let lam1, ..., lamr >0, W = (Wr, Wc), where Wr is m*r, Wc m*(n-r)
    => <Wc> = ker(P) = ker(A)
(3) <A*Wr*diag(1/lami)> = <A*Wr> = Im(A), where diag(1/lami) = diag(1/lam1, ..., 1/lamr)
(4) if r<n => exists Vc, s.t. V = (Vr, Vc) has orthonormal columns, where Vr := A*Wr*diag(1/lami). (Note m>=n)
(5) U:= VW' has orthonormal columns. (Note if C= A*B, A, B have orthonormal columns => C has orthonormal columns)
(6) Because UPW = VW'PW = V*D = (A*Wr, 0) = AW => UP = A
 * */

/*
polar decomposition => svd
proof:
suppose A in M(F, m, n), m>=n
A = UP = U*W*D*W' = (VW'W)DW' = VDW'
 * */

/*
svd => U in polar decomposition
proof:
suppose A in M(F, m, n), m>=n
A = VDW' (svd) and V=UW in polar decomposition
So U=VW'
 * */

/*
Uniqueness of polar decomposition ?
Uniqueness of svd ?
 * */
}
