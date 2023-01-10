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

#include "../matlab/files.h"
#include "OptAlg.h"
#include "LevMar.h"

namespace math21 {
    using namespace matlab;

// Deprecated.
// https://engineering.purdue.edu/kak/courses-i-teach/ECE661.08/homework/HW5_LM_handout.pdf
// the Levenberg Marquardt algorithm with finite difference methods
// fit f(x; theta) using data x, y.
    void math21_opt_levmar_fdm(void *data, const MatR &y, const VecR &theta_0,
                               void (*f)(const VecR &paras, MatR &y, const void *data), VecR &theta_est,
                               NumN max_iters, NumN logLevel) {
//        NumB useMarquardtUpdate = 1;
        NumB useMarquardtUpdate = 0;
//    NumN n_data = y.size();
//    NumN n_paras = theta_0.size();
//        NumR lambda = 1;
        NumR lambda = 0.01;
        const NumR lambda_max = 1e7;
        const NumR lambda_min = 1e-7;
        NumN updateJ = 1;
        theta_est = theta_0;
        NumN it;

        // J(n_data, n_paras);
        MatR J;
        MatR H;
        VecR d;
        NumR e;
        timer time;
        time.start();
        TenR H_lm_inv;
        for (it = 1; it <= max_iters; ++it) {
            if (updateJ == 1) {
                // evaluate the Jacobian matrix at current paras theta_est.
                math21_fdm_derivative_1_order_2_error_central_diff_Jacobian(f, theta_est, J, data);

                // evaluate the distance error at the current parameters
                VecR y_est;
                f(theta_est, y_est, data);
                d = (y - y_est).toVector();
                // compute the approximated Hessian matrix
                // H shape: n_paras * n_paras
                H = transpose(J) * J;
                if (it == 1) {
                    e = dot(d, d);
                }
            }

            // apply the damping factor to the Hessian matrix
            MatR H_lm;
            if (!useMarquardtUpdate) { // Levenberg's update
                H_lm = H + (lambda * eye(H.nrows(), H.ncols()));
            } else { // Marquardt's update
                H_lm = H + (lambda * diag(diag(H))); // did'n succeed.
            }

            // compute the updated paras, (J'WJ + lambda*I) * dp = J'W(y-y_est)
            if (!math21_operator_inverse(H_lm, H_lm_inv)) { // todo: may use svd
                if (logLevel > 10) {
                    H_lm.log("H_lm");
                    m21log("lambda", lambda);
                }
                break;
            }
            if (logLevel > 200) J.log("J");
            MatR dp = H_lm_inv * (transpose(J) * d);
            VecR theta_lm = theta_est + dp;

            // evaluate the total distance error at the updated paras.
            VecR y_est_lm;
            f(theta_lm, y_est_lm, data);
            VecR d_lm = y - y_est_lm;
            NumR e_lm = dot(d_lm, d_lm);

            // use the updated paras or discard
            if (e_lm < e) {
                lambda = xjmax(lambda / 10, lambda_min);
                theta_est = theta_lm;
                e = e_lm;
                if (logLevel > 10) {
                    printf("iteration %d, error %lf\n", it, e);
                }
                updateJ = 1;
            } else {
                updateJ = 0;
                lambda = xjmin(lambda * 10, lambda_max);
            }
        }
        time.end();
        if (logLevel > 1) {
            // evaluate the distance error at the current parameters
            VecR y_est;
            f(theta_est, y_est, data);
            if (logLevel > 3)y_est.log("y_est");
            printf("last iteration %d, error %lf\n", it > max_iters ? max_iters : it, e);
            printf("time used %lf ms\n", time.time());
        }
    }

    class SimpleSparseLevMarqCallback : public OptCallbackSparseLevMarq {
    private:
        FunctionNd *cbNoJ;
        OptCallbackLevMarq *cb;
    public:
        SimpleSparseLevMarqCallback() : cbNoJ(0), cb(0) {
        }

        explicit SimpleSparseLevMarqCallback(FunctionNd *_cb) : cbNoJ(_cb), cb(0) {
        }

        explicit SimpleSparseLevMarqCallback(OptCallbackLevMarq *_cb) : cbNoJ(0), cb(_cb) {
        }

        NumB compute(const VecR &x, VecR *Jtf, MatR *JtJ, NumR *loss2) override {
            MATH21_ASSERT(cbNoJ || cb);
            NumB flag = 1;

            VecR value;
            MatR J;

            if (cbNoJ) {
                if (!cbNoJ->valueAt(x, value))return 0;
                if (Jtf || JtJ) {
                    if (!math21_fdm_derivative_1_order_2_error_central_diff_Jacobian(*cbNoJ, x, J))return 0;
                }
            } else {
                if (!cb->compute(x, value, Jtf || JtJ ? &J : nullptr))return 0;
            }

            if (Jtf) {
                // dLt = L'.t = Jtf = J.t * f
                math21_op_mat_mul(J, value, *Jtf, 1);
            }
            if (JtJ) {
                // A = JtJ = J.t * J
                math21_op_mat_mul(J, J, *JtJ, 1);
            }
            if (loss2) {
                // loss2 = 2*L(x) = ||f(x)||^2, L(x) = 1/2 * ||f(x)||^2
                *loss2 = xjsquare(math21_op_vector_norm(value, 2));
            }
            return flag;
        }

        void setDebugValue(NumN value) override {
            if (cbNoJ)cbNoJ->setDebugValue(value);
            else cb->setDebugValue(value);
        }

        NumN getDebugValue() const override {
            if (cbNoJ) return cbNoJ->getDebugValue();
            else return cb->getDebugValue();
        }
    };

    class ILevMarqCallback {
    private:
        OptCallbackSparseLevMarq *cb;
        VecN mask;

        VecR m_x_full;
        VecR m_Jtf_full;
        MatR m_JtJ_full;

    public:
        ILevMarqCallback() : cb(0) {
        }

        void setMask(const VecN &mask_) {
            this->mask = mask_;
        }

        void setCb(OptCallbackSparseLevMarq *cb_) {
            this->cb = cb_;
        }

        OptCallbackSparseLevMarq *getCb() const {
            return cb;
        }

        void getx(VecR &x, const VecR &x_full) {
            if (mask.isEmpty()) {
                x = x_full;
                return;
            }
            MATH21_ASSERT(x_full.size() == mask.size());
            math21_op_subvector_get_by_mask(x, x_full, mask);
            m_x_full = x_full; // backup x full
        }

        void setx(const VecR &x, VecR &x_full) const {
            if (mask.isEmpty()) {
                math21_op_vector_set_by_vector(x, x_full);// x_full memory kept
                return;
            }
            MATH21_ASSERT(x_full.size() == mask.size());
            math21_op_subvector_set_by_mask(x, x_full, mask);
        }

        void setDebugValue(NumN value) {
            cb->setDebugValue(value);
        }

        NumN getDebugValue() const {
            return cb->getDebugValue();
        }

        NumB compute(const VecR &x, VecR *Jtf, MatR *JtJ, NumR *loss2) {
            MATH21_ASSERT(cb);
            NumB flag = 1;

            if (mask.isEmpty()) {
                if (!cb->compute(x, Jtf, JtJ, loss2))return 0;
                return 1;
            }

            math21_op_subvector_set_by_mask(x, m_x_full, mask);
            if (!cb->compute(m_x_full, Jtf ? &m_Jtf_full : 0, JtJ ? &m_JtJ_full : 0, loss2))return 0;
            if (Jtf) {
                math21_op_subvector_get_by_mask(*Jtf, m_Jtf_full, mask);
            }
            if (JtJ) {
                math21_op_submatrix_get_by_mask(*JtJ, m_JtJ_full, mask, mask);
            }
            return flag;
        }
    };


    /*
Let k in R, u(x), v(x) in R^m, x in R^n, A in R^(m*m),
v(x) = k(x)*u(x),  => dv/dx = u*k' + k*u'
k(x) = u(x).t * v(x),  => dk/dx = v.t * u' + u.t *v'
v(x) = A * u(x),  => dv/dx = A * u'

argmin L(x),
x := x + alpha * h, with alpha > 0

steepest descent method
h = -L'(x).t

Newton's method
L''(x) * h = -L'(x).t => h
if L(x) = 1/2 * ||f(x)||^2, f: R^n-> R^m => L' = f.t * J, L''(x) = J(x).t * J(x) + sum(i:1->m)(fi(x) * fi''(x))

Gauss-Newton method
Given the above conditions, and if f(x + h) = f_linear(h) = f(x) + J(x) * h
=> L(x+h) = 1/2 * ||f(x+h)||^2 = L_linear(h) = 1/2 * ||f_linear(h)||^2
here
     f(x) = f_linear(0), f(x+h) = f_linear(h), L(x) = L_linear(0), L(x+h) = L_linear(h)
     f'(x) = J(x) = f_linear'(h) = f_linear'(0), f''(x) !=0, f_linear''(h) = 0
     L'(x) = f(x).t * J(x) = f_linear(0).t * f_linear'(0) = L_linear'(0)
     L_linear'(h) = f_linear(h).t * f_linear'(h) = f_linear(h).t * J(x)
     L_linear''(h) = f_linear(h)'.t * J(x) = J(x).t * J(x)
Because L_linear''(0) * h = -L_linear'(0).t (Newton's method)
=> J(x).t * J(x) * h = -L'(x).t => h
2.
L_linear(0) - L_linear(h) = L(x) - L(x+h) = -L'(x) * h - 1/2 * h.t * J.t * J * h

Levenberg-Marquardt algorithm
(J(x).t * J(x) + mu*D) * h = -L'(x).t => h,
here the damping parameter mu > 0,
here if J.t*J not positive definite, D := I, else D := diag(J.t*J) or D := I.
So D must be positive definite.
From now on D := I is used if we use I instead of D.
1.
mu > 0 => J.t * J + mu * I positive definite => h is descent direction
if mu is small => h ~= h.GN
if mu is large => h ~= -(1/mu)*L', the short step in the steepest descent direction.
2. initial value mu0
Let A = J.t * J
=> h = -(A+mu*I).inv * L'.t = - sum(i:1->n){((L' * vi)/(lami + mu)) * vi},
here AV=V*Lam, V=(v1, vn), Lam = (lami), with V, Lam eigen-stuff respectively.
=> It is reasonable to relate the initial value mu0 to the size of Lam.
=> simple strategy for choosing mu0: mu0 = tao * max(diag(A)), tao in [1e-8, 1]
3. gain factor rou
rou = (L(x) - L(x+h))/(L_linear(0) - L_linear(h))
Because L_linear(0) - L_linear(h) = 1/2 * h.t (mu * D * h - L'.t) > 0
=> rou > 0 <=> L(x) > L(x+h)
3.2
-h.t * L'.t = h.t (A+mu*D) h > 0
3.3
(A + mu*D) * h = -L'.t => mu * D * h - L'.t = -Ah -2L'.t
=> So gain_linear = L_linear(0) - L_linear(h) = -1/2 * h.t * (Ah+2L'.t)
4. Marquardt's update strategy
if rou < rou1, then mu := beta * mu
if rou > rou2, then mu := mu / gamma
if rou > 0,    then x := x + h
where 0<rou1<rou2<1 and beta, gamma >1
Popular choice: beta = 2, gamma = 3, rou1 = 0.25, rou2 = 0.75
5 Nielsen's update strategy (avoiding the jumps in mu_new/mu)
if rou > 0 then mu := mu * max(1/gamma, 1-(beta-1)(2*rou-1)^p), nu := beta
else mu := mu * nu, nu := 2 * nu
if rou > 0,    then x := x + h
where p is an odd integer.
One choice: beta = 2, gamma = 3, nu = beta, p = 3
6 f: R^n-> R^m, why is m >= n required?
Reason one: m<n => J not full rank <=> A not full rank, so some theories may break.
# References
- [Damping parameter in Marquardt's method](https://www.imm.dtu.dk/documents/ftp/tr99/tr05_99.pdf)
*/
    class OptLevMarq : public OptDetail {
    private:
        ILevMarqCallback cb;
        NumB deleteCb;

        NumR epsh;
        NumR epsNormJtf;
        NumN maxIters;
        NumN logLevel;
        // 1: simple (https://engineering.purdue.edu/kak/courses-i-teach/ECE661.08/homework/HW5_LM_handout.pdf)
        // 2: Marquardt's update strategy, 3: Nielsen's update strategy
        // 4: matlab update strategy
        // 4: [Matlab's LMSolve package by Miroslav Balda](https://www.mathworks.com/matlabcentral/fileexchange/17534-lmfnlsq-solution-of-nonlinear-least-squares)
        NumN strategy; // see LevMarStrategySimple
    public:
        OptLevMarq() : maxIters(100), strategy(LevMarStrategyMatlab), logLevel(0) { init(); }

        OptLevMarq(const OptParasLevMarq &paras, FunctionNd *_cb) :
                maxIters(paras.maxIters), strategy(paras.strategy), logLevel(paras.logLevel) {
            init();
            auto scb = new SimpleSparseLevMarqCallback(_cb);
            cb.setCb(scb);
            cb.setMask(paras.mask);
            deleteCb = 1;
            epsh = paras.eps;
        }

        OptLevMarq(const OptParasLevMarq &paras, OptCallbackLevMarq *_cb) :
                maxIters(paras.maxIters), strategy(paras.strategy), logLevel(paras.logLevel) {
            init();
            auto scb = new SimpleSparseLevMarqCallback(_cb);
            cb.setCb(scb);
            cb.setMask(paras.mask);
            deleteCb = 1;
            epsh = paras.eps;
        }

        OptLevMarq(const OptParasLevMarq &paras, OptCallbackSparseLevMarq *_cb) :
                maxIters(paras.maxIters), strategy(paras.strategy), logLevel(paras.logLevel) {
            init();
            cb.setCb(_cb);
            cb.setMask(paras.mask);
            deleteCb = 0;
            epsh = paras.eps;
        }

        virtual ~OptLevMarq() {
            if (deleteCb)delete (SimpleSparseLevMarqCallback *) cb.getCb();
        }

        void init() {
            epsNormJtf = 1e-2;
            if (strategy == LevMarStrategyNone)strategy = LevMarStrategyMatlab;
            if (strategy > 4) {
                MATH21_ASSERT(0, "Strategy not support!");
            }
        }

        // x0 memory kept
        NumB run(VecR &x0) override {
            if (!maxIters)return 1;

            timer time;
            time.start();

            MatR temp_d_m;
            VecR x, x_new, h_neg;
            MatR A, A_lm;
            MatR A_inv;
            MatR Jtf;

            // strategy LevMarStrategySimple
            NumZ lambdaLg10 = -3;
            const NumR LOG10 = xjlog(10.);

            MATH21_ASSERT(x0.isColVector());

            cb.getx(x, x0);
            if (x.isEmpty())return 1;

            int nfJ = 0; // number of f and J

            //// compute the approximated Hessian matrix A, see ad_hessian
            // Jtf = dLt = L'.t = J.t * f
            // JtJ = H = A = J.t * J
            // loss2 = 2*L(x), L(x) = 1/2 * ||f(x)||^2
            NumR loss2;
            if (!cb.compute(x, &Jtf, &A, &loss2))return 0;
            nfJ += 2;

//            NumN m = J.nrows();
//            NumN n = J.ncols();
            NumN n = A.ncols();
            MATH21_ASSERT(n == x.size());

            const NumR rou1 = 0.25, rou2 = 0.75;
            const NumR beta = 2, gamma = 3, tao = 1e-8; // for Marquardt's and Nielsen's
            const NumR p = 3; // for Nielsen's
            NumR nu_n = beta; // for Nielsen's
            // ev_min ~= smallest eigenvalues of A
            NumR ev_min = 0.75; // for matlab
            NumR mu = 1;
            NumN i, iter = 0;
            NumN iter_accept = 0;

            if (0 && tao) {// choose mu0, slow
                NumR maxval = FLT_EPSILON;
                for (i = 1; i <= n; ++i)maxval = xjmax(maxval, xjabs(A(i, i)));
                mu = tao * maxval;
            }

            if (strategy == LevMarStrategySimple) {
                mu = xjexp(lambdaLg10 * LOG10); // lambda
            }

            if (logLevel != 0) {
                NumN nstars = 110;
                printf("%s\n", std::string(nstars, '*').c_str());
                printf("\titer\tnfJ\tloss^2\t\t||Jtf||\t\tx\t\t||h||\t\tmu\t\t\tev_min\n");
                printf("%s\n", std::string(nstars, '*').c_str());
            }

            timer time2;

            while (true) {
                //// step
                {
                    A_lm = A;

                    //// apply the damping factor to the Hessian matrix
                    // J.t * J + mu*diag(A), here J must be full rank, otherwise theory may break.
                    if (strategy == LevMarStrategySimple) {
                        // Marquardt's update
                        for (i = 1; i <= n; ++i)A_lm(i, i) *= 1 + mu; // slow for strategy Marquardt, Nielsen
                    } else {
                        // Levenberg's update
                        // J.t * J + mu*I
                        for (i = 1; i <= n; ++i)A_lm(i, i) += mu;
                    }

                    //// compute the updated x, (J'WJ + lambda*I) * dp = J'W(y-y_est)
                    // (J.t * J + mu*D) * h = -L'.t
                    time2.start();
                    math21_equation_solve_linear_equation_with_option(A_lm, Jtf, h_neg);
//                    math21_equation_solve_linear_equation_with_option(A_lm, Jtf, h_neg, m21_mat_decomp_svd);
                    time2.end();
                    if (logLevel >= 100)printf("solve_linear_equation time used %lf ms\n", time2.time());

                    // x_new = x + h
                    math21_op_subtract(x, h_neg, x_new);
                }

                //// evaluate the total distance error at the updated x.
                // 2*L(x+h) = ||f(x+h)||^2
                NumR loss2_new;
                if (!cb.compute(x_new, 0, 0, &loss2_new))return 0;
                nfJ++;

                NumR rou = 0;

                NumB accept = 0;
                ++iter;

                //// accept the updated x or discard
                if (strategy == LevMarStrategySimple) {
                    rou = loss2 - loss2_new;
                    if (rou > 0)accept = 1;
                    if (!accept) lambdaLg10 = xjmin(lambdaLg10 + 1, 16);
                    else lambdaLg10 = xjmax(lambdaLg10 - 1, -16);
                    mu = xjexp(lambdaLg10 * LOG10); // lambda
                } else {
                    // A*h + 2*L'.t
                    temp_d_m = Jtf;
                    math21_op_mat_mul_linear(-1, 2, A, h_neg, temp_d_m);
                    // gain_linear = L_linear(0) - L_linear(h) = 1/2 * h.t (mu * h - L'.t) = -1/2 * h.t * (Ah+2L't) > 0
                    // gain_linear2 = 2 * gain_linear = -h.t * (A*h + 2L'.t)
                    NumR gain_linear2 = math21_op_vector_inner_product(h_neg, temp_d_m);
//                    MATH21_ASSERT(gain_linear2 > 0, "Theory broken! gain_linear2 = " << gain_linear2);
                    if (gain_linear2 <= 0) {
                        m21log("iter", iter);
                        h_neg.log("h_neg");
                        temp_d_m.log("temp_d_m");
                        MATH21_ASSERT(0, "Theory broken! gain_linear2 = " << gain_linear2);
                    }

                    // rou = gain_factor = gain2 / gain_linear2 = gain / gain_linear = (L(x) - L(x+h))/(L_linear(0) - L_linear(h))
                    rou = (loss2 - loss2_new) / (fabs(gain_linear2) > FLT_EPSILON ? gain_linear2 : 1);

                    if (strategy == LevMarStrategyMarquardt) {// Marquardt's update strategy
                        if (rou > rou2)mu = mu / gamma;
                        else if (rou < rou1)mu = beta * mu;
                    } else if (strategy == LevMarStrategyNielsen) {// Nielsen's update strategy
                        if (rou > 0) {
                            mu = mu * xjmax(1 / gamma, 1 - (beta - 1) * xjpow(2 * rou - 1, p));
                            nu_n = beta;
                        } else {
                            mu = mu * nu_n;
                            nu_n = 2 * nu_n;
                        }
                    } else if (strategy == LevMarStrategyMatlab) {// matlab update strategy
                        if (rou > rou2) {
                            mu *= 0.5;
                            if (mu < ev_min)mu = 0;
                        } else if (rou < rou1) {
                            // find new nu if rou too low
                            // t = -h.t * L'.t = h.t (A+mu*D) h > 0
                            NumR t = math21_op_vector_inner_product(h_neg, Jtf);
                            MATH21_ASSERT(t > 0, "Theory broken2! t = " << t);

                            NumR nu = (loss2_new - loss2) / (fabs(t) > FLT_EPSILON ? t : 1) + 2;
                            nu = xjmin(xjmax(nu, 2.f), 10.f);

                            if (mu == 0) {
                                // J full rank <=> A full rank <=> A positive definite
                                // A is just positive semidefinite
//                            MATH21_ASSERT(m >= n, "Necessary condition not met!");

                                // from here A is full rank
                                // mu0 = tao * min(diag(A)) which is related to eigen-stuff.
                                math21_operator_inverse(A, A_inv);
                                NumR maxval = FLT_EPSILON;
                                for (i = 1; i <= n; ++i)maxval = xjmax(maxval, xjabs(A_inv(i, i)));
                                mu = ev_min = 1. / maxval;
                                nu *= 0.5;
                            }
                            mu *= nu;
                        }
                    }

                    if (rou > 0)accept = 1;
                }

                //// accepted
                // rou > 0 <=> loss2 > loss2_new
                if (accept) {//if rou > 0,    then x := x + h
                    ++iter_accept;

                    loss2 = loss2_new;
                    x.swap(x_new);

                    //// compute the approximated Hessian matrix A
                    // dLt = L'.t = J.t * f
                    // A = J.t * J
                    if (!cb.compute(x, &Jtf, &A, 0))return 0;
                    nfJ += 2;
                }

                // ||L'(x)|| <= epsilon1, ||h|| <= epsilon2 * ||x||, k < kmax
                // Must not rely on norm_value, because it can be large number.
                bool proceed;

                NumR norm_Jtf = math21_op_vector_norm(Jtf, "inf");
                NumR norm_h;

                if (strategy == LevMarStrategySimple) {
                    // see math21_op_vector_distance_relative
                    NumR dis_relative = math21_op_vector_norm(h_neg, 2) / math21_op_vector_norm(x, 2);
                    proceed = iter < maxIters && dis_relative >= epsh;
                    norm_h = dis_relative;
                } else {
                    NumR normInf_h = math21_op_vector_norm(h_neg, "inf");
                    proceed = iter < maxIters && normInf_h >= epsh && norm_Jtf >= epsNormJtf;
                    norm_h = normInf_h;
                }

                if (logLevel != 0 && (iter % logLevel == 0 || iter == 1 || !proceed)) {
                    printf("%c%10d %5d %15.4e %15.4e %14.4e %14.4e %14.4e %15.4e\n",
                           (proceed ? ' ' : '*'), iter, nfJ, loss2, norm_Jtf, x(1), norm_h, mu, ev_min);
                }

                if (!proceed)break;
            }

            cb.setx(x, x0);// x0 memory kept

            time.end();
            if (logLevel > 1) {
                printf("time used %lf ms\n", time.time());
            }
            return 1;
        }
    };

    OptDetail *math21_opt_create_LevMarq(const OptParasLevMarq &paras, FunctionNd *cb) {
        return new OptLevMarq(paras, cb);
    }

    OptDetail *math21_opt_create_LevMarq(const OptParasLevMarq &paras, OptCallbackLevMarq *cb) {
        return new OptLevMarq(paras, cb);
    }

    OptDetail *math21_opt_create_LevMarq(const OptParasLevMarq &paras, OptCallbackSparseLevMarq *cb) {
        return new OptLevMarq(paras, cb);
    }

    void math21_opt_destroy_LevMarq(OptDetail *lm) {
        delete (OptLevMarq *) (lm);
    }
}