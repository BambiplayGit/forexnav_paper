/*
    MIT License
    Copyright (c) 2021 Zhepei Wang (wangzhepei@live.com)
*/

#ifndef LBFGS_HPP
#define LBFGS_HPP

#include <Eigen/Eigen>
#include <cmath>
#include <algorithm>

namespace lbfgs
{
    struct lbfgs_parameter_t
    {
        int mem_size = 8;
        double g_epsilon = 1.0e-5;
        int past = 3;
        double delta = 1.0e-6;
        int max_iterations = 0;
        int max_linesearch = 64;
        double min_step = 1.0e-20;
        double max_step = 1.0e+20;
        double f_dec_coeff = 1.0e-4;
        double s_curv_coeff = 0.9;
        double cautious_factor = 1.0e-6;
        double machine_prec = 1.0e-16;
    };

    enum
    {
        LBFGS_CONVERGENCE = 0,
        LBFGS_STOP,
        LBFGS_CANCELED,
        LBFGSERR_UNKNOWNERROR = -1024,
        LBFGSERR_INVALID_N,
        LBFGSERR_INVALID_MEMSIZE,
        LBFGSERR_INVALID_GEPSILON,
        LBFGSERR_INVALID_TESTPERIOD,
        LBFGSERR_INVALID_DELTA,
        LBFGSERR_INVALID_MINSTEP,
        LBFGSERR_INVALID_MAXSTEP,
        LBFGSERR_INVALID_FDECCOEFF,
        LBFGSERR_INVALID_SCURVCOEFF,
        LBFGSERR_INVALID_MACHINEPREC,
        LBFGSERR_INVALID_MAXLINESEARCH,
        LBFGSERR_INVALID_FUNCVAL,
        LBFGSERR_MINIMUMSTEP,
        LBFGSERR_MAXIMUMSTEP,
        LBFGSERR_MAXIMUMLINESEARCH,
        LBFGSERR_MAXIMUMITERATION,
        LBFGSERR_WIDTHTOOSMALL,
        LBFGSERR_INVALIDPARAMETERS,
        LBFGSERR_INCREASEGRADIENT,
    };

    typedef double (*lbfgs_evaluate_t)(void *instance,
                                       const Eigen::VectorXd &x,
                                       Eigen::VectorXd &g);

    typedef double (*lbfgs_stepbound_t)(void *instance,
                                        const Eigen::VectorXd &xp,
                                        const Eigen::VectorXd &d);

    typedef int (*lbfgs_progress_t)(void *instance,
                                    const Eigen::VectorXd &x,
                                    const Eigen::VectorXd &g,
                                    const double fx,
                                    const double step,
                                    const int k,
                                    const int ls);

    struct callback_data_t
    {
        void *instance = nullptr;
        lbfgs_evaluate_t proc_evaluate = nullptr;
        lbfgs_stepbound_t proc_stepbound = nullptr;
        lbfgs_progress_t proc_progress = nullptr;
    };

    inline int line_search_lewisoverton(Eigen::VectorXd &x,
                                        double &f,
                                        Eigen::VectorXd &g,
                                        double &stp,
                                        const Eigen::VectorXd &s,
                                        const Eigen::VectorXd &xp,
                                        const Eigen::VectorXd &gp,
                                        const double stpmin,
                                        const double stpmax,
                                        const callback_data_t &cd,
                                        const lbfgs_parameter_t &param)
    {
        int count = 0;
        bool brackt = false, touched = false;
        double finit, dginit, dgtest, dstest;
        double mu = 0.0, nu = stpmax;

        if (!(stp > 0.0))
        {
            return LBFGSERR_INVALIDPARAMETERS;
        }

        dginit = gp.dot(s);

        if (0.0 < dginit)
        {
            return LBFGSERR_INCREASEGRADIENT;
        }

        finit = f;
        dgtest = param.f_dec_coeff * dginit;
        dstest = param.s_curv_coeff * dginit;

        while (true)
        {
            x = xp + stp * s;
            f = cd.proc_evaluate(cd.instance, x, g);
            ++count;

            if (std::isinf(f) || std::isnan(f))
            {
                return LBFGSERR_INVALID_FUNCVAL;
            }
            if (f > finit + stp * dgtest)
            {
                nu = stp;
                brackt = true;
            }
            else
            {
                if (g.dot(s) < dstest)
                {
                    mu = stp;
                }
                else
                {
                    return count;
                }
            }
            if (param.max_linesearch <= count)
            {
                return LBFGSERR_MAXIMUMLINESEARCH;
            }
            if (brackt && (nu - mu) < param.machine_prec * nu)
            {
                return LBFGSERR_WIDTHTOOSMALL;
            }

            if (brackt)
            {
                stp = 0.5 * (mu + nu);
            }
            else
            {
                stp *= 2.0;
            }

            if (stp < stpmin)
            {
                return LBFGSERR_MINIMUMSTEP;
            }
            if (stp > stpmax)
            {
                if (touched)
                {
                    return LBFGSERR_MAXIMUMSTEP;
                }
                else
                {
                    touched = true;
                    stp = stpmax;
                }
            }
        }
    }

    inline int lbfgs_optimize(Eigen::VectorXd &x,
                              double &f,
                              lbfgs_evaluate_t proc_evaluate,
                              lbfgs_stepbound_t proc_stepbound,
                              lbfgs_progress_t proc_progress,
                              void *instance,
                              const lbfgs_parameter_t &param)
    {
        int ret, i, j, k, ls, end, bound;
        double step, step_min, step_max, fx, ys, yy;
        double gnorm_inf, xnorm_inf, beta, rate, cau;

        const int n = x.size();
        const int m = param.mem_size;

        if (n <= 0) return LBFGSERR_INVALID_N;
        if (m <= 0) return LBFGSERR_INVALID_MEMSIZE;
        if (param.g_epsilon < 0.0) return LBFGSERR_INVALID_GEPSILON;
        if (param.past < 0) return LBFGSERR_INVALID_TESTPERIOD;
        if (param.delta < 0.0) return LBFGSERR_INVALID_DELTA;
        if (param.min_step < 0.0) return LBFGSERR_INVALID_MINSTEP;
        if (param.max_step < param.min_step) return LBFGSERR_INVALID_MAXSTEP;
        if (!(param.f_dec_coeff > 0.0 && param.f_dec_coeff < 1.0)) return LBFGSERR_INVALID_FDECCOEFF;
        if (!(param.s_curv_coeff < 1.0 && param.s_curv_coeff > param.f_dec_coeff)) return LBFGSERR_INVALID_SCURVCOEFF;
        if (!(param.machine_prec > 0.0)) return LBFGSERR_INVALID_MACHINEPREC;
        if (param.max_linesearch <= 0) return LBFGSERR_INVALID_MAXLINESEARCH;

        Eigen::VectorXd xp(n);
        Eigen::VectorXd g(n);
        Eigen::VectorXd gp(n);
        Eigen::VectorXd d(n);
        Eigen::VectorXd pf(std::max(1, param.past));

        Eigen::VectorXd lm_alpha = Eigen::VectorXd::Zero(m);
        Eigen::MatrixXd lm_s = Eigen::MatrixXd::Zero(n, m);
        Eigen::MatrixXd lm_y = Eigen::MatrixXd::Zero(n, m);
        Eigen::VectorXd lm_ys = Eigen::VectorXd::Zero(m);

        callback_data_t cd;
        cd.instance = instance;
        cd.proc_evaluate = proc_evaluate;
        cd.proc_stepbound = proc_stepbound;
        cd.proc_progress = proc_progress;

        fx = cd.proc_evaluate(cd.instance, x, g);
        pf(0) = fx;
        d = -g;

        gnorm_inf = g.cwiseAbs().maxCoeff();
        xnorm_inf = x.cwiseAbs().maxCoeff();

        if (gnorm_inf / std::max(1.0, xnorm_inf) < param.g_epsilon)
        {
            ret = LBFGS_CONVERGENCE;
        }
        else
        {
            step = 1.0 / d.norm();
            k = 1;
            end = 0;
            bound = 0;

            while (true)
            {
                xp = x;
                gp = g;

                step_min = param.min_step;
                step_max = param.max_step;
                if (cd.proc_stepbound)
                {
                    step_max = cd.proc_stepbound(cd.instance, xp, d);
                    step_max = step_max < param.max_step ? step_max : param.max_step;
                    step = step < step_max ? step : 0.5 * step_max;
                }

                ls = line_search_lewisoverton(x, fx, g, step, d, xp, gp, step_min, step_max, cd, param);

                if (ls < 0)
                {
                    x = xp;
                    g = gp;
                    ret = ls;
                    break;
                }

                if (cd.proc_progress)
                {
                    if (cd.proc_progress(cd.instance, x, g, fx, step, k, ls))
                    {
                        ret = LBFGS_CANCELED;
                        break;
                    }
                }

                gnorm_inf = g.cwiseAbs().maxCoeff();
                xnorm_inf = x.cwiseAbs().maxCoeff();
                if (gnorm_inf / std::max(1.0, xnorm_inf) < param.g_epsilon)
                {
                    ret = LBFGS_CONVERGENCE;
                    break;
                }

                if (0 < param.past)
                {
                    if (param.past <= k)
                    {
                        rate = std::fabs(pf(k % param.past) - fx) / std::max(1.0, std::fabs(fx));
                        if (rate < param.delta)
                        {
                            ret = LBFGS_STOP;
                            break;
                        }
                    }
                    pf(k % param.past) = fx;
                }

                if (param.max_iterations != 0 && param.max_iterations <= k)
                {
                    ret = LBFGSERR_MAXIMUMITERATION;
                    break;
                }

                ++k;

                lm_s.col(end) = x - xp;
                lm_y.col(end) = g - gp;

                ys = lm_y.col(end).dot(lm_s.col(end));
                yy = lm_y.col(end).squaredNorm();
                lm_ys(end) = ys;

                d = -g;

                cau = lm_s.col(end).squaredNorm() * gp.norm() * param.cautious_factor;

                if (ys > cau)
                {
                    ++bound;
                    bound = m < bound ? m : bound;
                    end = (end + 1) % m;

                    j = end;
                    for (i = 0; i < bound; ++i)
                    {
                        j = (j + m - 1) % m;
                        lm_alpha(j) = lm_s.col(j).dot(d) / lm_ys(j);
                        d += (-lm_alpha(j)) * lm_y.col(j);
                    }

                    d *= ys / yy;

                    for (i = 0; i < bound; ++i)
                    {
                        beta = lm_y.col(j).dot(d) / lm_ys(j);
                        d += (lm_alpha(j) - beta) * lm_s.col(j);
                        j = (j + 1) % m;
                    }
                }

                step = 1.0;
            }
        }

        f = fx;
        return ret;
    }

    inline const char *lbfgs_strerror(const int err)
    {
        switch (err)
        {
        case LBFGS_CONVERGENCE: return "Success: reached convergence.";
        case LBFGS_STOP: return "Success: met stopping criteria.";
        case LBFGS_CANCELED: return "Canceled by monitor callback.";
        case LBFGSERR_INVALID_FUNCVAL: return "Function value became NaN or Inf.";
        case LBFGSERR_MINIMUMSTEP: return "Line-search step too small.";
        case LBFGSERR_MAXIMUMSTEP: return "Line-search step too large.";
        case LBFGSERR_MAXIMUMLINESEARCH: return "Line search max iterations.";
        case LBFGSERR_MAXIMUMITERATION: return "Max iterations reached.";
        case LBFGSERR_INCREASEGRADIENT: return "Search direction increases cost.";
        default: return "(unknown error)";
        }
    }

} // namespace lbfgs

#endif
