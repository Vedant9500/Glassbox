#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Dense>

namespace sr {

// Batch-12 finite guards (M-339): generous modeling-neutral caps. Engine trig
// omegas live in [-8, 8] (evolution.h); 1e4 is headroom, not a modeling claim
// — it rejects only non-finite/runaway values while leaving legit fits alone.
constexpr double kRefineFreqOmegaMin = 0.01;
constexpr double kRefineFreqOmegaMax = 1e4;
constexpr double kRefineRationalOmegaMax = 1e4;
constexpr double kRefineRationalCMin = 1e-6;
constexpr double kRefineRationalCMax = 1e6;

template <typename Derived>
inline bool all_entries_finite(const Eigen::DenseBase<Derived>& m) {
    for (int i = 0; i < m.size(); ++i)
        if (!std::isfinite(m.derived()(i))) return false;
    return true;
}


// Apply sqrt(sample weights) row scaling for weighted least squares.
// Empty / wrong-size sw is a no-op. Non-positive weights clamp to 0.
inline void apply_sqrt_sample_weights(Eigen::MatrixXd& X, Eigen::VectorXd& y,
                                      const Eigen::VectorXd& sw) {
    if (sw.size() != y.size() || sw.size() == 0) return;
    const int n = static_cast<int>(y.size());
    for (int i = 0; i < n; ++i) {
        double w = sw(i);
        if (!std::isfinite(w) || w <= 0.0) {
            X.row(i).setZero();
            y(i) = 0.0;
            continue;
        }
        double s = std::sqrt(w);
        X.row(i) *= s;
        y(i) *= s;
    }
}

struct ElasticNetResult {
    Eigen::VectorXd weights;
    double mse = 1e15;
    // Batch-12 diagnostics (M-338; §3.399): convergence metadata. mse is the
    // data-fit term — weighted sum(w r^2)/sum(w) when sample_weight is given,
    // plain mean square otherwise. penalized_objective adds the l1/l2
    // penalties: the function coordinate descent actually minimizes.
    int iters_done = 0;
    bool converged = false;
    double penalized_objective = 1e15;
};

inline ElasticNetResult elastic_net_cd_cpp(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                                           double l1_weight, double l2_weight, 
                                           int max_iter = 1000, double tol = 1e-7,
                                           const Eigen::VectorXd& initial_w = Eigen::VectorXd(),
                                           const Eigen::VectorXd& sample_weight = Eigen::VectorXd()) {
    // §3.94 implemented objective: MSE + l1_weight*||w||_1 + l2_weight*||w||_2^2
    // with MSE = ||r||^2/n (threshold = l1_weight/2, NOT sklearn's
    // (1/2n)||r||^2 + alpha||w||_1 whose threshold is alpha). Callers passing
    // sklearn alpha must scale explicitly; pybind boundary documents this.
    // §3.403: X/y arrive sqrt(w)-scaled (apply_sqrt_sample_weights), so the
    // scaled residual sum is sum(w r_orig^2). With sample_weight given the
    // returned mse is sum(w r^2)/sum(w) (residual_mse_weighted contract);
    // otherwise legacy ||r||^2/n. Same data ⇒ same denominator, so
    // multi-start ranking is unaffected by the convention.
    int n = X.rows();
    int p = X.cols();
    
    Eigen::VectorXd w = initial_w;
    if (w.size() != p) w = Eigen::VectorXd::Zero(p);
    if (n == 0 || p == 0) return {w, 0.0, 0, true, 0.0};

    // §3.400: fail loud (inf mse) on non-finite data instead of silently
    // iterating poisoned residuals to a bogus zero solution.
    if (!all_entries_finite(X) || !all_entries_finite(y) || !all_entries_finite(w)) {
        ElasticNetResult bad;
        bad.weights = Eigen::VectorXd::Zero(p);
        bad.mse = std::numeric_limits<double>::infinity();
        bad.penalized_objective = std::numeric_limits<double>::infinity();
        return bad;
    }

    const bool has_w = (sample_weight.size() == y.size());
    double den = static_cast<double>(n);
    if (has_w) {
        const double s = sample_weight.sum();
        if (s > 0.0 && std::isfinite(s)) den = s;
    }

    auto penalized = [&](const Eigen::VectorXd& ww, double mse_val) {
        return mse_val + l1_weight * ww.cwiseAbs().sum() + l2_weight * ww.squaredNorm();
    };
    
    // Precompute z_j = 1/n ||X_j||^2
    Eigen::VectorXd z(p);
    for (int j = 0; j < p; ++j) {
        z(j) = X.col(j).squaredNorm() / n;
    }
    
    Eigen::VectorXd res = y - X * w;
    double l1_half = l1_weight / 2.0;
    double prev_obj = penalized(w, res.squaredNorm() / den);

    ElasticNetResult out;
    out.weights = w;
    out.mse = res.squaredNorm() / den;
    out.penalized_objective = prev_obj;
    
    for (int iter = 0; iter < max_iter; ++iter) {
        double max_change = 0.0;
        
        for (int j = 0; j < p; ++j) {
            if (z(j) < 1e-12) continue; // skip zero columns
            
            double old_w = w(j);
            double rho = X.col(j).dot(res) / n + old_w * z(j);
            
            double new_w = 0.0;
            if (rho > l1_half) {
                new_w = (rho - l1_half) / (z(j) + l2_weight);
            } else if (rho < -l1_half) {
                new_w = (rho + l1_half) / (z(j) + l2_weight);
            }
            
            if (new_w != old_w) {
                res -= X.col(j) * (new_w - old_w);
                w(j) = new_w;
                max_change = std::max(max_change, std::abs(new_w - old_w));
            }
        }

        double mse_now = res.squaredNorm() / den;
        double cur_obj = penalized(w, mse_now);
        out.iters_done = iter + 1;
        // §3.399: dual stop — legacy coeff tolerance AND penalized-objective
        // stall. Continuing while the objective still drops fixes premature
        // stops on tiny-scale columns; flat-objective runs stop as before.
        const double obj_drop = prev_obj - cur_obj;
        if (max_change < tol && obj_drop <= tol * (1.0 + std::abs(cur_obj))) {
            out.converged = true;
            out.weights = w;
            out.mse = mse_now;
            out.penalized_objective = cur_obj;
            break;
        }
        prev_obj = cur_obj;
        out.weights = w;
        out.mse = mse_now;
        out.penalized_objective = cur_obj;
    }
    
    return out;
}

inline ElasticNetResult multi_start_elastic_net(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                                                double l1_weight, double l2_weight, 
                                                int n_starts = 5, double init_scale = 0.1,
                                                int max_iter = 1000,
                                                const Eigen::VectorXd& sample_weight = Eigen::VectorXd(),
                                                unsigned int seed = 42) {
    int p = X.cols();
    ElasticNetResult best_res;
    best_res.mse = std::numeric_limits<double>::infinity();
    best_res.penalized_objective = std::numeric_limits<double>::infinity();
    best_res.weights = Eigen::VectorXd::Zero(p);
    
    // M-136: start seed exposed (default 42 = legacy sequence). Callers
    // wanting dataset-decorrelated starts pass their own seed.
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, init_scale);
    
    for (int i = 0; i < n_starts; ++i) {
        Eigen::VectorXd init_w = Eigen::VectorXd::Zero(p);
        if (i > 0) {
            for (int j = 0; j < p; ++j) init_w(j) = dist(gen);
        }
        
        auto res = elastic_net_cd_cpp(X, y, l1_weight, l2_weight, max_iter, 1e-7, init_w,
                                      sample_weight);
        // §3.401: compare the penalized objective CD minimizes, not raw MSE —
        // a lower-MSE start can be a worse solution of the regularized problem.
        if (res.penalized_objective < best_res.penalized_objective) {
            best_res = res;
        }
    }
    return best_res;
}

// Optional sample_weight: when non-empty and length matches y, run weighted LS
// via sqrt(w) row scaling (S5-9). Empty VectorXd keeps legacy unweighted path.
// §3.95: returned mse is normalized weighted MSE sum(w r^2)/sum(w), not the
// scaled sum/n (which dilutes excluded zero-weight rows).
inline ElasticNetResult iterative_elastic_net(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                                              double l1_weight, double l2_weight, 
                                              int n_starts = 3, int n_iterations = 3,
                                              double prune_threshold = 0.05,
                                              int max_iter = 1000,
                                              const Eigen::VectorXd& sample_weight = Eigen::VectorXd(),
                                              unsigned int seed = 42) {
    Eigen::MatrixXd X_work = X;
    Eigen::VectorXd y_work = y;
    apply_sqrt_sample_weights(X_work, y_work, sample_weight);

    int p = X_work.cols();
    const double l1_half = l1_weight / 2.0;
    std::vector<bool> active_mask(p, true);
    
    ElasticNetResult best_res;
    best_res.mse = std::numeric_limits<double>::infinity();
    best_res.penalized_objective = std::numeric_limits<double>::infinity();
    best_res.weights = Eigen::VectorXd::Zero(p);
    
    for (int it = 0; it < n_iterations; ++it) {
        int num_active = 0;
        for (bool a : active_mask) if (a) num_active++;
        if (num_active == 0) break;
        
        Eigen::MatrixXd X_active(X_work.rows(), num_active);
        std::vector<int> active_indices;
        int col_idx = 0;
        for (int j = 0; j < p; ++j) {
            if (active_mask[j]) {
                X_active.col(col_idx++) = X_work.col(j);
                active_indices.push_back(j);
            }
        }
        
        auto res = multi_start_elastic_net(X_active, y_work, l1_weight, l2_weight, n_starts,
                                           0.1, max_iter, sample_weight, seed);
        
        Eigen::VectorXd full_weights = Eigen::VectorXd::Zero(p);
        for (int j = 0; j < num_active; ++j) {
            full_weights(active_indices[j]) = res.weights(j);
        }
        
        // §3.403: res.mse already carries the weighted contract from
        // elastic_net_cd_cpp (single source of truth — no renormalization here).
        if (res.mse < best_res.mse) {
            best_res.mse = res.mse;
            best_res.weights = full_weights;
            best_res.iters_done = res.iters_done;
            best_res.converged = res.converged;
            best_res.penalized_objective = res.penalized_objective;
        }
        
        double max_w = full_weights.cwiseAbs().maxCoeff();
        if (max_w <= 0) break;
        
        double thresh = prune_threshold * max_w;
        std::vector<bool> new_mask(p, false);
        bool any_active = false;
        
        for (int j = 0; j < p; ++j) {
            if (std::abs(full_weights(j)) > thresh) {
                new_mask[j] = true;
                any_active = true;
            }
        }

        // §3.402: KKT reentry — a pruned column with |X_j·res|/n above the
        // soft threshold (same /n convention as the CD sweep) belongs back
        // in the active set. best_res already preserves the unpruned optimum,
        // so this only affects future iterations, bounded by n_iterations.
        {
            Eigen::VectorXd res_full = y_work - X_work * full_weights;
            const double rn = static_cast<double>(X_work.rows());
            for (int j = 0; j < p; ++j) {
                if (active_mask[j] && !new_mask[j] && rn > 0) {
                    const double score =
                        std::abs(X_work.col(j).dot(res_full)) / rn;
                    if (score > l1_half) {
                        new_mask[j] = true;
                        any_active = true;
                    }
                }
            }
        }

        bool changed = false;
        for (int j = 0; j < p; ++j) {
            if (new_mask[j] != active_mask[j]) { changed = true; break; }
        }
        
        if (!any_active || !changed) break;
        active_mask = new_mask;
    }
    
    return best_res;
}

// Evaluate linear coefficients given non-linear features
// X is N x M, y is N x 1. Returns M x 1 coefficients.
inline Eigen::VectorXd solve_linear(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    // using SVD or QR
    return X.colPivHouseholderQr().solve(y);
}

// S5-9: weighted least squares via sqrt(w) row scaling (same pattern as elastic net).
inline Eigen::VectorXd solve_linear_weighted(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& sample_weight
) {
    if (sample_weight.size() != y.size() || sample_weight.size() == 0) {
        return solve_linear(X, y);
    }
    Eigen::MatrixXd Xs = X;
    Eigen::VectorXd ys = y;
    apply_sqrt_sample_weights(Xs, ys, sample_weight);
    return solve_linear(Xs, ys);
}

// Weighted MSE: sum(w r^2)/sum(w). Empty weights => plain mean square error.
inline double residual_mse_weighted(
    const Eigen::VectorXd& pred,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& sample_weight
) {
    const int n = static_cast<int>(y.size());
    if (n <= 0) return 0.0;
    Eigen::ArrayXd r2 = (pred - y).array().square();
    if (sample_weight.size() != y.size() || sample_weight.size() == 0) {
        return r2.mean();
    }
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < n; ++i) {
        double w = sample_weight(i);
        if (!std::isfinite(w) || w <= 0.0) continue;
        num += w * r2(i);
        den += w;
    }
    if (!(den > 0.0) || !std::isfinite(den)) {
        return r2.mean();
    }
    return num / den;
}

// Frequency refinement: c0 + c1*x + c2*x^2 + sum_i [a_i*sin(omega_i*x) + b_i*cos(omega_i*x)]
// We want to optimize omegas.
struct FreqResult {
    std::vector<double> omegas;
    double mse;
    // §3.405: true once a finite evaluation updates best (all-fail runs
    // return the sanitized init with mse=inf instead of a 1e9 sentinel).
    bool success = false;
};

inline FreqResult refine_frequencies_cpp(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& y,
    std::vector<double> initial_omegas,
    int steps = 100,
    double lr = 0.1,
    const Eigen::VectorXd& sample_weight = Eigen::VectorXd()
) {
    int n = static_cast<int>(x.size());
    int k = static_cast<int>(initial_omegas.size());
    std::vector<double> omegas = initial_omegas;

    // M-339/§3.405: sanitize + bound the init; best starts at inf so the
    // first finite evaluation wins (never a 1e9 sentinel or garbage state).
    for (double& o : omegas) {
        if (!std::isfinite(o)) o = 1.0;
        o = std::clamp(o, kRefineFreqOmegaMin, kRefineFreqOmegaMax);
    }
    double best_mse = std::numeric_limits<double>::infinity();
    bool success = false;
    std::vector<double> best_omegas = omegas;
    
    for (int step = 0; step < steps; ++step) {
        // Build feature matrix
        int num_features = 3 + 2 * k; // 1, x, x^2, sin(w_i x), cos(w_i x)
        Eigen::MatrixXd X(n, num_features);
        X.col(0) = Eigen::VectorXd::Ones(n);
        X.col(1) = x;
        X.col(2) = x.array().square().matrix();
        
        for (int i = 0; i < k; ++i) {
            X.col(3 + 2*i) = (omegas[i] * x.array()).sin().matrix();
            X.col(4 + 2*i) = (omegas[i] * x.array()).cos().matrix();
        }
        
        // S5-9: WLS linear coeffs + weighted MSE objective when sample_weight set.
        Eigen::VectorXd coeffs = solve_linear_weighted(X, y, sample_weight);
        Eigen::VectorXd pred = X * coeffs;
        double mse = residual_mse_weighted(pred, y, sample_weight);
        
        if (mse < best_mse) {
            best_mse = mse;
            best_omegas = omegas;
            success = true;
        }
        
        // Gradient descent on omegas via finite differences
        // §3.97: scale-adaptive FD step (fixed 1e-4 under-resolves large
        // |omega| and over-steps near zero). Relative step with floor.
        // §3.404: probe on local copies (X_f/X_b), never by mutating the
        // shared design matrix — the old perturb/restore left X corrupted
        // on any exception or early return between the two.
        std::vector<double> grads(k, 0.0);
        
        for (int i = 0; i < k; ++i) {
            const double eps = 1e-4 * (1.0 + std::abs(omegas[i]));
            const double o_fwd = omegas[i] + eps;
            const double o_bwd = omegas[i] - eps;
            Eigen::MatrixXd X_f = X;
            X_f.col(3 + 2*i) = (o_fwd * x.array()).sin().matrix();
            X_f.col(4 + 2*i) = (o_fwd * x.array()).cos().matrix();
            Eigen::VectorXd c_fwd = solve_linear_weighted(X_f, y, sample_weight);
            double mse_fwd = residual_mse_weighted(X_f * c_fwd, y, sample_weight);

            Eigen::MatrixXd X_b = X;
            X_b.col(3 + 2*i) = (o_bwd * x.array()).sin().matrix();
            X_b.col(4 + 2*i) = (o_bwd * x.array()).cos().matrix();
            Eigen::VectorXd c_bwd = solve_linear_weighted(X_b, y, sample_weight);
            double mse_bwd = residual_mse_weighted(X_b * c_bwd, y, sample_weight);
            
            grads[i] = (mse_fwd - mse_bwd) / (2 * eps);
        }
        
        // Update omegas with projected gradient descent (§3.98): the FD
        // gradient is evaluated at the pre-clamp value, so project the step
        // onto the feasible set rather than bare-clamping afterwards. Skip
        // non-finite FD gradients instead of stepping on NaN. M-339: upper
        // projection + finite-trial guard — omega can no longer run away
        // to inf or persist as NaN.
        for (int i = 0; i < k; ++i) {
            if (!std::isfinite(grads[i])) continue;
            double trial = omegas[i] - lr * grads[i];
            if (!std::isfinite(trial)) continue;
            omegas[i] = std::clamp(trial, kRefineFreqOmegaMin, kRefineFreqOmegaMax);
        }
    }
    
    return {best_omegas, best_mse, success};
}

struct PowerResult {
    std::vector<double> powers;
    std::vector<double> coeffs;
    double constant = 0.0;
    double linear = 0.0;
    std::vector<double> periodic_coeffs; // sin_1, cos_1, ...
    double mse = 1e15;
};

// sign(x) * |x|^p (parity preserving)
// §3.1: shares eval.h canonical parity tol (1e-9) + eps (1e-10).
// Keep in sync with canonical_signed_power(); no independent formula.
inline Eigen::VectorXd safe_power(const Eigen::VectorXd& x, double p) {
    Eigen::VectorXd abs_pow = (x.array().abs() + 1e-10).pow(p);
    double p_round = std::round(p);
    bool is_even = (std::abs(p - p_round) < 1e-9) && (static_cast<long long>(p_round) % 2 == 0);
    if (is_even) {
        return abs_pow;
    } else {
        return (x.array().sign() * abs_pow.array()).matrix();
    }
}

inline PowerResult refine_powers_model_cpp(
    const Eigen::VectorXd& x_valid,
    const Eigen::VectorXd& y_valid,
    std::vector<double> powers,
    const std::vector<double>& omegas,
    int steps = 200,
    double lr = 0.05,
    const Eigen::VectorXd& sample_weight = Eigen::VectorXd(),
    double p_min_bound = -2.0,
    double p_max_bound = 5.0
) {
    int n = static_cast<int>(x_valid.size());
    int num_p = static_cast<int>(powers.size());
    int num_w = static_cast<int>(omegas.size());
    int num_features = 2 + num_p + 2 * num_w; // 1, x, p_i, sin(w_i), cos(w_i)

    // §3.332: caller-configured domain (was hard-coded [-2, 5] regardless
    // of p_min/p_max). Invalid bounds fall back to the legacy hard range.
    double plo = p_min_bound, phi_ = p_max_bound;
    if (!(plo < phi_) || !std::isfinite(plo) || !std::isfinite(phi_)) {
        plo = -2.0;
        phi_ = 5.0;
    }

    auto build_design = [&](const std::vector<double>& pw, Eigen::MatrixXd& X) {
        X.resize(n, num_features);
        X.col(0) = Eigen::VectorXd::Ones(n);
        X.col(1) = x_valid;
        for (int i = 0; i < num_p; ++i) {
            X.col(2 + i) = safe_power(x_valid, pw[i]);
        }
        for (int i = 0; i < num_w; ++i) {
            X.col(2 + num_p + 2*i) = (omegas[i] * x_valid.array()).sin().matrix();
            X.col(2 + num_p + 2*i + 1) = (omegas[i] * x_valid.array()).cos().matrix();
        }
    };

    double best_mse;
    std::vector<double> best_powers = powers;
    Eigen::VectorXd best_coeffs;
    {
        // §3.332: initialize from the first actual evaluation. Was 1e9, so
        // all-non-finite runs implicitly "accepted" the initial powers while
        // reporting 1e9 as the best objective.
        Eigen::MatrixXd X0;
        build_design(powers, X0);
        Eigen::VectorXd c0 = solve_linear_weighted(X0, y_valid, sample_weight);
        Eigen::VectorXd pred0 = X0 * c0;
        best_mse = residual_mse_weighted(pred0, y_valid, sample_weight);
        if (!std::isfinite(best_mse)) best_mse = std::numeric_limits<double>::infinity();
        else best_coeffs = c0;
    }
    
    for (int step = 0; step < steps; ++step) {
        Eigen::MatrixXd X(n, num_features);
        build_design(powers, X);
        
        Eigen::VectorXd c = solve_linear_weighted(X, y_valid, sample_weight);
        Eigen::VectorXd pred = X * c;
        double mse = residual_mse_weighted(pred, y_valid, sample_weight);
        
        if (mse < best_mse) {
            best_mse = mse;
            best_powers = powers;
            best_coeffs = c;
        }
        
        double eps = 1e-4;
        std::vector<double> grads(num_p, 0.0);
        
        for (int i = 0; i < num_p; ++i) {
            powers[i] += eps;
            X.col(2 + i) = safe_power(x_valid, powers[i]);
            Eigen::VectorXd c_fwd = solve_linear_weighted(X, y_valid, sample_weight);
            double mse_fwd = residual_mse_weighted(X * c_fwd, y_valid, sample_weight);
            
            powers[i] -= 2*eps;
            X.col(2 + i) = safe_power(x_valid, powers[i]);
            Eigen::VectorXd c_bwd = solve_linear_weighted(X, y_valid, sample_weight);
            double mse_bwd = residual_mse_weighted(X * c_bwd, y_valid, sample_weight);
            
            powers[i] += eps;
            X.col(2 + i) = safe_power(x_valid, powers[i]);
            
            grads[i] = (mse_fwd - mse_bwd) / (2 * eps);
        }
        
        // Projected bounds + finite-gradient guard (§3.98/§3.139): NaN powers
        // or non-finite FD probes must not step the search into NaN.
        // §3.332: project onto the caller domain (was hard-coded [-2, 5]).
        for (int i = 0; i < num_p; ++i) {
            if (!std::isfinite(grads[i])) continue;
            powers[i] -= lr * grads[i];
            if (powers[i] < plo) powers[i] = plo;
            if (powers[i] > phi_) powers[i] = phi_;
        }
    }
    
    PowerResult res;
    res.mse = best_mse;
    res.powers = best_powers;
    if (best_coeffs.size() > 0) {
        res.constant = best_coeffs[0];
        res.linear = best_coeffs[1];
        for (int i = 0; i < num_p; ++i) res.coeffs.push_back(best_coeffs[2 + i]);
        for (int i = 0; i < num_w; ++i) {
            res.periodic_coeffs.push_back(best_coeffs[2 + num_p + 2*i]);
            res.periodic_coeffs.push_back(best_coeffs[2 + num_p + 2*i + 1]);
        }
    }
    return res;
}

struct PeriodicRationalResult {
    double omega;
    double c;
    double a; // sin
    double b; // cos
    double d; // linear
    double e; // const
    double mse;
    // §3.405: true once a finite evaluation updates best (all-fail runs
    // return the sanitized init with mse=inf instead of a 1e9 sentinel
    // plus uninitialized fields).
    bool success = false;
};

inline PeriodicRationalResult refine_periodic_rational_cpp(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& y,
    double omega0,
    double c0,
    int steps = 200,
    double lr = 0.05,
    const Eigen::VectorXd& sample_weight = Eigen::VectorXd()
) {
    int n = static_cast<int>(x.size());
    // M-339/§3.405: sanitize + bound the init (omega keeps its sign freedom,
    // only magnitude is capped; c stays strictly positive for denominator
    // safety). best is fully initialized from the sanitized init with
    // mse=inf, so all-fail runs return honest state, never garbage.
    double omega = std::isfinite(omega0)
        ? std::clamp(omega0, -kRefineRationalOmegaMax, kRefineRationalOmegaMax)
        : 1.0;
    double c_val = std::isfinite(c0)
        ? std::clamp(c0, kRefineRationalCMin, kRefineRationalCMax)
        : 1.0;

    PeriodicRationalResult best;
    best.omega = omega;
    best.c = c_val;
    best.a = best.b = best.d = best.e = 0.0;
    best.mse = std::numeric_limits<double>::infinity();
    best.success = false;
    
    for (int step = 0; step < steps; ++step) {
        Eigen::MatrixXd X(n, 4); // sin/(x^2+c), cos/(x^2+c), x, 1
        Eigen::VectorXd denom = x.array().square() + c_val;
        X.col(0) = (omega * x.array()).sin() / denom.array();
        X.col(1) = (omega * x.array()).cos() / denom.array();
        X.col(2) = x;
        X.col(3) = Eigen::VectorXd::Ones(n);
        
        Eigen::VectorXd coef = solve_linear_weighted(X, y, sample_weight);
        Eigen::VectorXd pred = X * coef;
        double mse = residual_mse_weighted(pred, y, sample_weight);
        
        if (mse < best.mse) {
            best.omega = omega;
            best.c = c_val;
            best.a = coef[0];
            best.b = coef[1];
            best.d = coef[2];
            best.e = coef[3];
            best.mse = mse;
            best.success = true;
        }
        
        double eps = 1e-4;
        
        // gradient wrt omega
        double o_fwd, o_bwd;
        {
            Eigen::MatrixXd X_f = X;
            X_f.col(0) = ((omega+eps) * x.array()).sin() / denom.array();
            X_f.col(1) = ((omega+eps) * x.array()).cos() / denom.array();
            double m_f = residual_mse_weighted(X_f * solve_linear_weighted(X_f, y, sample_weight), y, sample_weight);
            
            Eigen::MatrixXd X_b = X;
            X_b.col(0) = ((omega-eps) * x.array()).sin() / denom.array();
            X_b.col(1) = ((omega-eps) * x.array()).cos() / denom.array();
            double m_b = residual_mse_weighted(X_b * solve_linear_weighted(X_b, y, sample_weight), y, sample_weight);
            
            o_fwd = m_f; o_bwd = m_b;
        }
        double grad_omega = (o_fwd - o_bwd) / (2 * eps);
        
        // gradient wrt c
        double c_fwd, c_bwd;
        {
            Eigen::MatrixXd X_f = X;
            Eigen::VectorXd d_f = x.array().square() + (c_val + eps);
            X_f.col(0) = (omega * x.array()).sin() / d_f.array();
            X_f.col(1) = (omega * x.array()).cos() / d_f.array();
            double m_f = residual_mse_weighted(X_f * solve_linear_weighted(X_f, y, sample_weight), y, sample_weight);
            
            Eigen::MatrixXd X_b = X;
            Eigen::VectorXd d_b = x.array().square() + (c_val - eps);
            X_b.col(0) = (omega * x.array()).sin() / d_b.array();
            X_b.col(1) = (omega * x.array()).cos() / d_b.array();
            double m_b = residual_mse_weighted(X_b * solve_linear_weighted(X_b, y, sample_weight), y, sample_weight);
            
            c_fwd = m_f; c_bwd = m_b;
        }
        double grad_c = (c_fwd - c_bwd) / (2 * eps);
        
        // Finite-gradient guard + projection (§3.98; M-339): skip NaN FD probes;
        // omega keeps sign freedom but is magnitude-capped and can never
        // persist as NaN; c_val stays in [1e-6, 1e6] (denominator safety).
        if (std::isfinite(grad_omega)) {
            double trial = omega - lr * grad_omega;
            if (std::isfinite(trial))
                omega = std::clamp(trial, -kRefineRationalOmegaMax,
                                   kRefineRationalOmegaMax);
        }
        if (std::isfinite(grad_c)) {
            double trial = c_val - lr * grad_c;
            if (std::isfinite(trial))
                c_val = std::clamp(trial, kRefineRationalCMin, kRefineRationalCMax);
        }
    }
    return best;
}

} // namespace sr