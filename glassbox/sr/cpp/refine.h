#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>

namespace sr {

struct ElasticNetResult {
    Eigen::VectorXd weights;
    double mse;
};

inline ElasticNetResult elastic_net_cd_cpp(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                                           double l1_weight, double l2_weight, 
                                           int max_iter=1000, double tol=1e-7,
                                           const Eigen::VectorXd& initial_w = Eigen::VectorXd()) {
    int n = X.rows();
    int p = X.cols();
    
    Eigen::VectorXd w = initial_w;
    if (w.size() != p) w = Eigen::VectorXd::Zero(p);
    if (n == 0 || p == 0) return {w, 0.0};
    
    // Precompute z_j = 1/n ||X_j||^2
    Eigen::VectorXd z(p);
    for (int j = 0; j < p; ++j) {
        z(j) = X.col(j).squaredNorm() / n;
    }
    
    Eigen::VectorXd res = y - X * w;
    double l1_half = l1_weight / 2.0;
    
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
        
        if (max_change < tol) break;
    }
    
    double mse = res.squaredNorm() / n;
    return {w, mse};
}

inline ElasticNetResult multi_start_elastic_net(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                                                double l1_weight, double l2_weight, 
                                                int n_starts=5, double init_scale=0.1,
                                                int max_iter=1000) {
    int p = X.cols();
    ElasticNetResult best_res;
    best_res.mse = 1e15;
    best_res.weights = Eigen::VectorXd::Zero(p);
    
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, init_scale);
    
    for (int i = 0; i < n_starts; ++i) {
        Eigen::VectorXd init_w = Eigen::VectorXd::Zero(p);
        if (i > 0) {
            for(int j=0; j<p; ++j) init_w(j) = dist(gen);
        }
        
        auto res = elastic_net_cd_cpp(X, y, l1_weight, l2_weight, max_iter, 1e-7, init_w);
        if (res.mse < best_res.mse) {
            best_res = res;
        }
    }
    return best_res;
}

inline ElasticNetResult iterative_elastic_net(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                                              double l1_weight, double l2_weight, 
                                              int n_starts=3, int n_iterations=3, 
                                              double prune_threshold=0.05,
                                              int max_iter=1000) {
    int p = X.cols();
    std::vector<bool> active_mask(p, true);
    
    ElasticNetResult best_res;
    best_res.mse = 1e15;
    best_res.weights = Eigen::VectorXd::Zero(p);
    
    for (int it = 0; it < n_iterations; ++it) {
        int num_active = 0;
        for (bool a : active_mask) if (a) num_active++;
        if (num_active == 0) break;
        
        Eigen::MatrixXd X_active(X.rows(), num_active);
        std::vector<int> active_indices;
        int col_idx = 0;
        for (int j = 0; j < p; ++j) {
            if (active_mask[j]) {
                X_active.col(col_idx++) = X.col(j);
                active_indices.push_back(j);
            }
        }
        
        auto res = multi_start_elastic_net(X_active, y, l1_weight, l2_weight, n_starts, 0.1, max_iter);
        
        Eigen::VectorXd full_weights = Eigen::VectorXd::Zero(p);
        for (int j = 0; j < num_active; ++j) {
            full_weights(active_indices[j]) = res.weights(j);
        }
        
        if (res.mse < best_res.mse) {
            best_res.mse = res.mse;
            best_res.weights = full_weights;
        }
        
        double max_w = full_weights.cwiseAbs().maxCoeff();
        if (max_w <= 0) break;
        
        double thresh = prune_threshold * max_w;
        std::vector<bool> new_mask(p, false);
        bool any_active = false;
        bool changed = false;
        
        for (int j = 0; j < p; ++j) {
            if (std::abs(full_weights(j)) > thresh) {
                new_mask[j] = true;
                any_active = true;
            }
            if (new_mask[j] != active_mask[j]) changed = true;
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

// Frequency refinement: c0 + c1*x + c2*x^2 + sum_i [a_i*sin(omega_i*x) + b_i*cos(omega_i*x)]
// We want to optimize omegas.
struct FreqResult {
    std::vector<double> omegas;
    double mse;
};

inline FreqResult refine_frequencies_cpp(const Eigen::VectorXd& x, const Eigen::VectorXd& y, 
                                         std::vector<double> initial_omegas, int steps = 100, double lr = 0.1) {
    int n = x.size();
    int k = initial_omegas.size();
    std::vector<double> omegas = initial_omegas;
    
    double best_mse = 1e9;
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
        
        // Solve for linear parameters
        Eigen::VectorXd coeffs = solve_linear(X, y);
        Eigen::VectorXd pred = X * coeffs;
        Eigen::VectorXd res = pred - y;
        double mse = res.squaredNorm() / n;
        
        if (mse < best_mse) {
            best_mse = mse;
            best_omegas = omegas;
        }
        
        // Gradient descent on omegas via finite differences
        double eps = 1e-4;
        std::vector<double> grads(k, 0.0);
        
        for (int i = 0; i < k; ++i) {
            // Forward step
            omegas[i] += eps;
            X.col(3 + 2*i) = (omegas[i] * x.array()).sin().matrix();
            X.col(4 + 2*i) = (omegas[i] * x.array()).cos().matrix();
            Eigen::VectorXd c_fwd = solve_linear(X, y);
            double mse_fwd = (X * c_fwd - y).squaredNorm() / n;
            
            // Backward step
            omegas[i] -= 2*eps;
            X.col(3 + 2*i) = (omegas[i] * x.array()).sin().matrix();
            X.col(4 + 2*i) = (omegas[i] * x.array()).cos().matrix();
            Eigen::VectorXd c_bwd = solve_linear(X, y);
            double mse_bwd = (X * c_bwd - y).squaredNorm() / n;
            
            // Restore
            omegas[i] += eps;
            X.col(3 + 2*i) = (omegas[i] * x.array()).sin().matrix();
            X.col(4 + 2*i) = (omegas[i] * x.array()).cos().matrix();
            
            grads[i] = (mse_fwd - mse_bwd) / (2 * eps);
        }
        
        // Update omegas
        for (int i = 0; i < k; ++i) {
            omegas[i] -= lr * grads[i];
            if (omegas[i] < 0.01) omegas[i] = 0.01; // constrain positive
        }
    }
    
    return {best_omegas, best_mse};
}

struct PowerResult {
    std::vector<double> powers;
    std::vector<double> coeffs;
    double constant;
    double linear;
    std::vector<double> periodic_coeffs; // sin_1, cos_1, ...
    double mse;
};

// sign(x) * |x|^p (parity preserving)
inline Eigen::VectorXd safe_power(const Eigen::VectorXd& x, double p) {
    Eigen::VectorXd abs_pow = (x.array().abs() + 1e-10).pow(p);
    double p_round = std::round(p);
    bool is_even = (std::abs(p - p_round) < 1e-6) && (static_cast<long long>(p_round) % 2 == 0);
    if (is_even) {
        return abs_pow;
    } else {
        return (x.array().sign() * abs_pow.array()).matrix();
    }
}

inline PowerResult refine_powers_model_cpp(const Eigen::VectorXd& x_valid, const Eigen::VectorXd& y_valid,
                                      std::vector<double> powers, const std::vector<double>& omegas,
                                      int steps = 200, double lr = 0.05) {
    int n = x_valid.size();
    int num_p = powers.size();
    int num_w = omegas.size();
    int num_features = 2 + num_p + 2 * num_w; // 1, x, p_i, sin(w_i), cos(w_i)
    
    double best_mse = 1e9;
    std::vector<double> best_powers = powers;
    Eigen::VectorXd best_coeffs;
    
    for (int step = 0; step < steps; ++step) {
        Eigen::MatrixXd X(n, num_features);
        X.col(0) = Eigen::VectorXd::Ones(n);
        X.col(1) = x_valid;
        for (int i = 0; i < num_p; ++i) {
            X.col(2 + i) = safe_power(x_valid, powers[i]);
        }
        for (int i = 0; i < num_w; ++i) {
            X.col(2 + num_p + 2*i) = (omegas[i] * x_valid.array()).sin().matrix();
            X.col(2 + num_p + 2*i + 1) = (omegas[i] * x_valid.array()).cos().matrix();
        }
        
        Eigen::VectorXd c = solve_linear(X, y_valid);
        Eigen::VectorXd pred = X * c;
        double mse = (pred - y_valid).squaredNorm() / n;
        
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
            Eigen::VectorXd c_fwd = solve_linear(X, y_valid);
            double mse_fwd = (X * c_fwd - y_valid).squaredNorm() / n;
            
            powers[i] -= 2*eps;
            X.col(2 + i) = safe_power(x_valid, powers[i]);
            Eigen::VectorXd c_bwd = solve_linear(X, y_valid);
            double mse_bwd = (X * c_bwd - y_valid).squaredNorm() / n;
            
            powers[i] += eps;
            X.col(2 + i) = safe_power(x_valid, powers[i]);
            
            grads[i] = (mse_fwd - mse_bwd) / (2 * eps);
        }
        
        for (int i = 0; i < num_p; ++i) {
            powers[i] -= lr * grads[i];
            if (powers[i] < -2.0) powers[i] = -2.0;
            if (powers[i] > 5.0) powers[i] = 5.0;
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
};

inline PeriodicRationalResult refine_periodic_rational_cpp(const Eigen::VectorXd& x, const Eigen::VectorXd& y,
                                                           double omega0, double c0, int steps = 200, double lr = 0.05) {
    int n = x.size();
    double omega = omega0;
    double c_val = std::max(c0, 1e-6); // enforce positivity
    
    double best_mse = 1e9;
    PeriodicRationalResult best;
    best.mse = best_mse;
    
    for (int step = 0; step < steps; ++step) {
        Eigen::MatrixXd X(n, 4); // sin/(x^2+c), cos/(x^2+c), x, 1
        Eigen::VectorXd denom = x.array().square() + c_val;
        X.col(0) = (omega * x.array()).sin() / denom.array();
        X.col(1) = (omega * x.array()).cos() / denom.array();
        X.col(2) = x;
        X.col(3) = Eigen::VectorXd::Ones(n);
        
        Eigen::VectorXd coef = solve_linear(X, y);
        Eigen::VectorXd pred = X * coef;
        double mse = (pred - y).squaredNorm() / n;
        
        if (mse < best_mse) {
            best_mse = mse;
            best.omega = omega;
            best.c = c_val;
            best.a = coef[0];
            best.b = coef[1];
            best.d = coef[2];
            best.e = coef[3];
            best.mse = mse;
        }
        
        double eps = 1e-4;
        
        // gradient wrt omega
        double o_fwd, o_bwd;
        {
            Eigen::MatrixXd X_f = X;
            X_f.col(0) = ((omega+eps) * x.array()).sin() / denom.array();
            X_f.col(1) = ((omega+eps) * x.array()).cos() / denom.array();
            double m_f = (X_f * solve_linear(X_f, y) - y).squaredNorm() / n;
            
            Eigen::MatrixXd X_b = X;
            X_b.col(0) = ((omega-eps) * x.array()).sin() / denom.array();
            X_b.col(1) = ((omega-eps) * x.array()).cos() / denom.array();
            double m_b = (X_b * solve_linear(X_b, y) - y).squaredNorm() / n;
            
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
            double m_f = (X_f * solve_linear(X_f, y) - y).squaredNorm() / n;
            
            Eigen::MatrixXd X_b = X;
            Eigen::VectorXd d_b = x.array().square() + (c_val - eps);
            X_b.col(0) = (omega * x.array()).sin() / d_b.array();
            X_b.col(1) = (omega * x.array()).cos() / d_b.array();
            double m_b = (X_b * solve_linear(X_b, y) - y).squaredNorm() / n;
            
            c_fwd = m_f; c_bwd = m_b;
        }
        double grad_c = (c_fwd - c_bwd) / (2 * eps);
        
        omega -= lr * grad_omega;
        c_val -= lr * grad_c;
        if (c_val < 1e-6) c_val = 1e-6;
    }
    return best;
}

} // namespace sr