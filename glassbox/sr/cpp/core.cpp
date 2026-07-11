#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include "ast.h"
#include "eval.h"
#include "evolution.h"
#include "formula_parser.h"
#include "refine.h"
#include "simplify.h"
#include "simplify_advanced.h"

#include <omp.h>
#include <iostream>
#include <limits>
#include <cctype>
#include <string>

namespace py = pybind11;

Eigen::ArrayXd evaluate_parse_node_exact(
    const std::shared_ptr<sr::ParseNode>& node,
    const std::vector<Eigen::ArrayXd>& X,
    int num_samples
) {
    if (!node) return Eigen::ArrayXd::Zero(num_samples);
    switch (node->type) {
        case sr::ParseNodeType::Input:
            if (node->feature_idx >= 0 && node->feature_idx < static_cast<int>(X.size())) {
                return X[node->feature_idx];
            }
            throw std::runtime_error("feature_index_out_of_range");
        case sr::ParseNodeType::Constant:
            return Eigen::ArrayXd::Constant(num_samples, node->value);
        case sr::ParseNodeType::Add:
            return evaluate_parse_node_exact(node->left, X, num_samples) + evaluate_parse_node_exact(node->right, X, num_samples);
        case sr::ParseNodeType::Sub:
            return evaluate_parse_node_exact(node->left, X, num_samples) - evaluate_parse_node_exact(node->right, X, num_samples);
        case sr::ParseNodeType::Mul:
            return evaluate_parse_node_exact(node->left, X, num_samples) * evaluate_parse_node_exact(node->right, X, num_samples);
        case sr::ParseNodeType::Div: {
            Eigen::ArrayXd left = evaluate_parse_node_exact(node->left, X, num_samples);
            Eigen::ArrayXd right = evaluate_parse_node_exact(node->right, X, num_samples);
            return left / right;
        }
        case sr::ParseNodeType::Pow: {
            Eigen::ArrayXd base = evaluate_parse_node_exact(node->left, X, num_samples);
            Eigen::ArrayXd exp_arr = evaluate_parse_node_exact(node->right, X, num_samples);
            Eigen::ArrayXd out(num_samples);
            bool constant_exp = exp_arr.size() == num_samples && (exp_arr - exp_arr(0)).abs().maxCoeff() < 1e-12;
            if (constant_exp) {
                double p = exp_arr(0);
                double p_round = std::round(p);
                if (std::abs(p - p_round) < 1e-10) {
                    out = base.pow(static_cast<int>(p_round));
                } else {
                    if ((base < 0.0).any()) {
                        throw std::runtime_error("power_domain_error");
                    }
                    out = base.pow(p);
                }
            } else {
                for (int i = 0; i < num_samples; ++i) {
                    out(i) = std::pow(base(i), exp_arr(i));
                }
            }
            return out;
        }
        case sr::ParseNodeType::Sin:
            return evaluate_parse_node_exact(node->left, X, num_samples).sin();
        case sr::ParseNodeType::Cos:
            return evaluate_parse_node_exact(node->left, X, num_samples).cos();
        case sr::ParseNodeType::Exp:
            return evaluate_parse_node_exact(node->left, X, num_samples).min(500.0).max(-500.0).exp();
        case sr::ParseNodeType::Log:
            return (evaluate_parse_node_exact(node->left, X, num_samples).abs() + 1e-300).log();
        case sr::ParseNodeType::Abs:
            return evaluate_parse_node_exact(node->left, X, num_samples).abs();
        case sr::ParseNodeType::Sqrt:
            return evaluate_parse_node_exact(node->left, X, num_samples).max(0.0).sqrt();
    }
    return Eigen::ArrayXd::Zero(num_samples);
}

std::shared_ptr<sr::ParseNode> parse_formula_exact(const std::string& formula) {
    std::string norm = sr::normalize_formula_string(formula);
    auto tokens = sr::tokenize(norm);
    sr::Parser parser(tokens);
    return parser.parse();
}

// Weighted helpers for Phase 2 candidate scoring (PhySO-style y_weights).
static Eigen::ArrayXd load_optional_weights(
    const py::object& weights_obj,
    int n,
    const char* name
) {
    if (weights_obj.is_none()) {
        return Eigen::ArrayXd();  // empty => uniform
    }
    auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(weights_obj);
    if (!arr) {
        throw std::runtime_error(std::string(name) + " must be convertible to float64 array");
    }
    auto buf = arr.request();
    if (buf.ndim != 1 || static_cast<int>(buf.size) != n) {
        throw std::runtime_error(
            std::string(name) + " must be 1D with length matching the corresponding y split"
        );
    }
    Eigen::Map<Eigen::ArrayXd> mapped(static_cast<double*>(buf.ptr), n);
    Eigen::ArrayXd w = mapped;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(w(i)) || w(i) < 0.0) {
            throw std::runtime_error(std::string(name) + " must be finite and non-negative");
        }
    }
    double total = w.sum();
    if (!(total > 0.0) || !std::isfinite(total)) {
        throw std::runtime_error(std::string(name) + " must have positive total weight");
    }
    return w;
}

static double weighted_mean(const Eigen::ArrayXd& v, const Eigen::ArrayXd& w, double w_sum) {
    return (w * v).sum() / w_sum;
}

static double weighted_mse(const Eigen::ArrayXd& err, const Eigen::ArrayXd& w, double w_sum) {
    return (w * err.square()).sum() / w_sum;
}

static double unweighted_mse(const Eigen::ArrayXd& err) {
    return err.square().mean();
}

py::list score_formula_candidates_cpp(
    py::list formulas_py,
    py::array_t<double> X_fit_array,
    py::array_t<double> y_fit_array,
    py::array_t<double> X_val_array,
    py::array_t<double> y_val_array,
    int num_threads = -1,
    py::object fit_weights_obj = py::none(),
    py::object val_weights_obj = py::none()
) {
    auto X_fit_contig = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(X_fit_array);
    auto y_fit_contig = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(y_fit_array);
    auto X_val_contig = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(X_val_array);
    auto y_val_contig = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(y_val_array);
    if (!X_fit_contig || !y_fit_contig || !X_val_contig || !y_val_contig) {
        throw std::runtime_error("score_formula_candidates expects contiguous float64 arrays");
    }

    auto X_fit_buf = X_fit_contig.request();
    auto y_fit_buf = y_fit_contig.request();
    auto X_val_buf = X_val_contig.request();
    auto y_val_buf = y_val_contig.request();
    if (X_fit_buf.ndim != 2 || X_val_buf.ndim != 2) {
        throw std::runtime_error("X_fit and X_val must be 2D arrays");
    }
    int n_fit = static_cast<int>(X_fit_buf.shape[0]);
    int n_val = static_cast<int>(X_val_buf.shape[0]);
    int p_fit = static_cast<int>(X_fit_buf.shape[1]);
    int p_val = static_cast<int>(X_val_buf.shape[1]);
    if (p_fit != p_val || y_fit_buf.size != n_fit || y_val_buf.size != n_val) {
        throw std::runtime_error("Input shapes are inconsistent");
    }

    // Optional per-point weights (Phase 2). Empty arrays => uniform / legacy path.
    Eigen::ArrayXd w_fit = load_optional_weights(fit_weights_obj, n_fit, "fit_weights");
    Eigen::ArrayXd w_val = load_optional_weights(val_weights_obj, n_val, "val_weights");
    const bool use_fit_w = w_fit.size() == n_fit;
    const bool use_val_w = w_val.size() == n_val;
    const double w_fit_sum = use_fit_w ? w_fit.sum() : static_cast<double>(n_fit);
    const double w_val_sum = use_val_w ? w_val.sum() : static_cast<double>(n_val);

    const double* X_fit_ptr = static_cast<double*>(X_fit_buf.ptr);
    const double* X_val_ptr = static_cast<double*>(X_val_buf.ptr);
    Eigen::Map<Eigen::ArrayXd> y_fit(static_cast<double*>(y_fit_buf.ptr), n_fit);
    Eigen::Map<Eigen::ArrayXd> y_val(static_cast<double*>(y_val_buf.ptr), n_val);

    std::vector<Eigen::ArrayXd> X_fit_cols;
    std::vector<Eigen::ArrayXd> X_val_cols;
    X_fit_cols.reserve(p_fit);
    X_val_cols.reserve(p_fit);
    for (int j = 0; j < p_fit; ++j) {
        Eigen::ArrayXd col_fit(n_fit);
        Eigen::ArrayXd col_val(n_val);
        for (int i = 0; i < n_fit; ++i) col_fit(i) = X_fit_ptr[i * p_fit + j];
        for (int i = 0; i < n_val; ++i) col_val(i) = X_val_ptr[i * p_fit + j];
        X_fit_cols.push_back(std::move(col_fit));
        X_val_cols.push_back(std::move(col_val));
    }

    struct CandidateScore {
        bool ok = false;
        bool weighted = false;
        std::string formula;
        std::string error;
        // Primary selection metrics (weighted when weights provided, else unweighted).
        double fit_mse = std::numeric_limits<double>::infinity();
        double val_mse = std::numeric_limits<double>::infinity();
        double r2 = -std::numeric_limits<double>::infinity();
        // Always populate unweighted diagnostics.
        double unweighted_fit_mse = std::numeric_limits<double>::infinity();
        double unweighted_val_mse = std::numeric_limits<double>::infinity();
        double unweighted_r2 = -std::numeric_limits<double>::infinity();
        // Weighted diagnostics (NaN when no weights).
        double weighted_fit_mse = std::numeric_limits<double>::quiet_NaN();
        double weighted_val_mse = std::numeric_limits<double>::quiet_NaN();
        double weighted_r2 = std::numeric_limits<double>::quiet_NaN();
        double scale = 0.0;
        double bias = 0.0;
    };

    std::vector<std::string> formulas;
    formulas.reserve(py::len(formulas_py));
    for (auto item : formulas_py) {
        formulas.push_back(item.cast<std::string>());
    }
    std::vector<CandidateScore> scores(formulas.size());

    int previous_omp_threads = omp_get_max_threads();
    if (num_threads > 0) omp_set_num_threads(num_threads);

    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(dynamic)
        for (int idx = 0; idx < static_cast<int>(formulas.size()); ++idx) {
            CandidateScore score;
            score.formula = formulas[idx];
            score.weighted = use_fit_w || use_val_w;
            try {
                auto parsed = parse_formula_exact(score.formula);
                Eigen::ArrayXd pred_fit = evaluate_parse_node_exact(parsed, X_fit_cols, n_fit);
                Eigen::ArrayXd pred_val = evaluate_parse_node_exact(parsed, X_val_cols, n_val);
                if (pred_fit.size() != n_fit || pred_val.size() != n_val) {
                    score.error = "shape_mismatch";
                    scores[idx] = score;
                    continue;
                }
                bool finite = true;
                for (int i = 0; i < n_fit; ++i) {
                    if (!std::isfinite(pred_fit(i)) || !std::isfinite(y_fit(i))) {
                        finite = false;
                        break;
                    }
                }
                if (finite) {
                    for (int i = 0; i < n_val; ++i) {
                        if (!std::isfinite(pred_val(i)) || !std::isfinite(y_val(i))) {
                            finite = false;
                            break;
                        }
                    }
                }
                if (!finite || n_fit < 2 || n_val < 1) {
                    score.error = "nonfinite";
                    scores[idx] = score;
                    continue;
                }

                // Affine scale/bias on fit split (weighted least squares when weights given).
                double mean_x, mean_y, var_x, cov_xy;
                if (use_fit_w) {
                    mean_x = weighted_mean(pred_fit, w_fit, w_fit_sum);
                    mean_y = weighted_mean(y_fit, w_fit, w_fit_sum);
                    Eigen::ArrayXd dx = pred_fit - mean_x;
                    Eigen::ArrayXd dy = y_fit - mean_y;
                    var_x = (w_fit * dx.square()).sum();
                    cov_xy = (w_fit * dx * dy).sum();
                } else {
                    mean_x = pred_fit.mean();
                    mean_y = y_fit.mean();
                    var_x = (pred_fit - mean_x).square().sum();
                    cov_xy = ((pred_fit - mean_x) * (y_fit - mean_y)).sum();
                }
                if (var_x > 1e-15) {
                    score.scale = cov_xy / var_x;
                    score.bias = mean_y - score.scale * mean_x;
                } else {
                    score.scale = 0.0;
                    score.bias = mean_y;
                }

                Eigen::ArrayXd fit_err = score.scale * pred_fit + score.bias - y_fit;
                Eigen::ArrayXd val_err = score.scale * pred_val + score.bias - y_val;

                score.unweighted_fit_mse = unweighted_mse(fit_err);
                score.unweighted_val_mse = unweighted_mse(val_err);
                {
                    double mean_y_val_u = y_val.mean();
                    double var_y_val_u = (y_val - mean_y_val_u).square().mean();
                    if (var_y_val_u < 1e-15) {
                        score.unweighted_r2 = score.unweighted_val_mse < 1e-15 ? 1.0 : 0.0;
                    } else {
                        score.unweighted_r2 = 1.0 - score.unweighted_val_mse / var_y_val_u;
                    }
                }

                if (use_fit_w) {
                    score.weighted_fit_mse = weighted_mse(fit_err, w_fit, w_fit_sum);
                }
                if (use_val_w) {
                    score.weighted_val_mse = weighted_mse(val_err, w_val, w_val_sum);
                    double mean_y_val_w = weighted_mean(y_val, w_val, w_val_sum);
                    double var_y_val_w = (w_val * (y_val - mean_y_val_w).square()).sum() / w_val_sum;
                    if (var_y_val_w < 1e-15) {
                        score.weighted_r2 = score.weighted_val_mse < 1e-15 ? 1.0 : 0.0;
                    } else {
                        score.weighted_r2 = 1.0 - score.weighted_val_mse / var_y_val_w;
                    }
                } else if (use_fit_w) {
                    // Fit weighted but val unweighted: still expose weighted_fit only.
                    score.weighted_val_mse = score.unweighted_val_mse;
                    score.weighted_r2 = score.unweighted_r2;
                }

                // Primary selection metrics: prefer weighted when available.
                score.fit_mse = use_fit_w ? score.weighted_fit_mse : score.unweighted_fit_mse;
                score.val_mse = use_val_w ? score.weighted_val_mse : score.unweighted_val_mse;
                score.r2 = use_val_w ? score.weighted_r2 : score.unweighted_r2;

                score.ok = std::isfinite(score.fit_mse) && std::isfinite(score.val_mse) && std::isfinite(score.r2);
                if (!score.ok) score.error = "invalid_score";
            } catch (const std::exception& e) {
                score.error = e.what();
            } catch (...) {
                score.error = "unknown_error";
            }
            scores[idx] = score;
        }
    }

    if (num_threads > 0) omp_set_num_threads(previous_omp_threads);

    py::list out;
    for (const auto& score : scores) {
        py::dict item;
        item["formula"] = score.formula;
        item["ok"] = score.ok;
        // Primary (selection) metrics: weighted when weights provided.
        item["fit_mse"] = score.fit_mse;
        item["mse"] = score.val_mse;
        item["validation_mse"] = score.val_mse;
        item["r2"] = score.r2;
        item["validation_r2"] = score.r2;
        // Explicit dual metrics (Phase 2 contract).
        item["unweighted_fit_mse"] = score.unweighted_fit_mse;
        item["unweighted_validation_mse"] = score.unweighted_val_mse;
        item["unweighted_r2"] = score.unweighted_r2;
        item["weighted_fit_mse"] = score.weighted_fit_mse;
        item["weighted_validation_mse"] = score.weighted_val_mse;
        item["weighted_r2"] = score.weighted_r2;
        item["weighted"] = score.weighted;
        item["scale"] = score.scale;
        item["bias"] = score.bias;
        item["error"] = score.error;
        out.append(item);
    }
    return out;
}

// Pybind wrapper for the evolution engine
py::dict run_evolution_cpp(
    py::list X_list, // List of numpy arrays (features)
    py::array_t<double> y_array,
    int pop_size,
    int generations,
    double early_stop_mse,
    py::list seed_omegas = py::list(),
    int timeout_seconds = 120,
    py::list op_priors = py::list(),
    py::list allowed_unary_ops = py::list(),
    py::list binary_op_priors = py::list(),
    py::list allowed_binary_ops = py::list(),
    // Power exponent bounds
    double p_min = -2.0,
    double p_max = 3.0,
    // P5: NSGA-II
    bool use_nsga2 = false,
    // P6: Island Model
    int num_islands = 1,
    int migration_interval = 25,
    int migration_size = 2,
    // P7: Dimensional Analysis
    py::list input_units = py::list(),
    py::list output_units = py::list(),
    double arithmetic_temperature = 5.0,
    std::string trace_path = "",
    bool trace_include_formulas = false,
    bool use_staged_schedule = true,
    int topology_phase_generations = 40,
    double topology_phase_mutation_boost = 1.5,
    int topology_refine_interval = 20,
    bool use_adaptive_restart = true,
    int stagnation_window = 40,
    double stagnation_min_improvement = 1e-5,
    double diversity_floor = 0.25,
    double restart_fraction = 0.25,
    double post_restart_mutation_boost = 1.5,
    int random_seed = -1,
    double acceptable_mse = 1e-8,
    int acceptable_complexity = 15,
    int early_stop_max_nodes = 50,
    int num_threads = -1,
    // Diverse Islands Support
    py::list multi_op_priors = py::list(),
    py::list multi_allowed_unary_ops = py::list(),
    py::list multi_binary_op_priors = py::list(),
    py::list multi_allowed_binary_ops = py::list(),
    py::list multi_seed_omegas = py::list(),
    py::list seed_graphs_py = py::list(),
    py::object y_weights_obj = py::none(),
    std::string loss_mode = "mse",
    double huber_delta = -1.0,
    double trim_fraction = 0.1
) {
    // 1. Convert Python/Numpy inputs to C++/Eigen
    std::vector<Eigen::ArrayXd> X;
    for (auto item : X_list) {
        auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(item);
        if (!arr) {
            throw std::runtime_error("X_list entries must be convertible to contiguous float64 arrays");
        }
        auto buf = arr.request();
        double* ptr = static_cast<double*>(buf.ptr);
        X.emplace_back(Eigen::Map<Eigen::ArrayXd>(ptr, buf.size));
    }
    
    auto y_contig = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(y_array);
    if (!y_contig) {
        throw std::runtime_error("y must be convertible to a contiguous float64 array");
    }
    auto y_buf = y_contig.request();
    double* y_ptr = static_cast<double*>(y_buf.ptr);
    Eigen::Map<Eigen::ArrayXd> y(y_ptr, y_buf.size);

    // Optional per-point weights (Phase 3). Empty => uniform / legacy path.
    Eigen::ArrayXd y_weights = load_optional_weights(
        y_weights_obj, static_cast<int>(y_buf.size), "y_weights"
    );
    
    // Parse seed omegas
    std::vector<double> cpp_seed_omegas;
    for (auto item : seed_omegas) {
        cpp_seed_omegas.push_back(item.cast<double>());
    }

    // Parse op priors
    std::vector<double> cpp_op_priors;
    for (auto item : op_priors) {
        cpp_op_priors.push_back(item.cast<double>());
    }

    std::vector<double> cpp_binary_op_priors;
    for (auto item : binary_op_priors) {
        cpp_binary_op_priors.push_back(item.cast<double>());
    }

    std::vector<int> cpp_allowed_unary_ops;
    for (auto item : allowed_unary_ops) {
        cpp_allowed_unary_ops.push_back(item.cast<int>());
    }

    std::vector<int> cpp_allowed_binary_ops;
    for (auto item : allowed_binary_ops) {
        cpp_allowed_binary_ops.push_back(item.cast<int>());
    }

    // Parse multi_op_priors
    std::vector<std::vector<double>> cpp_multi_op_priors;
    for (auto item : multi_op_priors) {
        std::vector<double> prior_vec;
        for (auto val : item.cast<py::list>()) {
            prior_vec.push_back(val.cast<double>());
        }
        cpp_multi_op_priors.push_back(prior_vec);
    }

    std::vector<std::vector<double>> cpp_multi_binary_op_priors;
    for (auto item : multi_binary_op_priors) {
        std::vector<double> prior_vec;
        for (auto val : item.cast<py::list>()) {
            prior_vec.push_back(val.cast<double>());
        }
        cpp_multi_binary_op_priors.push_back(prior_vec);
    }

    std::vector<std::vector<int>> cpp_multi_allowed_unary_ops;
    for (auto item : multi_allowed_unary_ops) {
        std::vector<int> allowed_vec;
        for (auto val : item.cast<py::list>()) {
            allowed_vec.push_back(val.cast<int>());
        }
        cpp_multi_allowed_unary_ops.push_back(allowed_vec);
    }

    std::vector<std::vector<int>> cpp_multi_allowed_binary_ops;
    for (auto item : multi_allowed_binary_ops) {
        std::vector<int> allowed_vec;
        for (auto val : item.cast<py::list>()) {
            allowed_vec.push_back(val.cast<int>());
        }
        cpp_multi_allowed_binary_ops.push_back(allowed_vec);
    }

    // Parse multi_seed_omegas
    std::vector<std::vector<double>> cpp_multi_seed_omegas;
    for (auto item : multi_seed_omegas) {
        std::vector<double> seed_vec;
        for (auto val : item.cast<py::list>()) {
            seed_vec.push_back(val.cast<double>());
        }
        cpp_multi_seed_omegas.push_back(seed_vec);
    }

    // Parse seed_graphs
    std::vector<sr::IndividualGraph> cpp_seed_graphs;
    int seed_graphs_skipped_oversized = 0;
    int seed_graph_node_limit = std::max(24, std::min(64, early_stop_max_nodes));
    for (auto item : seed_graphs_py) {
        auto gdict = item.cast<py::dict>();
        sr::IndividualGraph g;
        
        if (gdict.contains("nodes")) {
            auto nodes_list = gdict["nodes"].cast<py::list>();
            for (auto n_item : nodes_list) {
                auto ndict = n_item.cast<py::dict>();
                sr::OpNode node;
                node.type = static_cast<sr::NodeType>(ndict["type"].cast<int>());
                if (ndict.contains("feature_idx")) node.feature_idx = ndict["feature_idx"].cast<int>();
                if (ndict.contains("value")) node.value = ndict["value"].cast<double>();
                if (ndict.contains("unary_op")) node.unary_op = static_cast<sr::UnaryOp>(ndict["unary_op"].cast<int>());
                if (ndict.contains("binary_op")) node.binary_op = static_cast<sr::BinaryOp>(ndict["binary_op"].cast<int>());
                if (ndict.contains("p")) node.p = ndict["p"].cast<double>();
                if (ndict.contains("omega")) node.omega = ndict["omega"].cast<double>();
                if (ndict.contains("phi")) node.phi = ndict["phi"].cast<double>();
                if (ndict.contains("amplitude")) node.amplitude = ndict["amplitude"].cast<double>();
                if (ndict.contains("beta")) node.beta = ndict["beta"].cast<double>();
                if (ndict.contains("gamma")) node.gamma = ndict["gamma"].cast<double>();
                if (ndict.contains("tau")) node.tau = ndict["tau"].cast<double>();
                if (ndict.contains("left_child")) node.left_child = ndict["left_child"].cast<int>();
                if (ndict.contains("right_child")) node.right_child = ndict["right_child"].cast<int>();
                g.nodes.push_back(node);
            }
        }
        
        if (gdict.contains("output_weights")) {
            auto weights_list = gdict["output_weights"].cast<py::list>();
            for (auto w : weights_list) {
                g.output_weights.push_back(w.cast<double>());
            }
        }
        
        if (gdict.contains("output_bias")) g.output_bias = gdict["output_bias"].cast<double>();
        
        if (static_cast<int>(g.nodes.size()) > seed_graph_node_limit) {
            ++seed_graphs_skipped_oversized;
            continue;
        }
        cpp_seed_graphs.push_back(g);
    }

    // Parse input_units (list of lists)
    std::vector<std::vector<double>> cpp_input_units;
    for (auto item : input_units) {
        std::vector<double> unit_vec;
        for (auto val : item.cast<py::list>()) {
            unit_vec.push_back(val.cast<double>());
        }
        cpp_input_units.push_back(unit_vec);
    }

    // Parse output_units
    std::vector<double> cpp_output_units;
    for (auto item : output_units) {
        cpp_output_units.push_back(item.cast<double>());
    }

    // 2. Configure engine
    sr::EvolutionConfig config;
    config.timeout_seconds = timeout_seconds;
    config.pop_size = pop_size;
    config.generations = generations;
    config.early_stop_mse = early_stop_mse;
    config.op_priors = cpp_op_priors;
    config.allowed_unary_ops = cpp_allowed_unary_ops;
    config.binary_op_priors = cpp_binary_op_priors;
    config.allowed_binary_ops = cpp_allowed_binary_ops;
    config.multi_op_priors = cpp_multi_op_priors;
    config.multi_allowed_unary_ops = cpp_multi_allowed_unary_ops;
    config.multi_binary_op_priors = cpp_multi_binary_op_priors;
    config.multi_allowed_binary_ops = cpp_multi_allowed_binary_ops;
    config.multi_seed_omegas = cpp_multi_seed_omegas;
    config.p_min = p_min;
    config.p_max = p_max;
    config.use_nsga2 = use_nsga2;
    config.num_islands = num_islands;
    config.migration_interval = migration_interval;
    config.migration_size = migration_size;
    config.input_units = cpp_input_units;
    config.output_units = cpp_output_units;
    config.enable_trace = !trace_path.empty();
    config.trace_path = trace_path;
    config.trace_include_formulas = trace_include_formulas;
    config.use_staged_schedule = use_staged_schedule;
    config.topology_phase_generations = topology_phase_generations;
    config.topology_phase_mutation_boost = topology_phase_mutation_boost;
    config.topology_refine_interval = topology_refine_interval;
    config.use_adaptive_restart = use_adaptive_restart;
    config.stagnation_window = stagnation_window;
    config.stagnation_min_improvement = stagnation_min_improvement;
    config.diversity_floor = diversity_floor;
    config.restart_fraction = restart_fraction;
    config.post_restart_mutation_boost = post_restart_mutation_boost;
    config.random_seed = random_seed;
    config.acceptable_mse = acceptable_mse;
    config.acceptable_complexity = acceptable_complexity;
    config.early_stop_max_nodes = early_stop_max_nodes;

    // Phase 4: robust search loss (default mse preserves legacy behaviour).
    {
        std::string mode = loss_mode;
        for (char& c : mode) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (mode == "huber") config.loss_mode = sr::LossMode::Huber;
        else if (mode == "trimmed_mse" || mode == "trimmed") config.loss_mode = sr::LossMode::TrimmedMse;
        else if (mode == "student_t" || mode == "student-t" || mode == "studentt") {
            config.loss_mode = sr::LossMode::StudentT;
        } else {
            config.loss_mode = sr::LossMode::Mse;
        }
        config.huber_delta = huber_delta;
        config.trim_fraction = trim_fraction;
    }

    // Sync evaluator temperature so arithmetic blend sharpness is tunable from Python.
    sr::set_arithmetic_temperature(arithmetic_temperature);

    int previous_omp_threads = omp_get_max_threads();
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    std::cout << "[v6-nsga2] Starting C++ Evolution with " << omp_get_max_threads() << " OpenMP Threads!";
    if (use_nsga2) std::cout << " (NSGA-II mode)";
    if (num_islands > 1) std::cout << " (Island Model: " << num_islands << " islands)";
    std::cout << std::endl;
    
    sr::EvolutionEngine engine(
        config, X, y, cpp_seed_omegas, cpp_seed_graphs, y_weights
    );
    
    // 3. Run evolution loop natively in C++
    {
        py::gil_scoped_release release;
        if (num_islands > 1) {
            engine.run_islands();
        } else {
            engine.run();
        }
    }

    // Restore thread count if modified
    if (num_threads > 0) {
        omp_set_num_threads(previous_omp_threads);
    }
    
    // 4. Return results as Python dict
    auto best = engine.get_best();
    
    py::dict result;
    // best_mse remains unweighted for back-compat / benchmarks.
    result["best_mse"] = best.raw_mse;
    result["best_weighted_mse"] = best.weighted_mse;
    result["weighted"] = (y_weights.size() == static_cast<int>(y_buf.size));
    result["loss_mode"] = loss_mode;
    result["search_loss"] = best.weighted_mse;
    result["penalized_fitness"] = best.fitness;
    result["time_to_first_exact_sec"] = engine.get_first_exact_time_sec();
    result["generation_to_first_exact"] = engine.get_first_exact_generation();
    result["time_to_first_acceptable_sec"] = engine.get_first_acceptable_time_sec();
    result["generation_to_first_acceptable"] = engine.get_first_acceptable_generation();
    result["evolution_wall_time_sec"] = engine.get_run_wall_time_sec();
    result["random_seed"] = engine.get_random_seed();
    result["openmp_threads"] = omp_get_max_threads();
    result["island_outer_threads"] = engine.get_last_island_outer_threads();
    result["island_inner_threads"] = engine.get_last_island_inner_threads();
    result["seed_graphs_used"] = static_cast<int>(cpp_seed_graphs.size());
    result["seed_graphs_skipped_oversized"] = seed_graphs_skipped_oversized;
    result["seed_graph_node_limit"] = seed_graph_node_limit;
    
    // Serialize graph structure
    py::list nodes_list;
    for (const auto& node : best.nodes) {
        py::dict ndict;
        ndict["type"] = static_cast<int>(node.type);
        ndict["feature_idx"] = node.feature_idx;
        ndict["value"] = node.value;
        ndict["unary_op"] = static_cast<int>(node.unary_op);
        ndict["binary_op"] = static_cast<int>(node.binary_op);
        ndict["p"] = node.p;
        ndict["omega"] = node.omega;
        ndict["phi"] = node.phi;
        ndict["amplitude"] = node.amplitude;
        ndict["beta"] = node.beta;
        ndict["gamma"] = node.gamma;
        ndict["tau"] = node.tau;
        ndict["left_child"] = node.left_child;
        ndict["right_child"] = node.right_child;
        nodes_list.append(ndict);
    }
    result["nodes"] = nodes_list;
    
    py::list weights_list;
    for (double w : best.output_weights) {
        weights_list.append(w);
    }
    result["output_weights"] = weights_list;
    result["output_bias"] = best.output_bias;
    
    // Simplify best graph before string export.
    // Note: this is structural/constant-fold simplification (not SymPy-level algebra).
    sr::simplify_ast(best);

    // Add the parsed formula string for Python compatibility
    result["formula"] = sr::get_formula_string(best, static_cast<int>(X.size()));

    // P5: Pareto front (if NSGA-II enabled)
    if (use_nsga2) {
        auto pareto = engine.get_pareto_front();
        py::list pareto_list;
        for (auto ind : pareto) {
            sr::simplify_ast(ind);
            py::dict pdict;
            pdict["mse"] = ind.raw_mse;
            pdict["weighted_mse"] = ind.weighted_mse;
            pdict["complexity"] = ind.active_complexity();
            pdict["raw_nodes"] = ind.complexity();
            pdict["formula"] = sr::get_formula_string(ind, static_cast<int>(X.size()));
            pdict["pareto_rank"] = ind.pareto_rank;
            pareto_list.append(pdict);
        }
        result["pareto_front"] = pareto_list;
    }
    
    return result;
}

// Wrapper for refine_frequencies_cpp
py::tuple refine_frequencies_wrapper(py::array_t<double> x_arr, py::array_t<double> y_arr, py::list initial_omegas, int steps = 100, double lr = 0.1) {
    auto x_buf = x_arr.request();
    Eigen::Map<Eigen::VectorXd> x(static_cast<double*>(x_buf.ptr), x_buf.size);
    auto y_buf = y_arr.request();
    Eigen::Map<Eigen::VectorXd> y(static_cast<double*>(y_buf.ptr), y_buf.size);

    std::vector<double> omegas;
    for (auto item : initial_omegas) omegas.push_back(item.cast<double>());

    auto res = sr::refine_frequencies_cpp(x, y, omegas, steps, lr);
    
    py::list omegas_out;
    for (double w : res.omegas) omegas_out.append(w);
    
    return py::make_tuple(omegas_out, res.mse);
}

// Wrapper for refine_powers_model_cpp
py::tuple refine_powers_model_wrapper(py::array_t<double> x_arr, py::array_t<double> y_arr, py::list initial_powers, py::list initial_omegas, int steps = 200, double lr = 0.05) {
    auto x_buf = x_arr.request();
    Eigen::Map<Eigen::VectorXd> x(static_cast<double*>(x_buf.ptr), x_buf.size);
    auto y_buf = y_arr.request();
    Eigen::Map<Eigen::VectorXd> y(static_cast<double*>(y_buf.ptr), y_buf.size);

    std::vector<double> powers, omegas;
    for (auto item : initial_powers) powers.push_back(item.cast<double>());
    for (auto item : initial_omegas) omegas.push_back(item.cast<double>());

    auto res = sr::refine_powers_model_cpp(x, y, powers, omegas, steps, lr);
    
    py::dict out;
    out["mse"] = res.mse;
    out["constant"] = res.constant;
    out["linear"] = res.linear;
    
    py::list p_out, c_out, w_out;
    for (double p : res.powers) p_out.append(p);
    for (double c : res.coeffs) c_out.append(c);
    for (double pc : res.periodic_coeffs) w_out.append(pc);
    
    out["powers"] = p_out;
    out["coeffs"] = c_out;
    out["periodic_coeffs"] = w_out;
    
    return py::make_tuple(out, res.mse);
}

// Wrapper for refine_periodic_rational_cpp
py::dict refine_periodic_rational_wrapper(py::array_t<double> x_arr, py::array_t<double> y_arr, double omega0, double c0, int steps = 200, double lr = 0.05) {
    auto x_buf = x_arr.request();
    Eigen::Map<Eigen::VectorXd> x(static_cast<double*>(x_buf.ptr), x_buf.size);
    auto y_buf = y_arr.request();
    Eigen::Map<Eigen::VectorXd> y(static_cast<double*>(y_buf.ptr), y_buf.size);

    auto res = sr::refine_periodic_rational_cpp(x, y, omega0, c0, steps, lr);
    
    py::dict out;
    out["omega"] = res.omega;
    out["c"] = res.c;
    out["a"] = res.a;
    out["b"] = res.b;
    out["d"] = res.d;
    out["e"] = res.e;
    out["mse"] = res.mse;
    
    return out;
}

// Wrapper for iterative_elastic_net
py::tuple iterative_elastic_net_wrapper(py::array_t<double> X_arr, py::array_t<double> y_arr, 
                                        double l1_weight, double l2_weight, 
                                        int n_starts=3, int n_iterations=3, 
                                        double prune_threshold=0.05, int max_iter=1000) {
    auto X_buf = X_arr.request();
    auto y_buf = y_arr.request();
    
    int n = X_buf.shape[0];
    int p = X_buf.shape[1];
    
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X(
        static_cast<double*>(X_buf.ptr), n, p);
    Eigen::Map<Eigen::VectorXd> y(static_cast<double*>(y_buf.ptr), y_buf.size);

    auto res = sr::iterative_elastic_net(X, y, l1_weight, l2_weight, n_starts, n_iterations, prune_threshold, max_iter);
    
    py::list w_out;
    for (int i = 0; i < res.weights.size(); ++i) {
        w_out.append(res.weights(i));
    }
    
    return py::make_tuple(w_out, res.mse);
}

// Wrapper for lasso_coordinate_descent
py::list lasso_coordinate_descent_wrapper(py::array_t<double> X_arr, py::array_t<double> y_arr, 
                                          double alpha, int max_iter=1000, double tol=1e-4) {
    auto X_buf = X_arr.request();
    auto y_buf = y_arr.request();
    
    int n = X_buf.shape[0];
    int p = X_buf.shape[1];
    
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X(
        static_cast<double*>(X_buf.ptr), n, p);
    Eigen::Map<Eigen::VectorXd> y(static_cast<double*>(y_buf.ptr), y_buf.size);

    auto res = sr::elastic_net_cd_cpp(X, y, alpha, 0.0, max_iter, tol);
    
    py::list w_out;
    for (int i = 0; i < res.weights.size(); ++i) {
        w_out.append(res.weights(i));
    }
    
    return w_out;
}

// Wrapper for simplify_formula_cpp
std::string simplify_formula_wrapper(
    std::string formula_str,
    double int_tol = 1e-5,
    double zero_tol = 1e-8,
    int max_passes = 6,
    bool use_nsimplify = true,
    bool use_identities = true,
    bool approximate_trig = false,
    double dominant_trig_ratio = 0.9,
    double small_term_ratio = 0.08,
    int n_features = 1
) {
    return sr::simplify_formula_cpp(
        formula_str, int_tol, zero_tol, max_passes, use_nsimplify, use_identities,
        approximate_trig, dominant_trig_ratio, small_term_ratio, n_features
    );
}

std::string simplify_formula_cpp_wrapper(std::string formula_str) {
    return sr::simplify_formula_cpp(formula_str);
}

// Wrapper for formula_to_seed_graph_cpp
py::dict formula_to_seed_graph_wrapper(std::string formula_str) {
    sr::IndividualGraph graph = sr::formula_to_graph(formula_str);

    py::dict result;
    py::list nodes_list;
    for (const auto& node : graph.nodes) {
        py::dict ndict;
        ndict["type"] = static_cast<int>(node.type);
        ndict["feature_idx"] = node.feature_idx;
        ndict["value"] = node.value;
        ndict["unary_op"] = static_cast<int>(node.unary_op);
        ndict["binary_op"] = static_cast<int>(node.binary_op);
        ndict["p"] = node.p;
        ndict["omega"] = node.omega;
        ndict["phi"] = node.phi;
        ndict["amplitude"] = node.amplitude;
        ndict["beta"] = node.beta;
        ndict["gamma"] = node.gamma;
        ndict["tau"] = node.tau;
        ndict["left_child"] = node.left_child;
        ndict["right_child"] = node.right_child;
        nodes_list.append(ndict);
    }

    py::list weights_list;
    for (double w : graph.output_weights) {
        weights_list.append(w);
    }

    result["nodes"] = nodes_list;
    result["output_weights"] = weights_list;
    result["output_bias"] = graph.output_bias;
    return result;
}

// Wrapper for snap_formula_floats_cpp
std::string snap_formula_floats_wrapper(std::string formula_str, int n_features = 1) {
    sr::IndividualGraph graph = sr::formula_to_graph(formula_str);
    sr::simplify_ast(graph);
    return sr::get_formula_string(graph, n_features);
}

// Wrapper for reduce_formula_noise_cpp
std::string reduce_formula_noise_wrapper(
    std::string formula_str,
    py::list X_list,
    py::array_t<double> y_array
) {
    std::vector<Eigen::ArrayXd> X;
    for (auto item : X_list) {
        auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(item);
        if (!arr) {
            throw std::runtime_error("X_list entries must be convertible to contiguous float64 arrays");
        }
        auto buf = arr.request();
        double* ptr = static_cast<double*>(buf.ptr);
        X.emplace_back(Eigen::Map<Eigen::ArrayXd>(ptr, buf.size));
    }
    
    auto y_contig = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(y_array);
    if (!y_contig) {
        throw std::runtime_error("y must be convertible to a contiguous float64 array");
    }
    auto y_buf = y_contig.request();
    double* y_ptr = static_cast<double*>(y_buf.ptr);
    Eigen::Map<Eigen::ArrayXd> y(y_ptr, y_buf.size);
    
    return sr::reduce_formula_noise_cpp(formula_str, X, y);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Fast C++ core for Glassbox Symbolic Regression";
    m.def("score_formula_candidates", &score_formula_candidates_cpp,
          "Parse and score formulas with affine scaling using OpenMP. "
          "Optional fit_weights/val_weights enable Phase 2 weighted affine fit "
          "and weighted validation metrics (primary mse/r2 become weighted).",
          py::arg("formulas"), py::arg("X_fit"), py::arg("y_fit"),
          py::arg("X_val"), py::arg("y_val"), py::arg("num_threads")=-1,
          py::arg("fit_weights")=py::none(), py::arg("val_weights")=py::none());
    m.def("run_evolution", &run_evolution_cpp, "Runs the evolutionary algorithm natively in C++",
          py::arg("X_list"), py::arg("y"), py::arg("pop_size")=50, py::arg("generations")=1000, 
          py::arg("early_stop_mse")=1e-6, py::arg("seed_omegas")=py::list(),
          py::arg("timeout_seconds")=120,
          py::arg("op_priors")=py::list(),
          py::arg("allowed_unary_ops")=py::list(),
          py::arg("binary_op_priors")=py::list(),
          py::arg("allowed_binary_ops")=py::list(),
          py::arg("p_min")=-2.0,
          py::arg("p_max")=3.0,
          py::arg("use_nsga2")=false,
          py::arg("num_islands")=1,
          py::arg("migration_interval")=25,
          py::arg("migration_size")=2,
          py::arg("input_units")=py::list(),
          py::arg("output_units")=py::list(),
          py::arg("arithmetic_temperature")=5.0,
          py::arg("trace_path")="",
          py::arg("trace_include_formulas")=false,
          py::arg("use_staged_schedule")=true,
          py::arg("topology_phase_generations")=40,
          py::arg("topology_phase_mutation_boost")=1.5,
          py::arg("topology_refine_interval")=20,
          py::arg("use_adaptive_restart")=true,
          py::arg("stagnation_window")=40,
          py::arg("stagnation_min_improvement")=1e-5,
          py::arg("diversity_floor")=0.25,
          py::arg("restart_fraction")=0.25,
          py::arg("post_restart_mutation_boost")=1.5,
          py::arg("random_seed")=-1,
          py::arg("acceptable_mse")=1e-8,
          py::arg("acceptable_complexity")=15,
          py::arg("early_stop_max_nodes")=50,
          py::arg("num_threads")=-1,
          py::arg("multi_op_priors")=py::list(),
          py::arg("multi_allowed_unary_ops")=py::list(),
          py::arg("multi_binary_op_priors")=py::list(),
          py::arg("multi_allowed_binary_ops")=py::list(),
          py::arg("multi_seed_omegas")=py::list(),
          py::arg("seed_graphs_py")=py::list(),
          py::arg("y_weights")=py::none(),
          py::arg("loss_mode")="mse",
          py::arg("huber_delta")=-1.0,
          py::arg("trim_fraction")=0.1);

    m.def("refine_frequencies", &refine_frequencies_wrapper, "Refines frequencies via Eigen varpro");
    m.def("refine_powers", &refine_powers_model_wrapper, "Refines powers via Eigen varpro");
    m.def("refine_periodic_rational", &refine_periodic_rational_wrapper, "Refines periodic rational params via Eigen varpro");
    m.def("iterative_elastic_net", &iterative_elastic_net_wrapper, "Iterative Elastic Net for regularized pruning");
    m.def("lasso_coordinate_descent", &lasso_coordinate_descent_wrapper, "LASSO regression using coordinate descent");
    m.def("simplify_formula", &simplify_formula_wrapper, "Simplifies a math formula string natively in C++",
          py::arg("formula_str"), py::arg("int_tol")=1e-5, py::arg("zero_tol")=1e-8, py::arg("max_passes")=6,
          py::arg("use_nsimplify")=true, py::arg("use_identities")=true, py::arg("approximate_trig")=false,
          py::arg("dominant_trig_ratio")=0.9, py::arg("small_term_ratio")=0.08, py::arg("n_features")=1);
    m.def("simplify_formula_cpp", &simplify_formula_cpp_wrapper, "Simplifies a math formula string natively in C++",
          py::arg("formula_str"));
    m.def("formula_to_seed_graph", &formula_to_seed_graph_wrapper, "Parse a formula into a seed graph dict",
          py::arg("formula_str"));
    m.def("formula_to_seed_graph_cpp", &formula_to_seed_graph_wrapper, "Alias for formula_to_seed_graph",
          py::arg("formula_str"));
    m.def("snap_formula_floats", &snap_formula_floats_wrapper, "Snaps display floats in a formula string",
          py::arg("formula_str"), py::arg("n_features")=1);
    m.def("snap_formula_floats_cpp", &snap_formula_floats_wrapper, "Alias for snap_formula_floats");
    m.def("reduce_formula_noise", &reduce_formula_noise_wrapper, "Greedy backward elimination of terms to reduce noise",
          py::arg("formula_str"), py::arg("X_list"), py::arg("y"));
    m.def("reduce_formula_noise_cpp", &reduce_formula_noise_wrapper, "Alias for reduce_formula_noise");
}

