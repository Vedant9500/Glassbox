#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include "ast.h"
#include "eval.h"
#include "evolution.h"
#include "refine.h"
#include "simplify.h"
#include "simplify_advanced.h"

#include <omp.h>
#include <iostream>

namespace py = pybind11;

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
    py::list seed_graphs_py = py::list()
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
    
    sr::EvolutionEngine engine(config, X, y, cpp_seed_omegas, cpp_seed_graphs);
    
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
    result["best_mse"] = best.raw_mse;
    result["penalized_fitness"] = best.fitness;
    result["time_to_first_exact_sec"] = engine.get_first_exact_time_sec();
    result["generation_to_first_exact"] = engine.get_first_exact_generation();
    result["time_to_first_acceptable_sec"] = engine.get_first_acceptable_time_sec();
    result["generation_to_first_acceptable"] = engine.get_first_acceptable_generation();
    result["evolution_wall_time_sec"] = engine.get_run_wall_time_sec();
    result["random_seed"] = engine.get_random_seed();
    result["openmp_threads"] = omp_get_max_threads();
    
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
          py::arg("seed_graphs_py")=py::list());

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

