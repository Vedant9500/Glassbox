#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include "ast.h"
#include "eval.h"
#include "evolution.h"

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
    py::list op_priors = py::list(),
    // P5: NSGA-II
    bool use_nsga2 = false,
    // P6: Island Model
    int num_islands = 1,
    int migration_interval = 25,
    int migration_size = 2,
    // P7: Dimensional Analysis
    py::list input_units = py::list(),
    py::list output_units = py::list()
) {
    // 1. Convert Python/Numpy to C++ Eigen
    std::vector<Eigen::ArrayXd> X;
    for (auto item : X_list) {
        auto arr = item.cast<py::array_t<double>>();
        auto buf = arr.request();
        double* ptr = static_cast<double*>(buf.ptr);
        X.emplace_back(Eigen::Map<Eigen::ArrayXd>(ptr, buf.size));
    }
    
    auto y_buf = y_array.request();
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
    config.pop_size = pop_size;
    config.generations = generations;
    config.early_stop_mse = early_stop_mse;
    config.op_priors = cpp_op_priors;
    config.use_nsga2 = use_nsga2;
    config.num_islands = num_islands;
    config.migration_interval = migration_interval;
    config.migration_size = migration_size;
    config.input_units = cpp_input_units;
    config.output_units = cpp_output_units;
    
    std::cout << "[v6-nsga2] Starting C++ Evolution with " << omp_get_max_threads() << " OpenMP Threads!";
    if (use_nsga2) std::cout << " (NSGA-II mode)";
    if (num_islands > 1) std::cout << " (Island Model: " << num_islands << " islands)";
    std::cout << std::endl;
    
    sr::EvolutionEngine engine(config, X, y, cpp_seed_omegas);
    
    // 3. Run evolution loop natively in C++
    if (num_islands > 1) {
        engine.run_islands();
    } else {
        engine.run();
    }
    
    // 4. Return results as Python dict
    auto best = engine.get_best();
    
    py::dict result;
    result["best_mse"] = best.raw_mse;
    result["penalized_fitness"] = best.fitness;
    
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
    
    // Add the parsed formula string for Python compatibility
    result["formula"] = sr::get_formula_string(best, static_cast<int>(X.size()));

    // P5: Pareto front (if NSGA-II enabled)
    if (use_nsga2) {
        auto pareto = engine.get_pareto_front();
        py::list pareto_list;
        for (const auto& ind : pareto) {
            py::dict pdict;
            pdict["mse"] = ind.raw_mse;
            pdict["complexity"] = ind.complexity();
            pdict["formula"] = sr::get_formula_string(ind, static_cast<int>(X.size()));
            pdict["pareto_rank"] = ind.pareto_rank;
            pareto_list.append(pdict);
        }
        result["pareto_front"] = pareto_list;
    }
    
    return result;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Fast C++ core for Glassbox Symbolic Regression";
    m.def("run_evolution", &run_evolution_cpp, "Runs the evolutionary algorithm natively in C++",
          py::arg("X_list"), py::arg("y"), py::arg("pop_size")=50, py::arg("generations")=1000, 
          py::arg("early_stop_mse")=1e-6, py::arg("seed_omegas")=py::list(),
          py::arg("op_priors")=py::list(),
          py::arg("use_nsga2")=false,
          py::arg("num_islands")=1,
          py::arg("migration_interval")=25,
          py::arg("migration_size")=2,
          py::arg("input_units")=py::list(),
          py::arg("output_units")=py::list());
}
