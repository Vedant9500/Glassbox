#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include "ast.h"
#include "eval.h"
#include "evolution.h"

namespace py = pybind11;

// Pybind wrapper for the evolution engine
py::dict run_evolution_cpp(
    py::list X_list, // List of numpy arrays (features)
    py::array_t<double> y_array,
    int pop_size,
    int generations,
    double early_stop_mse
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
    
    // 2. Configure engine
    sr::EvolutionConfig config;
    config.pop_size = pop_size;
    config.generations = generations;
    config.early_stop_mse = early_stop_mse;
    
    sr::EvolutionEngine engine(config, X, y);
    
    // 3. Run evolution loop natively in C++
    engine.run();
    
    // 4. Return results as Python dict
    auto best = engine.get_best();
    
    py::dict result;
    result["best_mse"] = best.fitness;
    
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
    
    return result;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Fast C++ core for Glassbox Symbolic Regression";
    m.def("run_evolution", &run_evolution_cpp, "Runs the evolutionary algorithm natively in C++",
          py::arg("X_list"), py::arg("y"), py::arg("pop_size")=50, py::arg("generations")=1000, 
          py::arg("early_stop_mse")=1e-6);
}
