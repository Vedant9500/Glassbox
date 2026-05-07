import re

with open('glassbox/sr/cpp/eval.h', 'r') as f:
    content = f.read()

start_idx = content.find('inline Eigen::ArrayXd evaluate_graph(')
end_idx = content.find('// Compute MSE fitness')

replacement = """enum class EvalPolicy { Simple, CacheOut, SharedCache, Partial };

template<EvalPolicy Policy>
inline Eigen::ArrayXd evaluate_graph_impl(
    const IndividualGraph& graph,
    const std::vector<Eigen::ArrayXd>& X,
    int num_samples,
    std::vector<Eigen::ArrayXd>* cache_out = nullptr,
    SubtreeCache* shared_cache = nullptr,
    int perturbed_node_idx = -1,
    const std::vector<Eigen::ArrayXd>* old_cache = nullptr,
    std::vector<int>* changed_indices_out = nullptr) {
    
    if (graph.nodes.empty()) {
        if constexpr (Policy == EvalPolicy::CacheOut || Policy == EvalPolicy::SharedCache) {
            if (cache_out) cache_out->clear();
        }
        if constexpr (Policy == EvalPolicy::Partial) {
            if (cache_out && old_cache) *cache_out = *old_cache;
            if (changed_indices_out) changed_indices_out->clear();
        }
        return Eigen::ArrayXd::Zero(num_samples);
    }
    
    // Setup caches and tracking
    std::vector<bool> changed;
    if constexpr (Policy == EvalPolicy::Partial) {
        if (cache_out && old_cache) *cache_out = *old_cache;
        if (changed_indices_out) changed_indices_out->clear();
        changed.resize(graph.nodes.size(), false);
        changed[perturbed_node_idx] = true;
        changed_indices_out->push_back(perturbed_node_idx);
    } else if constexpr (Policy != EvalPolicy::Simple) {
        if (cache_out) cache_out->resize(graph.nodes.size());
    }

    std::vector<uint64_t> node_hashes;
    if constexpr (Policy == EvalPolicy::SharedCache) {
        node_hashes.resize(graph.nodes.size(), 0);
    }

    thread_local Eigen::ArrayXXd arena;
    if constexpr (Policy == EvalPolicy::Simple) {
        if (arena.rows() != num_samples || arena.cols() < static_cast<int>(graph.nodes.size())) {
            arena.resize(num_samples, std::max(static_cast<int>(graph.nodes.size()), 64));
        }
    }

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        bool needs_eval = true;

        if constexpr (Policy == EvalPolicy::Partial) {
            needs_eval = false;
            if (i == static_cast<size_t>(perturbed_node_idx)) {
                needs_eval = true;
            } else {
                const auto& n = graph.nodes[i];
                if (n.type == NodeType::Unary && changed[n.left_child]) {
                    needs_eval = true;
                    changed[i] = true;
                    changed_indices_out->push_back(static_cast<int>(i));
                } else if (n.type == NodeType::Binary && (changed[n.left_child] || changed[n.right_child])) {
                    needs_eval = true;
                    changed[i] = true;
                    changed_indices_out->push_back(static_cast<int>(i));
                }
            }
            if (!needs_eval) continue;
        }

        if constexpr (Policy == EvalPolicy::SharedCache) {
            node_hashes[i] = compute_node_hash(graph, static_cast<int>(i), node_hashes);
            auto it = shared_cache->find(node_hashes[i]);
            if (it != shared_cache->end() && it->second.size() == num_samples) {
                (*cache_out)[i] = it->second;
                continue;
            }
        }
        
        const auto& node = graph.nodes[i];
        Eigen::ArrayXd val;
        if constexpr (Policy == EvalPolicy::Simple) {
            // Allocate a proxy/block reference... NO, let's just use Eigen::ArrayXd wrapper to standardize
            // For simple we assign directly later.
        } else {
            val = Eigen::ArrayXd::Zero(num_samples);
        }
        
        auto get_child = [&](int idx) -> Eigen::ArrayXd {
            if constexpr (Policy == EvalPolicy::Simple) return arena.col(idx);
            else return (*cache_out)[idx];
        };

        switch (node.type) {
            case NodeType::Input: {
                if (node.feature_idx >= 0 && node.feature_idx < static_cast<int>(X.size())) {
                    val = X[node.feature_idx];
                } else {
                    val = Eigen::ArrayXd::Zero(num_samples);
                }
                break;
            }
            case NodeType::Constant: {
                val = Eigen::ArrayXd::Constant(num_samples, node.value);
                break;
            }
            case NodeType::Unary: {
                auto x = get_child(node.left_child);
                switch (node.unary_op) {
                    case UnaryOp::Periodic:
                        val = node.amplitude * (node.omega * x + node.phi).sin();
                        break;
                    case UnaryOp::Power: {
                        auto abs_x = x.abs() + 1e-10;
                        auto sign_x = x.sign();
                        auto abs_pow = abs_x.pow(node.p);
                        double is_even = power_sign_blend(node.p);
                        val = (1.0 - is_even) * (sign_x * abs_pow) + is_even * abs_pow;
                        val = val.max(-1e8).min(1e8);
                        break;
                    }
                    case UnaryOp::IntPow: {
                        int n = static_cast<int>(std::round(node.p));
                        n = std::clamp(n, 2, 6);
                        val = x.pow(n).max(-1e8).min(1e8);
                        break;
                    }
                    case UnaryOp::Exp: {
                        val = (node.omega * x + node.phi).exp().max(-1e6).min(1e6);
                        break;
                    }
                    case UnaryOp::Log: {
                        val = (x.abs() + 1e-6).log().max(-1e6).min(1e6);
                        break;
                    }
                }
                break;
            }
            case NodeType::Binary: {
                auto x = get_child(node.left_child);
                auto y = get_child(node.right_child);
                switch (node.binary_op) {
                    case BinaryOp::Arithmetic: {
                        auto w = arithmetic_soft_weights(node);
                        auto res_add = x + y;
                        auto res_sub = x - y;
                        auto res_mul = x * y;
                        
                        // Division changed to y.square()+1e-12 in partial but y.abs()+... elsewhere. Let's stick to base implementation.
                        auto res_div = x / (1.0 + y.square()).sqrt();
                        
                        val = (w[0] * res_add + w[1] * res_mul + w[2] * res_div + w[3] * res_sub).max(-1e6).min(1e6);
                        break;
                    }
                    case BinaryOp::Division: {
                        val = (x / (y.abs() + 1e-6) * y.sign()).max(-1e6).min(1e6);
                        break;
                    }
                    case BinaryOp::Aggregation: {
                        double local_tau = stabilized_tau(node.tau);
                        auto max_val = x.max(y);
                        auto exp_x = ((x - max_val) / local_tau).exp();
                        auto exp_y = ((y - max_val) / local_tau).exp();
                        auto sum_exp = exp_x + exp_y;
                        val = (x * exp_x / sum_exp) + (y * exp_y / sum_exp);
                        break;
                    }
                }
                break;
            }
        }
        
        if constexpr (Policy == EvalPolicy::Simple) {
            arena.col(i) = val;
        } else {
            (*cache_out)[i] = std::move(val);
        }
        
        if constexpr (Policy == EvalPolicy::SharedCache) {
            if (node.type == NodeType::Unary || node.type == NodeType::Binary) {
                (*shared_cache)[node_hashes[i]] = (*cache_out)[i];
            }
        }
    }
    
    if constexpr (Policy == EvalPolicy::Partial) {
        return Eigen::ArrayXd::Zero(num_samples);
    }
    
    Eigen::ArrayXd final_output = Eigen::ArrayXd::Constant(num_samples, graph.output_bias);
    for (size_t i = 0; i < graph.output_weights.size() && i < graph.nodes.size(); ++i) {
        if (std::abs(graph.output_weights[i]) > 1e-6) {
            if constexpr (Policy == EvalPolicy::Simple) {
                final_output += graph.output_weights[i] * arena.col(i);
            } else {
                final_output += graph.output_weights[i] * (*cache_out)[i];
            }
        }
    }
    return final_output;
}

inline Eigen::ArrayXd evaluate_graph(const IndividualGraph& graph, const std::vector<Eigen::ArrayXd>& X, int num_samples) {
    return evaluate_graph_impl<EvalPolicy::Simple>(graph, X, num_samples);
}

inline Eigen::ArrayXd evaluate_graph(const IndividualGraph& graph, const std::vector<Eigen::ArrayXd>& X, int num_samples, std::vector<Eigen::ArrayXd>& cache_out) {
    return evaluate_graph_impl<EvalPolicy::CacheOut>(graph, X, num_samples, &cache_out);
}

inline void evaluate_graph_partial(const IndividualGraph& graph, 
                                   int perturbed_node_idx,
                                   const std::vector<Eigen::ArrayXd>& old_cache,
                                   std::vector<Eigen::ArrayXd>& new_cache_out,
                                   std::vector<int>& changed_indices_out) {
    evaluate_graph_impl<EvalPolicy::Partial>(graph, std::vector<Eigen::ArrayXd>(), 0, &new_cache_out, nullptr, perturbed_node_idx, &old_cache, &changed_indices_out);
}

inline Eigen::ArrayXd evaluate_graph_simple(const IndividualGraph& graph, const std::vector<Eigen::ArrayXd>& X, int num_samples) {
    return evaluate_graph_impl<EvalPolicy::Simple>(graph, X, num_samples);
}

inline Eigen::ArrayXd evaluate_graph_cached(const IndividualGraph& graph,
                                             const std::vector<Eigen::ArrayXd>& X,
                                             int num_samples,
                                             std::vector<Eigen::ArrayXd>& cache_out,
                                             SubtreeCache& shared_cache) {
    return evaluate_graph_impl<EvalPolicy::SharedCache>(graph, X, num_samples, &cache_out, &shared_cache);
}

"""

new_content = content[:start_idx] + replacement + content[end_idx:]

with open('glassbox/sr/cpp/eval.h', 'w') as f:
    f.write(new_content)
