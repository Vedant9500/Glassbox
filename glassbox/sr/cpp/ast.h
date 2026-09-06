#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <list>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace sr {

// Math constants (shared; prefer over M_PI/M_E macros).
inline constexpr double kPi = 3.14159265358979323846;
inline constexpr double kE = 2.71828182845904523536;

// Unified output-weight activity threshold (S5-6).
// Eval, formula export, and active_complexity must agree so terms that affect
// predictions are not silently dropped from displayed formulas.
inline constexpr double kOutputWeightActive = 1e-6;
// Compact/simplify may still drop weights far below activity (dead columns).
inline constexpr double kOutputWeightDead = 1e-8;


enum class NodeType {
    Input,
    Constant,
    Unary,
    Binary
};

// Operator enums for discrete ops
enum class UnaryOp {
    Periodic,
    Power,
    IntPow,
    Exp,
    Log,
    Abs
};

enum class BinaryOp {
    Arithmetic,
    Division,
    Aggregation  // Sum, Mean, Max
};

// Represents a node in the computational graph
struct OpNode {
    NodeType type = NodeType::Constant;

    // For Input nodes
    int feature_idx = 0;

    // For Constant nodes
    double value = 0.0;

    // For Unary/Binary nodes — default so value-init / default-construct is defined
    // (Google: initialize all members; avoids indeterminate enums on seed/export paths).
    UnaryOp unary_op = UnaryOp::Periodic;
    BinaryOp binary_op = BinaryOp::Arithmetic;

    // Meta-operation parameters
    double p = 1.0;          // Power / IntPow exponent
    double omega = 1.0;      // Periodic frequency
    double phi = 0.0;        // Periodic phase
    double amplitude = 1.0;  // Periodic amplitude
    double beta = 1.5;       // Arithmetic (1.0 = add, 2.0 = mul)
    double gamma = 1.0;      // Arithmetic sign (for sub/div)
    double tau = 1.0;        // Aggregation temperature

    // Child pointers (indices in the layer)
    int left_child = -1;
    int right_child = -1;
};

// Mark nodes reachable from active output weights (shared by complexity metrics).
inline void mark_active_nodes(const std::vector<OpNode>& nodes,
                              const std::vector<double>& output_weights,
                              std::vector<char>& active) {
    active.assign(nodes.size(), 0);
    if (nodes.empty()) return;
    const int n = static_cast<int>(nodes.size());
    const int nw = static_cast<int>(output_weights.size());
    for (int i = 0; i < n && i < nw; ++i) {
        if (std::abs(output_weights[i]) <= kOutputWeightActive) continue;
        std::vector<int> stack = {i};
        while (!stack.empty()) {
            int idx = stack.back();
            stack.pop_back();
            if (idx < 0 || idx >= n || active[idx]) continue;
            active[idx] = 1;
            const auto& node = nodes[idx];
            if ((node.type == NodeType::Unary || node.type == NodeType::Binary) &&
                node.left_child >= 0) {
                stack.push_back(node.left_child);
            }
            if (node.type == NodeType::Binary && node.right_child >= 0) {
                stack.push_back(node.right_child);
            }
        }
    }
}

// §3.414: true when victim also feeds another active output root — zeroing
// its direct weight would drop an independent additive contribution, not
// just the composed operand. Bounds-checked; inactive roots never share.
inline bool is_shared_basis(const std::vector<OpNode>& nodes,
                            const std::vector<double>& output_weights,
                            int victim) {
    const int n = static_cast<int>(nodes.size());
    const int nw = static_cast<int>(output_weights.size());
    if (victim < 0 || victim >= n) return false;
    for (int r = 0; r < n && r < nw; ++r) {
        if (r == victim) continue;
        if (std::abs(output_weights[static_cast<size_t>(r)]) <= kOutputWeightActive)
            continue;
        std::vector<int> stack = {r};
        std::vector<char> seen(static_cast<size_t>(n), 0);
        while (!stack.empty()) {
            int idx = stack.back();
            stack.pop_back();
            if (idx < 0 || idx >= n || seen[static_cast<size_t>(idx)]) continue;
            seen[static_cast<size_t>(idx)] = 1;
            if (idx == victim) return true;
            const auto& node = nodes[static_cast<size_t>(idx)];
            if ((node.type == NodeType::Unary || node.type == NodeType::Binary) &&
                node.left_child >= 0) {
                stack.push_back(node.left_child);
            }
            if (node.type == NodeType::Binary && node.right_child >= 0) {
                stack.push_back(node.right_child);
            }
        }
    }
    return false;
}

// Pre-allocated array representing a formula's structure
struct IndividualGraph {
    std::vector<OpNode> nodes;
    std::vector<double> output_weights; // Linear combination of top nodes
    double output_bias = 0.0;

    double fitness = 1e9; // Penalized fitness (uses weighted MSE when weights set)
    double raw_mse = 1e9; // Unweighted mathematical MSE (diagnostics / back-compat)
    double weighted_mse = 1e9; // Weighted MSE when y_weights provided; else == raw_mse
    // E6: when true, evaluate_population may skip re-scoring this individual
    // (elites / children already scored at birth remain valid until mutated).
    bool fitness_valid = false;

    // NSGA-II fields (P5)
    int pareto_rank = 0;           // Non-domination rank (0 = Pareto front)
    double crowding_distance = 0.0; // Crowding distance within the same rank
    int age = 0;                    // AFPO: generations survived (0 = newly created)
    int complexity() const {
        return static_cast<int>(nodes.size());
    } // AST node count as 2nd objective

    int active_complexity() const {
        if (nodes.empty()) return 0;
        std::vector<char> active;
        mark_active_nodes(nodes, output_weights, active);

        int total = 0;
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
            if (!active[i]) continue;
            const auto& node = nodes[i];
            switch (node.type) {
                case NodeType::Input:
                    total += 1;
                    break;
                case NodeType::Constant:
                    total += 1;
                    break;
                case NodeType::Unary:
                    if (node.unary_op == UnaryOp::IntPow) total += 2;
                    else if (node.unary_op == UnaryOp::Abs) total += 2;
                    else if (node.unary_op == UnaryOp::Power) total += 3;
                    else if (node.unary_op == UnaryOp::Periodic) total += 3;
                    else total += 4;  // Exp, Log
                    break;
                case NodeType::Binary:
                    if (node.binary_op == BinaryOp::Arithmetic) total += 2;
                    else if (node.binary_op == BinaryOp::Aggregation) total += 3;
                    else total += 5;
                    break;
            }
        }
        return std::max(1, total);
    }

    // Count distinct active nodes (S5-13): unit-matched with nodes.size().
    int active_node_count() const {
        if (nodes.empty()) return 0;
        std::vector<char> active;
        mark_active_nodes(nodes, output_weights, active);
        int n = 0;
        for (char a : active) if (a) ++n;
        return n;
    }
};


// M-161: reset type-irrelevant fields to struct defaults so stale params
// (e.g. a Constant carrying an old p/omega, or Division carrying beta) can
// never leak into hashes, exports, or later type transitions.
inline void normalize_node_fields(OpNode& node) {
    OpNode dflt;
    if (node.type == NodeType::Input) {
        node.value = 0.0;
        node.unary_op = dflt.unary_op;
        node.binary_op = dflt.binary_op;
        node.p = dflt.p; node.omega = dflt.omega; node.phi = dflt.phi;
        node.amplitude = dflt.amplitude;
        node.beta = dflt.beta; node.gamma = dflt.gamma; node.tau = dflt.tau;
        node.left_child = -1; node.right_child = -1;
    } else if (node.type == NodeType::Constant) {
        node.unary_op = dflt.unary_op;
        node.binary_op = dflt.binary_op;
        node.p = dflt.p; node.omega = dflt.omega; node.phi = dflt.phi;
        node.amplitude = dflt.amplitude;
        node.beta = dflt.beta; node.gamma = dflt.gamma; node.tau = dflt.tau;
        node.left_child = -1; node.right_child = -1;
    } else if (node.type == NodeType::Unary) {
        node.binary_op = dflt.binary_op;
        node.beta = dflt.beta; node.gamma = dflt.gamma; node.tau = dflt.tau;
        node.right_child = -1;
        switch (node.unary_op) {
            case UnaryOp::Periodic: node.p = dflt.p; break;
            case UnaryOp::Power:
            case UnaryOp::IntPow:
                node.omega = dflt.omega; node.phi = dflt.phi;
                node.amplitude = dflt.amplitude; break;
            case UnaryOp::Exp: node.p = dflt.p; node.amplitude = dflt.amplitude; break;
            case UnaryOp::Log:
            case UnaryOp::Abs:
                node.p = dflt.p; node.omega = dflt.omega; node.phi = dflt.phi;
                node.amplitude = dflt.amplitude; break;
        }
    } else if (node.type == NodeType::Binary) {
        node.unary_op = dflt.unary_op;
        node.p = dflt.p; node.omega = dflt.omega; node.phi = dflt.phi;
        node.amplitude = dflt.amplitude;
        switch (node.binary_op) {
            case BinaryOp::Arithmetic: node.tau = dflt.tau; break;
            case BinaryOp::Division:
                node.beta = dflt.beta; node.gamma = dflt.gamma;
                node.tau = dflt.tau; break;
            case BinaryOp::Aggregation:
                node.beta = dflt.beta; node.gamma = dflt.gamma; break;
        }
    }
}


// Remove dead nodes and compact the graph (shared by parse post-compile and simplify).
// Child indices are bounds-checked so malformed graphs do not OOB.
inline void compact_graph(IndividualGraph& graph) {
    if (graph.nodes.empty()) return;

    const int n = static_cast<int>(graph.nodes.size());
    std::vector<int> new_indices(static_cast<size_t>(n), -1);
    std::vector<OpNode> new_nodes;
    std::vector<double> new_weights;

    // Mark used nodes from output weights
    std::vector<bool> used(static_cast<size_t>(n), false);
    for (size_t i = 0; i < graph.output_weights.size() && i < static_cast<size_t>(n); ++i) {
        if (std::abs(graph.output_weights[i]) > kOutputWeightDead) {
            used[i] = true;
        }
    }

    // Propagate used flags downwards
    for (int i = n - 1; i >= 0; --i) {
        if (!used[static_cast<size_t>(i)]) continue;
        const OpNode& node = graph.nodes[static_cast<size_t>(i)];
        if (node.type == NodeType::Unary) {
            if (node.left_child >= 0 && node.left_child < n) {
                used[static_cast<size_t>(node.left_child)] = true;
            }
        } else if (node.type == NodeType::Binary) {
            if (node.left_child >= 0 && node.left_child < n) {
                used[static_cast<size_t>(node.left_child)] = true;
            }
            if (node.right_child >= 0 && node.right_child < n) {
                used[static_cast<size_t>(node.right_child)] = true;
            }
        }
    }

    // Build compacted nodes
    // §3.377: a retained operator node whose child maps to -1 keeps -1 here
    // ON PURPOSE. The used-propagation above marks every in-range child of a
    // retained node, so -1 survivors are exactly the out-of-range (invalid)
    // references — the evaluator substitutes zero and topology validation
    // rejects the graph downstream (§3.118). Do not silently drop the node —
    // that would renumber output weights past a loud validation failure.
    for (int i = 0; i < n; ++i) {
        if (!used[static_cast<size_t>(i)]) continue;
        new_indices[static_cast<size_t>(i)] = static_cast<int>(new_nodes.size());
        OpNode node = graph.nodes[static_cast<size_t>(i)];
        if (node.type == NodeType::Unary) {
            if (node.left_child >= 0 && node.left_child < n) {
                node.left_child = new_indices[static_cast<size_t>(node.left_child)];
            } else {
                node.left_child = -1;
            }
        } else if (node.type == NodeType::Binary) {
            if (node.left_child >= 0 && node.left_child < n) {
                node.left_child = new_indices[static_cast<size_t>(node.left_child)];
            } else {
                node.left_child = -1;
            }
            if (node.right_child >= 0 && node.right_child < n) {
                node.right_child = new_indices[static_cast<size_t>(node.right_child)];
            } else {
                node.right_child = -1;
            }
        }
        // M-161: normalize type-irrelevant fields on copy — stale params
        // otherwise survive into hashes/exports (hashing excludes them by
        // type, but serialized consumers see meaningless values).
        normalize_node_fields(node);
        new_nodes.push_back(node);

        if (static_cast<size_t>(i) < graph.output_weights.size() &&
            std::abs(graph.output_weights[static_cast<size_t>(i)]) > kOutputWeightDead) {
            while (static_cast<int>(new_weights.size()) <= new_indices[static_cast<size_t>(i)]) {
                new_weights.push_back(0.0);
            }
            new_weights[static_cast<size_t>(new_indices[static_cast<size_t>(i)])] =
                graph.output_weights[static_cast<size_t>(i)];
        }
    }

    graph.nodes = std::move(new_nodes);
    graph.output_weights = std::move(new_weights);
}

// --- Structural Hashing -------------------------------------------------
// Combines node type, op, quantized parameters, and children hashes into
// a 64-bit fingerprint. Two subtrees with the same hash produce identical
// outputs and can share cached Eigen::ArrayXd results.

inline uint64_t hash_combine(uint64_t seed, uint64_t v) {
    seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    return seed;
}

// S5-5: default was 2 decimals -> SharedCache collisions (omega=1.004 vs 1.006).
// Use 8 decimals for eval cache keys. decimals < 0 => bit-exact double bits.
// Coarse CSE (if desired) can call quantize(v, 2) explicitly.
// §3.375: boundary contract. Values straddling a rounding boundary hash
// distinctly (cache miss only — safe, perf-level). Values inside one bin
// collide by design and SHARE cache payloads, so this grid is approximate:
// keep it cache/diversity-only and never infer semantic equality from a
// hash (cleanup dedup decides by output correlation, not hashes).
inline uint64_t quantize(double v, int decimals = 8) {
    if (decimals < 0) {
        uint64_t u = 0;
        static_assert(sizeof(double) == sizeof(uint64_t), "double must be 64-bit");
        std::memcpy(&u, &v, sizeof(u));
        return u;
    }
    // Quantize to N decimal places for near-match dedup
    double scale = 1.0;
    for (int i = 0; i < decimals; ++i) scale *= 10.0;
    // Clamp huge values so round stays in int64 range.
    double scaled = v * scale;
    if (!std::isfinite(scaled)) {
        uint64_t u = 0;
        std::memcpy(&u, &v, sizeof(u));
        return u;
    }
    const double kMaxI64 = static_cast<double>(std::numeric_limits<int64_t>::max());
    const double kMinI64 = static_cast<double>(std::numeric_limits<int64_t>::min());
    if (scaled > kMaxI64) scaled = kMaxI64;
    if (scaled < kMinI64) scaled = kMinI64;
    int64_t q = static_cast<int64_t>(std::round(scaled));
    uint64_t u;
    std::memcpy(&u, &q, sizeof(u));
    return u;
}

// Compute structural hash for node at index `idx` in the graph.
// `node_hashes` must be pre-allocated to graph.nodes.size() and filled
// bottom-up (lower indices first, which is the natural DAG order).
inline uint64_t compute_node_hash(const IndividualGraph& graph, int idx,
                                   std::vector<uint64_t>& node_hashes) {
    const auto& node = graph.nodes[idx];
    uint64_t h = static_cast<uint64_t>(node.type);

    switch (node.type) {
        case NodeType::Input:
            h = hash_combine(h, static_cast<uint64_t>(node.feature_idx));
            break;
        case NodeType::Constant:
            h = hash_combine(h, quantize(node.value));
            break;
        case NodeType::Unary:
            h = hash_combine(h, static_cast<uint64_t>(node.unary_op));
            h = hash_combine(h, quantize(node.p));
            h = hash_combine(h, quantize(node.omega));
            h = hash_combine(h, quantize(node.phi));
            h = hash_combine(h, quantize(node.amplitude));
            if (node.left_child >= 0 && node.left_child < idx) {
                h = hash_combine(h, node_hashes[node.left_child]);
            } else {
                // §3.374: invalid children hash a distinct sentinel instead
                // of vanishing — a malformed node can no longer collide with
                // a well-formed node that merely shares op/params, and
                // evaluation's zero-substitution has no cache twin.
                h = hash_combine(h, 0xBADC0DEDU);
            }
            break;
        case NodeType::Binary:
            h = hash_combine(h, static_cast<uint64_t>(node.binary_op));
            h = hash_combine(h, quantize(node.beta));
            h = hash_combine(h, quantize(node.gamma));
            h = hash_combine(h, quantize(node.tau));
            if (node.left_child >= 0 && node.left_child < idx) {
                h = hash_combine(h, node_hashes[node.left_child]);
            } else {
                h = hash_combine(h, 0x1EF7BABEU);
            }
            if (node.right_child >= 0 && node.right_child < idx) {
                h = hash_combine(h, node_hashes[node.right_child]);
            } else {
                h = hash_combine(h, 0x6BADF00DU);
            }
            break;
    }
    return h;
}

// Bounded LRU subtree cache (P-01 / P-10).
// Each entry stores a full ArrayXd(n_samples); without caps a single generation
// can spike multi-GB. Evicts least-recently-used entries when over entry/byte limits.
class SubtreeCache {
public:
    static constexpr std::size_t kDefaultMaxEntries = 4096;
    // ~256 MiB default hard ceiling for cached ArrayXd payloads.
    static constexpr std::size_t kDefaultMaxBytes = 256ull * 1024ull * 1024ull;

    using key_type = uint64_t;
    using mapped_type = Eigen::ArrayXd;
    using entry_type = std::pair<key_type, mapped_type>;
    using list_type = std::list<entry_type>;
    using iterator = list_type::iterator;
    using const_iterator = list_type::const_iterator;

    explicit SubtreeCache(std::size_t max_entries = kDefaultMaxEntries,
                          std::size_t max_bytes = kDefaultMaxBytes)
        : max_entries_(max_entries == 0 ? kDefaultMaxEntries : max_entries),
          max_bytes_(max_bytes == 0 ? kDefaultMaxBytes : max_bytes) {}

    SubtreeCache(const SubtreeCache& other)
        : max_entries_(other.max_entries_),
          max_bytes_(other.max_bytes_),
          bytes_used_(other.bytes_used_),
          evictions_(other.evictions_) {
        for (const auto& entry : other.order_) {
            order_.push_back(entry);
            index_.emplace(order_.back().first, std::prev(order_.end()));
        }
    }

    SubtreeCache& operator=(const SubtreeCache& other) {
        if (this == &other) return *this;
        clear();
        max_entries_ = other.max_entries_;
        max_bytes_ = other.max_bytes_;
        bytes_used_ = other.bytes_used_;
        evictions_ = other.evictions_;
        for (const auto& entry : other.order_) {
            order_.push_back(entry);
            index_.emplace(order_.back().first, std::prev(order_.end()));
        }
        return *this;
    }

    SubtreeCache(SubtreeCache&& other) noexcept
        : order_(std::move(other.order_)),
          index_(std::move(other.index_)),
          max_entries_(other.max_entries_),
          max_bytes_(other.max_bytes_),
          bytes_used_(other.bytes_used_),
          evictions_(other.evictions_) {
        other.bytes_used_ = 0;
    }

    SubtreeCache& operator=(SubtreeCache&& other) noexcept {
        if (this == &other) return *this;
        order_ = std::move(other.order_);
        index_ = std::move(other.index_);
        max_entries_ = other.max_entries_;
        max_bytes_ = other.max_bytes_;
        bytes_used_ = other.bytes_used_;
        evictions_ = other.evictions_;
        other.bytes_used_ = 0;
        return *this;
    }

    iterator begin() { return order_.begin(); }
    iterator end() { return order_.end(); }
    const_iterator begin() const { return order_.begin(); }
    const_iterator end() const { return order_.end(); }
    const_iterator cbegin() const { return order_.cbegin(); }
    const_iterator cend() const { return order_.cend(); }

    std::size_t size() const { return order_.size(); }
    bool empty() const { return order_.empty(); }
    std::size_t bytes_used() const { return bytes_used_; }
    std::size_t evictions() const { return evictions_; }
    std::size_t max_entries() const { return max_entries_; }
    std::size_t max_bytes() const { return max_bytes_; }

    void add_evictions(std::size_t n) { evictions_ += n; }

    void set_limits(std::size_t max_entries, std::size_t max_bytes) {
        max_entries_ = max_entries == 0 ? kDefaultMaxEntries : max_entries;
        max_bytes_ = max_bytes == 0 ? kDefaultMaxBytes : max_bytes;
        evict_while_over_budget();
    }

    void clear() {
        order_.clear();
        index_.clear();
        bytes_used_ = 0;
        // Keep eviction counter cumulative within a process/run for diagnostics.
    }

    iterator find(key_type key) {
        auto it = index_.find(key);
        if (it == index_.end()) return order_.end();
        touch(it->second);
        return it->second;
    }

    const_iterator find(key_type key) const {
        auto it = index_.find(key);
        if (it == index_.end()) return order_.cend();
        return it->second;
    }

    // Insert or replace; returns true if stored (may evict LRU peers).
    bool insert_or_assign(key_type key, mapped_type value) {
        const std::size_t new_bytes = entry_bytes(value);
        // Refuse a single entry larger than the whole budget.
        if (new_bytes > max_bytes_) {
            ++evictions_;
            return false;
        }

        auto it = index_.find(key);
        if (it != index_.end()) {
            bytes_used_ -= entry_bytes(it->second->second);
            it->second->second = std::move(value);
            bytes_used_ += new_bytes;
            touch(it->second);
            evict_while_over_budget(/*keep=*/key);
            return true;
        }

        order_.emplace_front(key, std::move(value));
        index_.emplace(key, order_.begin());
        bytes_used_ += new_bytes;
        evict_while_over_budget(/*keep=*/key);
        return index_.find(key) != index_.end();
    }

    // Map-compatible insert: no-op if key already present.
    template <typename M>
    std::pair<iterator, bool> try_emplace(key_type key, M&& value) {
        auto it = index_.find(key);
        if (it != index_.end()) {
            touch(it->second);
            return {it->second, false};
        }
        mapped_type owned(std::forward<M>(value));
        const std::size_t new_bytes = entry_bytes(owned);
        if (new_bytes > max_bytes_) {
            ++evictions_;
            return {order_.end(), false};
        }
        order_.emplace_front(key, std::move(owned));
        auto inserted = index_.emplace(key, order_.begin());
        bytes_used_ += new_bytes;
        evict_while_over_budget(/*keep=*/key);
        auto kept = index_.find(key);
        if (kept == index_.end()) {
            return {order_.end(), false};
        }
        return {kept->second, inserted.second};
    }

    // Convenience for sites that previously used operator[] = value.
    mapped_type& operator[](key_type key) {
        auto it = index_.find(key);
        if (it != index_.end()) {
            touch(it->second);
            return it->second->second;
        }
        order_.emplace_front(key, mapped_type());
        index_.emplace(key, order_.begin());
        // Empty array contributes 0 bytes until assigned; callers that only
        // use operator[] for assignment should prefer insert_or_assign.
        evict_while_over_budget(/*keep=*/key);
        auto kept = index_.find(key);
        if (kept == index_.end()) {
            // Extremely constrained budget: re-insert empty sentinel.
            order_.emplace_front(key, mapped_type());
            index_[key] = order_.begin();
            return order_.begin()->second;
        }
        return kept->second->second;
    }

private:
    static std::size_t entry_bytes(const mapped_type& v) {
        if (v.size() <= 0) return 0;
        return static_cast<std::size_t>(v.size()) * sizeof(double);
    }

    void touch(iterator it) {
        if (it != order_.begin()) {
            order_.splice(order_.begin(), order_, it);
        }
    }

    void erase_iterator(iterator it) {
        bytes_used_ -= entry_bytes(it->second);
        index_.erase(it->first);
        order_.erase(it);
        ++evictions_;
    }

    void evict_while_over_budget(key_type keep = 0) {
        const bool has_keep = index_.find(keep) != index_.end();
        while ((order_.size() > max_entries_ || bytes_used_ > max_bytes_) && !order_.empty()) {
            auto victim = std::prev(order_.end());
            if (has_keep && victim->first == keep) {
                // Keep the just-inserted key; evict next-oldest if possible.
                if (order_.size() == 1) break;
                victim = std::prev(victim);
                if (victim->first == keep) break;
            }
            erase_iterator(victim);
        }
        // If still over bytes with only `keep`, drop keep as last resort.
        if (bytes_used_ > max_bytes_ && !order_.empty()) {
            erase_iterator(order_.begin());
        }
    }

    list_type order_; // front = MRU, back = LRU
    std::unordered_map<key_type, iterator> index_;
    std::size_t max_entries_;
    std::size_t max_bytes_;
    std::size_t bytes_used_ = 0;
    std::size_t evictions_ = 0;
};

} // namespace sr
