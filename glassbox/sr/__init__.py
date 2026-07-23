# Symbolic Regression Module
#
# Production fit path: GlassboxRegressor → C++ _core.run_evolution (+ optional
# curve classifier / universal proposer). Alternate stacks below remain
# importable for research/legacy; they are not paid for at import of the
# public estimator (S10-2: lazy-load heavy alternate stacks).

# Public sklearn estimator (S1-11) — primary surface
from glassbox.sr.sklearn_wrapper import GlassboxRegressor

# v2: FPIP v2 (used by fast-path / proposer handoff)
from glassbox.sr.fpip_v2 import (
    build_fpip_v2_from_fast_path,
    validate_fpip_v2_payload,
    FPIPv2,
)
# Back-compat alias (older __all__ listed FPIPv2Payload)
FPIPv2Payload = FPIPv2

# Lightweight meta-ops / constants used by multiple paths
from glassbox.sr.operations.meta_ops import (
    MetaPeriodic,
    MetaPower,
    MetaArithmetic,
    MetaArithmeticExtended,
    MetaAggregation,
    MetaExp,
    MetaLog,
    MetaOperationLibrary,
    create_meta_op,
    KNOWN_CONSTANTS,
    snap_to_constant,
    snap_tensor_to_constants,
    get_constant_symbol,
    snap_edge_weights,
    snap_value_to_constant,
    ConstantAwareLinear,
)


def __getattr__(name: str):
    """Lazy-load alternate stacks (S10-2) so ``import glassbox.sr`` stays light.

    Production fit does not need hard_concrete, Python evolution, phased
    regression, HC/RSPG, or visualization; those modules remain available by
    name for research/legacy imports.
    """
    # v2: Hard Concrete
    if name in (
        "hard_concrete_sample",
        "HardConcreteGate",
        "HardConcreteSelector",
        "HardConcreteOperationSelector",
        "anneal_tau",
        "anneal_beta",
    ):
        from glassbox.sr import hard_concrete as _hc
        mapping = {
            "hard_concrete_sample": _hc.hard_concrete_sample,
            "HardConcreteGate": _hc.HardConcreteGate,
            "HardConcreteSelector": _hc.HardConcreteSelector,
            "HardConcreteOperationSelector": _hc.HardConcreteOperationSelector,
            "anneal_tau": _hc.anneal_tau,
            "anneal_beta": _hc.anneal_beta,
        }
        return mapping[name]

    # v2: Operation Nodes and DAG
    if name in (
        "OperationDAG",
        "OperationDAGSimple",
        "ONNLoss",
        "train_onn",
        "OperationNode",
        "OperationNodeSimple",
        "OperationLayer",
    ):
        from glassbox.sr.core import operation_dag as _dag
        from glassbox.sr.core import operation_node as _node
        mapping = {
            "OperationDAG": _dag.OperationDAG,
            "OperationDAGSimple": _dag.OperationDAGSimple,
            "ONNLoss": _dag.ONNLoss,
            "train_onn": _dag.train_onn,
            "OperationNode": _node.OperationNode,
            "OperationNodeSimple": _node.OperationNodeSimple,
            "OperationLayer": _node.OperationLayer,
        }
        return mapping[name]

    # v2: Hybrid / BFGS optimizers
    if name in (
        "LBFGSConstantOptimizer",
        "EvolutionaryOptimizer",
        "HybridOptimizer",
        "GradientGuidedEvolution",
        "RegularizedBFGS",
        "MultiStartBFGS",
        "IterativeBFGSRefiner",
        "fit_coefficients_bfgs",
        "build_formula_from_weights",
    ):
        from glassbox.sr.optimizers import hybrid_optimizer as _hyb
        from glassbox.sr.optimizers import bfgs_optimizer as _bfgs
        mapping = {
            "LBFGSConstantOptimizer": _hyb.LBFGSConstantOptimizer,
            "EvolutionaryOptimizer": _hyb.EvolutionaryOptimizer,
            "HybridOptimizer": _hyb.HybridOptimizer,
            "GradientGuidedEvolution": _hyb.GradientGuidedEvolution,
            "RegularizedBFGS": _bfgs.RegularizedBFGS,
            "MultiStartBFGS": _bfgs.MultiStartBFGS,
            "IterativeBFGSRefiner": _bfgs.IterativeBFGSRefiner,
            "fit_coefficients_bfgs": _bfgs.fit_coefficients_bfgs,
            "build_formula_from_weights": _bfgs.build_formula_from_weights,
        }
        return mapping[name]

    # v2: Evolutionary Training (legacy Python path)
    if name in (
        "EvolutionaryONNTrainer",
        "train_onn_evolutionary",
        "random_operation_init",
        "mutate_operations",
        "refine_constants",
    ):
        try:
            from glassbox.evolution import (
                EvolutionaryONNTrainer,
                train_onn_evolutionary,
                random_operation_init,
                mutate_operations,
                refine_constants,
            )
        except ImportError as exc:  # pragma: no cover
            raise AttributeError(name) from exc
        mapping = {
            "EvolutionaryONNTrainer": EvolutionaryONNTrainer,
            "train_onn_evolutionary": train_onn_evolutionary,
            "random_operation_init": random_operation_init,
            "mutate_operations": mutate_operations,
            "refine_constants": refine_constants,
        }
        return mapping[name]

    # v2: Visualization
    if name in (
        "ONNVisualizer",
        "LiveTrainingVisualizer",
        "create_network_diagram",
        "visualize_evolution",
    ):
        from glassbox.sr import visualization as _viz
        mapping = {
            "ONNVisualizer": _viz.ONNVisualizer,
            "LiveTrainingVisualizer": _viz.LiveTrainingVisualizer,
            "create_network_diagram": _viz.create_network_diagram,
            "visualize_evolution": _viz.visualize_evolution,
        }
        return mapping[name]

    # v2: Post-Training Pruning
    if name in ("PostTrainingPruner", "prune_model", "analyze_model_sensitivity"):
        from glassbox.sr import pruning as _pr
        mapping = {
            "PostTrainingPruner": _pr.PostTrainingPruner,
            "prune_model": _pr.prune_model,
            "analyze_model_sensitivity": _pr.analyze_model_sensitivity,
        }
        return mapping[name]

    # v2: Risk-Seeking Policy Gradient
    if name in (
        "GradientMonitor",
        "RiskSeekingEvolutionMixin",
        "compute_risk_seeking_fitness",
        "compute_selection_probabilities_rspg",
    ):
        from glassbox.sr import risk_seeking_policy_gradient as _rspg
        mapping = {
            "GradientMonitor": _rspg.GradientMonitor,
            "RiskSeekingEvolutionMixin": _rspg.RiskSeekingEvolutionMixin,
            "compute_risk_seeking_fitness": _rspg.compute_risk_seeking_fitness,
            "compute_selection_probabilities_rspg": _rspg.compute_selection_probabilities_rspg,
        }
        return mapping[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Public estimator
    "GlassboxRegressor",
    # Meta-Operations
    "MetaPeriodic",
    "MetaPower",
    "MetaArithmetic",
    "MetaArithmeticExtended",
    "MetaAggregation",
    "MetaExp",
    "MetaLog",
    "MetaOperationLibrary",
    "create_meta_op",
    "KNOWN_CONSTANTS",
    "snap_to_constant",
    "snap_tensor_to_constants",
    "get_constant_symbol",
    "snap_edge_weights",
    "snap_value_to_constant",
    "ConstantAwareLinear",
    # Hard Concrete (lazy)
    "hard_concrete_sample",
    "HardConcreteGate",
    "HardConcreteSelector",
    "HardConcreteOperationSelector",
    "anneal_tau",
    "anneal_beta",
    # DAG (lazy)
    "OperationNode",
    "OperationNodeSimple",
    "OperationLayer",
    "OperationDAG",
    "OperationDAGSimple",
    "ONNLoss",
    "train_onn",
    # Optimizers (lazy)
    "LBFGSConstantOptimizer",
    "EvolutionaryOptimizer",
    "HybridOptimizer",
    "GradientGuidedEvolution",
    "RegularizedBFGS",
    "MultiStartBFGS",
    "IterativeBFGSRefiner",
    "fit_coefficients_bfgs",
    "build_formula_from_weights",
    # Evolutionary Training (lazy)
    "EvolutionaryONNTrainer",
    "train_onn_evolutionary",
    "random_operation_init",
    "mutate_operations",
    "refine_constants",
    # Visualization (lazy)
    "ONNVisualizer",
    "LiveTrainingVisualizer",
    "create_network_diagram",
    "visualize_evolution",
    # Pruning (lazy)
    "PostTrainingPruner",
    "prune_model",
    "analyze_model_sensitivity",
    # RSPG (lazy)
    "GradientMonitor",
    "RiskSeekingEvolutionMixin",
    "compute_risk_seeking_fitness",
    "compute_selection_probabilities_rspg",
    # FPIP v2
    "build_fpip_v2_from_fast_path",
    "validate_fpip_v2_payload",
    "FPIPv2",
    "FPIPv2Payload",
]
