# Symbolic Regression Module

# v2: Meta-Operations (continuous parametric)
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
    # Constant snapping utilities
    KNOWN_CONSTANTS,
    snap_to_constant,
    snap_tensor_to_constants,
    get_constant_symbol,
    snap_edge_weights,
    snap_value_to_constant,
    ConstantAwareLinear,
)

# v2: Hard Concrete Distribution
from glassbox.sr.hard_concrete import (
    hard_concrete_sample,
    HardConcreteGate,
    HardConcreteSelector,
    HardConcreteOperationSelector,
    anneal_tau,
    anneal_beta,
)

# v2: Operation Nodes and DAG
from glassbox.sr.core.operation_dag import (
    OperationDAG,
    OperationDAGSimple,
    ONNLoss,
    train_onn,
)

from glassbox.sr.core.operation_node import (
    OperationNode,
    OperationNodeSimple,
    OperationLayer,
)

# v2: Hybrid Optimization
from glassbox.sr.optimizers.hybrid_optimizer import (
    LBFGSConstantOptimizer,
    EvolutionaryOptimizer,
    HybridOptimizer,
    GradientGuidedEvolution,
)

# v2: BFGS Optimizer
from glassbox.sr.optimizers.bfgs_optimizer import (
    RegularizedBFGS,
    MultiStartBFGS,
    IterativeBFGSRefiner,
    fit_coefficients_bfgs,
    build_formula_from_weights,
)

# v2: Benchmarking

# v2: Evolutionary Training (PROPER approach)
# Note: Main evolution module moved to glassbox.evolution
# Kept here for backward compatibility
try:
    from glassbox.evolution import (
        EvolutionaryONNTrainer,
        train_onn_evolutionary,
        random_operation_init,
        mutate_operations,
        refine_constants,
    )
except ImportError:
    # Fallback if glassbox.evolution is not available
    pass

# v2: Visualization
from glassbox.sr.visualization import (
    ONNVisualizer,
    LiveTrainingVisualizer,
    create_network_diagram,
    visualize_evolution,
)

# v2: Post-Training Pruning
from glassbox.sr.pruning import (
    PostTrainingPruner,
    prune_model,
    analyze_model_sensitivity,
)

# v2: Risk-Seeking Policy Gradient
from glassbox.sr.risk_seeking_policy_gradient import (
    GradientMonitor,
    RiskSeekingEvolutionMixin,
    compute_risk_seeking_fitness,
    compute_selection_probabilities_rspg,
)

# v2: FPIP v2
from glassbox.sr.fpip_v2 import (
    build_fpip_v2_from_fast_path,
    validate_fpip_v2_payload,
    FPIPv2,
)

# v2: GPU Optimization


__all__ = [
    # v2: Meta-Operations
    'MetaPeriodic',
    'MetaPower',
    'MetaArithmetic',
    'MetaArithmeticExtended',
    'MetaAggregation',
    'MetaExp',
    'MetaLog',
    'MetaOperationLibrary',
    'create_meta_op',
    
    # v2: Constant Snapping
    'KNOWN_CONSTANTS',
    'snap_to_constant',
    'snap_tensor_to_constants',
    'get_constant_symbol',
    'snap_edge_weights',
    'snap_value_to_constant',
    'ConstantAwareLinear',
    
    # v2: Hard Concrete
    'hard_concrete_sample',
    'HardConcreteGate',
    'HardConcreteSelector',
    'HardConcreteOperationSelector',
    'anneal_tau',
    'anneal_beta',
    
    # v2: DAG Architecture
    'OperationNode',
    'OperationNodeSimple',
    'OperationLayer',
    'OperationDAG',
    'OperationDAGSimple',
    'ONNLoss',
    'train_onn',
    
    # v2: Hybrid Optimization
    'LBFGSConstantOptimizer',
    'EvolutionaryOptimizer',
    'HybridOptimizer',
    'GradientGuidedEvolution',
    
    # v2: BFGS Optimizer
    'RegularizedBFGS',
    'MultiStartBFGS',
    'IterativeBFGSRefiner',
    'fit_coefficients_bfgs',
    'build_formula_from_weights',
    
    # v2: Evolutionary Training
    'EvolutionaryONNTrainer',
    'train_onn_evolutionary',
    'random_operation_init',
    'mutate_operations',
    'refine_constants',
    
    # v2: Visualization
    'ONNVisualizer',
    'LiveTrainingVisualizer',
    'create_network_diagram',
    'visualize_evolution',
    
    # v2: Post-Training Pruning
    'PostTrainingPruner',
    'prune_model',
    'analyze_model_sensitivity',
    
    # v2: Risk-Seeking Policy Gradient
    'GradientMonitor',
    'RiskSeekingEvolutionMixin',
    'compute_risk_seeking_fitness',
    'compute_selection_probabilities_rspg',
    
    # v2: FPIP v2
    'build_fpip_v2_from_fast_path',
    'validate_fpip_v2_payload',
    'FPIPv2Payload',
]
