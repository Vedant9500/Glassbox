# Symbolic Regression Module

# v2: Meta-Operations (continuous parametric)
from .meta_ops import (
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
from .hard_concrete import (
    hard_concrete_sample,
    HardConcreteGate,
    HardConcreteSelector,
    HardConcreteOperationSelector,
    anneal_tau,
    anneal_beta,
)

# v2: Operation Nodes and DAG
from .operation_node import (
    OperationNode,
    OperationNodeSimple,
    OperationLayer,
)
from .operation_dag import (
    OperationDAG,
    OperationDAGSimple,
    ONNLoss,
    train_onn,
)
# v2: Hybrid Optimization
from .hybrid_optimizer import (
    LBFGSConstantOptimizer,
    EvolutionaryOptimizer,
    HybridOptimizer,
    GradientGuidedEvolution,
)

# v2: Benchmarking
from .benchmark import (
    BaselineMLP,
    BaselineLSTM,
    BaselineCNN,
    BenchmarkRunner,
    run_all_benchmarks,
    quick_comparison,
    generate_polynomial_data,
    generate_time_series_data,
    generate_multivariate_data,
    get_device,
)

# v2: Evolutionary Training (PROPER approach)
from .evolution import (
    EvolutionaryONNTrainer,
    train_onn_evolutionary,
    random_operation_init,
    mutate_operations,
    refine_constants,
)

# v2: Visualization
from .visualization import (
    ONNVisualizer,
    LiveTrainingVisualizer,
    create_network_diagram,
    visualize_evolution,
)

# v2: Post-Training Pruning
from .pruning import (
    PostTrainingPruner,
    prune_model,
    analyze_model_sensitivity,
)

# v2: Risk-Seeking Policy Gradient
from .risk_seeking_policy_gradient import (
    GradientMonitor,
    RiskSeekingEvolutionMixin,
    compute_risk_seeking_fitness,
    compute_selection_probabilities_rspg,
)

# v2: Multi-Start BFGS Optimizer
from .bfgs_optimizer import (
    RegularizedBFGS,
    MultiStartBFGS,
    IterativeBFGSRefiner,
    fit_coefficients_bfgs,
    build_formula_from_weights,
)


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
    
    # v2: Benchmarking
    'BaselineMLP',
    'BaselineLSTM',
    'BaselineCNN',
    'BenchmarkRunner',
    'run_all_benchmarks',
    'quick_comparison',
    
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
    
    # v2: Multi-Start BFGS Optimizer
    'RegularizedBFGS',
    'MultiStartBFGS',
    'IterativeBFGSRefiner',
    'fit_coefficients_bfgs',
    'build_formula_from_weights',
]

