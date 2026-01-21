# Symbolic Regression Module

from .tree import DiffTreeNode, SymbolicTree, UNARY_OPS, BINARY_OPS
from .searcher import SymbolicSearcher

# v1: Operation-based RNN (legacy)
from .operation_rnn import (
    OperationCell,
    OperationRNN,
    OperationLSTM,
    OperationLSTMCell,
    set_global_temperature,
    set_global_hard_mode,
    anneal_temperature,
    compute_regularized_loss,
    UNARY_OPS as OP_RNN_UNARY_OPS,
    BINARY_OPS as OP_RNN_BINARY_OPS,
    AGGREGATION_OPS as OP_RNN_AGGREGATION_OPS,
)

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
)

# v2: Differentiable Routing
from .routing import (
    DifferentiableRouter,
    EdgeWeights,
    RoutedOperationInput,
    AdaptiveArityRouter,
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


__all__ = [
    # Tree-based symbolic regression
    'DiffTreeNode',
    'SymbolicTree',
    'SymbolicSearcher',
    
    # v1: Operation-based RNN (legacy)
    'OperationCell',
    'OperationRNN',
    'OperationLSTM',
    'OperationLSTMCell',
    'set_global_temperature',
    'set_global_hard_mode',
    'anneal_temperature',
    'compute_regularized_loss',
    
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
    
    # v2: Routing
    'DifferentiableRouter',
    'EdgeWeights',
    'RoutedOperationInput',
    'AdaptiveArityRouter',
    
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
]
