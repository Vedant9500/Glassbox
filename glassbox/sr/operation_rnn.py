"""
Operation-Based Recurrent Neural Networks (v2).

Instead of traditional neurons with weights, each "neuron" is a learnable
mathematical operation (sin, log, +c, *c, ^2, etc.). The network learns
WHICH operation to apply at each step, along with any operation constants.

Key Features:
- Gumbel-Softmax for differentiable operation selection
- Multi-input aggregation: binary ops can reduce N inputs (sum_all, prod_all, etc.)
- Entropy regularization to encourage diverse operation exploration
- Numerical stability: safe log, safe div, clamped outputs
- Supports both simple RNN and LSTM-like architectures

v2 Changes:
- Added AGGREGATION_OPS for multi-input reduction
- Entropy loss to encourage operation diversity
- Better initialization to reduce identity bias
- Learnable edge weights as coefficients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Callable
import math

# =============================================================================
# Safe Mathematical Operations (avoid NaN/Inf)
# =============================================================================

def safe_log(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Safe log that handles negative and zero values."""
    return torch.log(torch.abs(x) + eps)

def safe_div(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Safe division that avoids divide-by-zero."""
    return x / (y + eps * torch.sign(y + 1e-9))

def safe_sqrt(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Safe sqrt for negative values."""
    return torch.sqrt(torch.abs(x) + eps)

def safe_exp(x: torch.Tensor, max_val: float = 10.0) -> torch.Tensor:
    """Safe exp with clamping to avoid overflow."""
    return torch.exp(torch.clamp(x, -max_val, max_val))

def safe_pow(x: torch.Tensor, power: float, eps: float = 1e-6) -> torch.Tensor:
    """Safe power for negative bases."""
    return torch.sign(x) * torch.pow(torch.abs(x) + eps, power)

def safe_prod(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Safe product with clamping to avoid explosion."""
    # Clamp individual values before product
    x_clamped = torch.clamp(x, -10.0, 10.0)
    return torch.prod(x_clamped, dim=dim)

# =============================================================================
# Operation Definitions
# =============================================================================

# Unary operations: take one input, may have a learnable constant
UNARY_OPS: Dict[str, Callable] = {
    'identity':  lambda x, c: x,
    'add_const': lambda x, c: x + c,
    'mul_const': lambda x, c: x * c,
    'square':    lambda x, c: x ** 2,
    'cube':      lambda x, c: x ** 3,
    'sqrt':      lambda x, c: safe_sqrt(x),
    'abs':       lambda x, c: torch.abs(x),
    'neg':       lambda x, c: -x,
    'sin':       lambda x, c: torch.sin(x),
    'cos':       lambda x, c: torch.cos(x),
    'tanh':      lambda x, c: torch.tanh(x),
    'sigmoid':   lambda x, c: torch.sigmoid(x),
    'exp':       lambda x, c: safe_exp(x),
    'log':       lambda x, c: safe_log(x),
    'inv':       lambda x, c: safe_div(1.0, x),
}
UNARY_NAMES = list(UNARY_OPS.keys())

# Binary operations: combine exactly two inputs
BINARY_OPS: Dict[str, Callable] = {
    'add':   lambda a, b: a + b,
    'sub':   lambda a, b: a - b,
    'mul':   lambda a, b: a * b,
    'div':   lambda a, b: safe_div(a, b),
    'max':   lambda a, b: torch.maximum(a, b),
    'min':   lambda a, b: torch.minimum(a, b),
    'pow':   lambda a, b: safe_pow(a, torch.clamp(b, -3, 3)),  # Keep tensor for gradients
}
BINARY_NAMES = list(BINARY_OPS.keys())

# Aggregation operations: reduce N inputs to 1 output
# These work on tensor with shape (..., N) and reduce along last dim
AGGREGATION_OPS: Dict[str, Callable] = {
    'sum_all':  lambda x: x.sum(dim=-1),
    'mean_all': lambda x: x.mean(dim=-1),
    'prod_all': lambda x: safe_prod(x, dim=-1),
    'max_all':  lambda x: x.max(dim=-1).values,
    'min_all':  lambda x: x.min(dim=-1).values,
    'std_all':  lambda x: x.std(dim=-1) + 1e-6,
    'norm_all': lambda x: torch.norm(x, dim=-1),
}
AGGREGATION_NAMES = list(AGGREGATION_OPS.keys())

# Output clamp range to prevent gradient explosion
CLAMP_MIN = -100.0
CLAMP_MAX = 100.0

# Global temperature for Gumbel-Softmax
_GLOBAL_TAU = 1.0
_GLOBAL_HARD = True


def set_global_temperature(tau: float):
    """Set global Gumbel-Softmax temperature."""
    global _GLOBAL_TAU
    _GLOBAL_TAU = tau


def set_global_hard_mode(hard: bool):
    """Set global hard/soft selection mode."""
    global _GLOBAL_HARD
    _GLOBAL_HARD = hard


def get_global_temperature() -> float:
    """Get current global temperature."""
    return _GLOBAL_TAU


# =============================================================================
# Operation Cell v2: Supports multi-input aggregation
# =============================================================================

class OperationCell(nn.Module):
    """
    A single operation cell that learns which mathematical operation to apply.
    
    Supports three types of operations:
    1. Unary: single input → output (sin, square, etc.)
    2. Binary: two inputs (x, h) → output (add, mul, etc.)
    3. Aggregation: N inputs → output (sum_all, prod_all, etc.)
    
    The operation is selected via Gumbel-Softmax for differentiability.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_binary_ops: bool = True,
        use_aggregation_ops: bool = True,
        n_aggregation_inputs: int = 4,
    ):
        """
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden/output features
            use_binary_ops: If True, can combine input + hidden via binary ops
            use_aggregation_ops: If True, can aggregate multiple inputs
            n_aggregation_inputs: Number of inputs for aggregation operations
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_binary_ops = use_binary_ops
        self.use_aggregation_ops = use_aggregation_ops
        self.n_aggregation_inputs = n_aggregation_inputs
        
        n_unary = len(UNARY_NAMES)
        n_binary = len(BINARY_NAMES) if use_binary_ops else 0
        n_agg = len(AGGREGATION_NAMES) if use_aggregation_ops else 0
        self.n_ops = n_unary + n_binary + n_agg
        
        # For each hidden unit, learn which operation to apply
        self.op_weights = nn.Parameter(torch.zeros(hidden_size, self.n_ops))
        
        # Learnable constants for each hidden unit (used by add_const, mul_const)
        self.constants = nn.Parameter(torch.ones(hidden_size))
        
        # Linear projections
        self.input_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.hidden_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Edge weights for aggregation: learn coefficients for each input
        if use_aggregation_ops:
            self.edge_weights = nn.Parameter(torch.ones(hidden_size, n_aggregation_inputs))
            self.agg_proj = nn.Linear(input_size + hidden_size, hidden_size * n_aggregation_inputs, bias=False)
        
        # Initialize with balanced weights (no identity bias)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better exploration of diverse operations."""
        n_unary = len(UNARY_NAMES)
        
        # Start with small random weights - no strong bias toward any operation
        nn.init.normal_(self.op_weights, mean=0.0, std=0.3)
        
        # Slightly favor "interesting" operations to encourage exploration
        interesting_ops = {
            'sin': 0.3, 'cos': 0.2, 'square': 0.4, 'add_const': 0.2,
            'mul_const': 0.2, 'tanh': 0.1, 'sqrt': 0.1
        }
        for i, name in enumerate(UNARY_NAMES):
            if name in interesting_ops:
                self.op_weights.data[:, i] += interesting_ops[name]
        
        # Initialize constants
        nn.init.normal_(self.constants, mean=1.0, std=0.5)
        
        # Initialize edge weights uniformly
        if self.use_aggregation_ops:
            nn.init.uniform_(self.edge_weights, 0.5, 1.5)
    
    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the operation cell.
        
        Args:
            x: Input tensor (batch, input_size)
            h: Previous hidden state (batch, hidden_size), optional
            
        Returns:
            output: New hidden state (batch, hidden_size)
            entropy: Entropy of operation selection (for regularization)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Project input to hidden size
        x_proj = self.input_proj(x)  # (batch, hidden_size)
        
        # Get operation probabilities
        op_logits = self.op_weights.unsqueeze(0).expand(batch_size, -1, -1)
        op_probs_soft = F.softmax(op_logits / _GLOBAL_TAU, dim=-1)
        
        # Gumbel-Softmax for differentiable discrete selection
        op_probs = F.gumbel_softmax(op_logits, tau=_GLOBAL_TAU, hard=_GLOBAL_HARD, dim=-1)
        
        # Calculate entropy for regularization (encourages diverse operation use)
        entropy = -(op_probs_soft * torch.log(op_probs_soft + 1e-10)).sum(dim=-1).mean()
        
        n_unary = len(UNARY_NAMES)
        n_binary = len(BINARY_NAMES) if self.use_binary_ops else 0
        
        # Compute all unary operations
        unary_outputs = []
        for op_name in UNARY_NAMES:
            op_fn = UNARY_OPS[op_name]
            out = op_fn(x_proj, self.constants)
            unary_outputs.append(out)
        all_outputs = torch.stack(unary_outputs, dim=-1)  # (batch, hidden, n_unary)
        
        # Compute binary operations if enabled
        if self.use_binary_ops:
            if h is not None:
                h_proj = self.hidden_proj(h)
            else:
                h_proj = torch.zeros_like(x_proj)
            
            binary_outputs = []
            for op_name in BINARY_NAMES:
                op_fn = BINARY_OPS[op_name]
                out = op_fn(x_proj, h_proj)
                binary_outputs.append(out)
            binary_stack = torch.stack(binary_outputs, dim=-1)
            all_outputs = torch.cat([all_outputs, binary_stack], dim=-1)
        
        # Compute aggregation operations if enabled
        if self.use_aggregation_ops:
            # Combine x and h for aggregation input
            if h is not None:
                combined = torch.cat([x, h], dim=-1)
            else:
                combined = torch.cat([x, torch.zeros(batch_size, self.hidden_size, device=device)], dim=-1)
            
            # Project to multiple inputs for aggregation
            agg_inputs = self.agg_proj(combined)  # (batch, hidden * n_agg_inputs)
            agg_inputs = agg_inputs.view(batch_size, self.hidden_size, self.n_aggregation_inputs)
            
            # Apply edge weights
            weighted_inputs = agg_inputs * self.edge_weights.unsqueeze(0)  # (batch, hidden, n_agg_inputs)
            
            # Compute each aggregation operation
            agg_outputs = []
            for op_name in AGGREGATION_NAMES:
                op_fn = AGGREGATION_OPS[op_name]
                out = op_fn(weighted_inputs)  # (batch, hidden)
                agg_outputs.append(out)
            agg_stack = torch.stack(agg_outputs, dim=-1)  # (batch, hidden, n_agg)
            all_outputs = torch.cat([all_outputs, agg_stack], dim=-1)
        
        # Weighted sum by operation probabilities
        output = (all_outputs * op_probs).sum(dim=-1)  # (batch, hidden_size)
        
        # Clamp to prevent explosion
        output = torch.clamp(output, CLAMP_MIN, CLAMP_MAX)
        
        return output, entropy
    
    def get_selected_ops(self) -> List[str]:
        """Get the name of the selected operation for each hidden unit."""
        all_ops = UNARY_NAMES.copy()
        if self.use_binary_ops:
            all_ops += BINARY_NAMES
        if self.use_aggregation_ops:
            all_ops += AGGREGATION_NAMES
        
        indices = self.op_weights.argmax(dim=-1).tolist()
        return [all_ops[i] if i < len(all_ops) else "unknown" for i in indices]
    
    def get_op_distribution(self) -> Dict[str, float]:
        """Get the distribution of selected operations across hidden units."""
        ops = self.get_selected_ops()
        counts = {}
        for op in ops:
            counts[op] = counts.get(op, 0) + 1
        # Normalize
        total = len(ops)
        return {k: v/total for k, v in sorted(counts.items(), key=lambda x: -x[1])}


# =============================================================================
# Operation RNN v2: With entropy regularization
# =============================================================================

class OperationRNN(nn.Module):
    """
    Recurrent Neural Network where each recurrent step applies learned operations.
    
    Architecture:
        h_t = OperationCell(x_t, h_{t-1})
        
    The network learns:
    1. Which operation to apply at each step
    2. Constants used by operations (like +c, *c)
    3. Edge weights for aggregation operations
    
    Features:
    - Entropy regularization encourages diverse operation exploration
    - Multi-input aggregation for binary-style reduce operations
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        output_size: Optional[int] = None,
        use_aggregation: bool = True,
        n_aggregation_inputs: int = 4,
    ):
        """
        Args:
            input_size: Size of input features at each timestep
            hidden_size: Size of hidden state
            num_layers: Number of stacked operation cells
            output_size: Size of output (if None, equals hidden_size)
            use_aggregation: Enable multi-input aggregation operations
            n_aggregation_inputs: Number of inputs for aggregation
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size or hidden_size
        
        # Stack of operation cells
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell_input_size = input_size if i == 0 else hidden_size
            self.cells.append(OperationCell(
                cell_input_size, hidden_size,
                use_aggregation_ops=use_aggregation,
                n_aggregation_inputs=n_aggregation_inputs
            ))
        
        # Output projection
        if self.output_size != hidden_size:
            self.output_proj = nn.Linear(hidden_size, self.output_size)
        else:
            self.output_proj = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        return_entropy: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the Operation RNN.
        
        Args:
            x: Input sequence (batch, seq_len, input_size)
            h0: Initial hidden states (num_layers, batch, hidden_size)
            return_entropy: Whether to return entropy for regularization
            
        Returns:
            outputs: Output sequence (batch, seq_len, output_size)
            h_n: Final hidden states (num_layers, batch, hidden_size)
            entropy: Mean entropy across all cells (for loss regularization)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device, dtype=x.dtype)
        
        h = [h0[i] for i in range(self.num_layers)]
        outputs = []
        total_entropy = 0.0
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            layer_input = x_t
            for i, cell in enumerate(self.cells):
                h[i], entropy = cell(layer_input, h[i])
                total_entropy += entropy
                layer_input = h[i]
            
            out = self.output_proj(h[-1])
            outputs.append(out)
        
        outputs = torch.stack(outputs, dim=1)
        h_n = torch.stack(h, dim=0)
        
        # Average entropy across all timesteps and layers
        avg_entropy = total_entropy / (seq_len * self.num_layers)
        
        if return_entropy:
            return outputs, h_n, avg_entropy
        return outputs, h_n, None
    
    def get_formula_description(self) -> str:
        """Get human-readable description of learned operations."""
        desc = []
        for i, cell in enumerate(self.cells):
            ops = cell.get_selected_ops()
            op_dist = cell.get_op_distribution()
            consts = cell.constants.data.tolist()
            desc.append(f"Layer {i}:")
            desc.append(f"  Operations: {ops[:8]}...")
            desc.append(f"  Distribution: {dict(list(op_dist.items())[:5])}")
            desc.append(f"  Constants: {[f'{c:.2f}' for c in consts[:5]]}...")
        return "\n".join(desc)
    
    def get_total_op_distribution(self) -> Dict[str, float]:
        """Get operation distribution across all layers."""
        all_ops = []
        for cell in self.cells:
            all_ops.extend(cell.get_selected_ops())
        
        counts = {}
        for op in all_ops:
            counts[op] = counts.get(op, 0) + 1
        total = len(all_ops)
        return {k: v/total for k, v in sorted(counts.items(), key=lambda x: -x[1])}


# =============================================================================
# Operation LSTM v2: With entropy regularization
# =============================================================================

class OperationLSTMCell(nn.Module):
    """
    LSTM-like cell where gates are operation-based instead of weight-based.
    
    Each gate (forget, input, candidate, output) uses an OperationCell
    to learn which mathematical operation to apply.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_aggregation: bool = True,
        n_aggregation_inputs: int = 3,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        combined_size = input_size + hidden_size
        
        # Four operation cells for LSTM gates
        self.forget_cell = OperationCell(
            combined_size, hidden_size,
            use_binary_ops=False,
            use_aggregation_ops=use_aggregation,
            n_aggregation_inputs=n_aggregation_inputs
        )
        self.input_cell = OperationCell(
            combined_size, hidden_size,
            use_binary_ops=False,
            use_aggregation_ops=use_aggregation,
            n_aggregation_inputs=n_aggregation_inputs
        )
        self.candidate_cell = OperationCell(
            combined_size, hidden_size,
            use_binary_ops=False,
            use_aggregation_ops=use_aggregation,
            n_aggregation_inputs=n_aggregation_inputs
        )
        self.output_cell = OperationCell(
            combined_size, hidden_size,
            use_binary_ops=False,
            use_aggregation_ops=use_aggregation,
            n_aggregation_inputs=n_aggregation_inputs
        )
        
        # Learnable gate combination (defaults favor mul)
        self.gate_combine_weights = nn.Parameter(torch.tensor([0., 0., 1., 0., 0., 0., 0.]))
    
    def _gate_combine(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Combine two tensors using learned binary operation."""
        probs = F.softmax(self.gate_combine_weights, dim=0)
        
        outputs = []
        for op_name in BINARY_NAMES:
            op_fn = BINARY_OPS[op_name]
            outputs.append(op_fn(a, b))
        
        stacked = torch.stack(outputs, dim=-1)
        combined = (stacked * probs).sum(dim=-1)
        
        return torch.clamp(combined, CLAMP_MIN, CLAMP_MAX)
    
    def forward(
        self,
        x: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass through Operation LSTM cell.
        
        Returns:
            h_new: New hidden state
            (h_new, c_new): New states tuple
            entropy: Combined entropy from all gates
        """
        batch_size = x.shape[0]
        device = x.device
        
        if states is None:
            h = torch.zeros(batch_size, self.hidden_size, device=device, dtype=x.dtype)
            c = torch.zeros(batch_size, self.hidden_size, device=device, dtype=x.dtype)
        else:
            h, c = states
        
        combined = torch.cat([x, h], dim=-1)
        
        # Compute gate operations with entropy
        f_raw, e1 = self.forget_cell(combined)
        i_raw, e2 = self.input_cell(combined)
        g_raw, e3 = self.candidate_cell(combined)
        o_raw, e4 = self.output_cell(combined)
        
        total_entropy = (e1 + e2 + e3 + e4) / 4
        
        # Apply gate activations
        f = torch.sigmoid(f_raw)
        i = torch.sigmoid(i_raw)
        g = torch.tanh(g_raw)
        o = torch.sigmoid(o_raw)
        
        # Update cell state
        c_new = self._gate_combine(f, c) + self._gate_combine(i, g)
        c_new = torch.clamp(c_new, CLAMP_MIN, CLAMP_MAX)
        
        # Update hidden state
        h_new = self._gate_combine(o, torch.tanh(c_new))
        
        return h_new, (h_new, c_new), total_entropy
    
    def get_gate_ops(self) -> Dict[str, List[str]]:
        """Get selected operations for each gate."""
        return {
            'forget': self.forget_cell.get_selected_ops()[:4],
            'input': self.input_cell.get_selected_ops()[:4],
            'candidate': self.candidate_cell.get_selected_ops()[:4],
            'output': self.output_cell.get_selected_ops()[:4],
        }
    
    def get_gate_distributions(self) -> Dict[str, Dict[str, float]]:
        """Get operation distributions for each gate."""
        return {
            'forget': self.forget_cell.get_op_distribution(),
            'input': self.input_cell.get_op_distribution(),
            'candidate': self.candidate_cell.get_op_distribution(),
            'output': self.output_cell.get_op_distribution(),
        }


class OperationLSTM(nn.Module):
    """
    Full Operation LSTM network with stacked layers and entropy regularization.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        output_size: Optional[int] = None,
        use_aggregation: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size or hidden_size
        
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell_input = input_size if i == 0 else hidden_size
            self.cells.append(OperationLSTMCell(cell_input, hidden_size, use_aggregation=use_aggregation))
        
        if self.output_size != hidden_size:
            self.output_proj = nn.Linear(hidden_size, self.output_size)
        else:
            self.output_proj = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_entropy: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through Operation LSTM.
        
        Returns:
            outputs: (batch, seq_len, output_size)
            (h_n, c_n): Final states
            entropy: Mean entropy for regularization
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        if states is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=device, dtype=x.dtype)
                 for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, device=device, dtype=x.dtype)
                 for _ in range(self.num_layers)]
        else:
            h = [states[0][i] for i in range(self.num_layers)]
            c = [states[1][i] for i in range(self.num_layers)]
        
        outputs = []
        total_entropy = 0.0
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            layer_input = x_t
            for i, cell in enumerate(self.cells):
                h[i], (h[i], c[i]), entropy = cell(layer_input, (h[i], c[i]))
                total_entropy += entropy
                layer_input = h[i]
            
            out = self.output_proj(h[-1])
            outputs.append(out)
        
        outputs = torch.stack(outputs, dim=1)
        h_n = torch.stack(h, dim=0)
        c_n = torch.stack(c, dim=0)
        
        avg_entropy = total_entropy / (seq_len * self.num_layers)
        
        if return_entropy:
            return outputs, (h_n, c_n), avg_entropy
        return outputs, (h_n, c_n), None
    
    def get_formula_description(self) -> str:
        """Get description of learned operations per layer and gate."""
        desc = []
        for i, cell in enumerate(self.cells):
            gate_ops = cell.get_gate_ops()
            desc.append(f"Layer {i}:")
            for gate, ops in gate_ops.items():
                desc.append(f"  {gate}: {ops}")
        return "\n".join(desc)


# =============================================================================
# Utility Functions
# =============================================================================

def anneal_temperature(
    epoch: int,
    total_epochs: int,
    start_tau: float = 2.0,
    end_tau: float = 0.1,
) -> float:
    """
    Anneal Gumbel-Softmax temperature from start to end over training.
    Uses cosine annealing for smoother transition.
    """
    progress = epoch / max(total_epochs - 1, 1)
    # Cosine annealing: slower at start and end, faster in middle
    tau = end_tau + 0.5 * (start_tau - end_tau) * (1 + math.cos(math.pi * progress))
    set_global_temperature(tau)
    return tau


def compute_regularized_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    entropy: torch.Tensor,
    mse_criterion: nn.MSELoss,
    entropy_weight: float = 0.1,
    target_entropy: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute loss with entropy regularization.
    
    Encourages the model to explore diverse operations by penalizing
    low entropy (i.e., when it always picks the same operation).
    
    Args:
        pred: Predicted values
        target: Target values
        entropy: Entropy from operation selection
        mse_criterion: MSE loss function
        entropy_weight: Weight for entropy term
        target_entropy: Target entropy level (higher = more exploration)
        
    Returns:
        total_loss: Combined loss
        mse_loss: Raw MSE component
        entropy_loss: Entropy regularization component
    """
    mse_loss = mse_criterion(pred, target)
    
    # Penalize entropy below target (encourage exploration)
    entropy_loss = F.relu(target_entropy - entropy)
    
    total_loss = mse_loss + entropy_weight * entropy_loss
    
    return total_loss, mse_loss, entropy_loss
