"""
Evolutionary Training for Operation-Based Neural Networks.

The PROPER approach: Evolution for discrete structure, gradient for constants.

Key insight from research:
- Gradient descent CANNOT effectively select discrete operations
- Evolution is better for structural search (which operation)
- Gradient descent is better for continuous optimization (what parameters)

This implements the "EmbedGrad-Evolution" strategy:
1. Population of individuals with different operation structures
2. Fitness = actual prediction error (not training loss)
3. Discrete mutations on operation selection
4. Gradient refinement on constants only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, LBFGS
from typing import Optional, Dict, List, Tuple, Callable, Any
import copy
import random
import math
import time
import numpy as np


def set_model_tau(model: nn.Module, tau: float):
    """
    Set temperature (tau) on all selectors in the model.
    
    Lower tau = more discrete selection (sharper softmax).
    """
    for module in model.modules():
        if hasattr(module, 'set_tau') and callable(getattr(module, 'set_tau')):
            module.set_tau(tau)
        elif hasattr(module, 'tau') and not isinstance(getattr(module, 'tau'), nn.Parameter):
            module.tau = tau


def anneal_tau(generation: int, total_generations: int, 
               tau_start: float = 1.0, tau_end: float = 0.2) -> float:
    """
    Cosine annealing for temperature.
    
    Args:
        generation: Current generation
        total_generations: Total number of generations
        tau_start: Starting temperature (high = soft selection)
        tau_end: Ending temperature (low = hard selection)
    
    Returns:
        Annealed tau value
    """
    progress = generation / max(total_generations - 1, 1)
    # Cosine annealing
    tau = tau_end + 0.5 * (tau_start - tau_end) * (1 + math.cos(math.pi * progress))
    return max(tau, tau_end)  # Floor at tau_end


def anneal_entropy_weight(generation: int, total_generations: int,
                         start_weight: float = 0.001, end_weight: float = 0.1) -> float:
    """
    Entropy annealing schedule: mild early, strong late.
    
    Research insight: Start with weak entropy penalty (allow exploration),
    gradually strengthen to force discrete decisions.
    
    Uses exponential schedule for sharper increase near end.
    
    Args:
        generation: Current generation
        total_generations: Total generations
        start_weight: Initial entropy weight (small = exploration)
        end_weight: Final entropy weight (large = force discrete)
    
    Returns:
        Annealed entropy weight
    """
    progress = generation / max(total_generations - 1, 1)
    # Exponential increase (slow start, fast end)
    # w = start * (end/start)^progress
    ratio = end_weight / max(start_weight, 1e-8)
    weight = start_weight * (ratio ** progress)
    return weight


def soft_round(x: torch.Tensor, sharpness: float = 5.0) -> torch.Tensor:
    """
    Differentiable soft rounding that nudges values toward integers.
    
    Instead of hard snap (p=1.97 -> 2.0), this creates a smooth
    potential well around each integer. As sharpness increases,
    approaches hard rounding.
    
    Uses: round_soft(x) = x - sin(2πx) / (2π * sharpness)
    
    Args:
        x: Input tensor
        sharpness: How strongly to push toward integers (higher = sharper)
    
    Returns:
        Soft-rounded tensor (differentiable)
    """
    # Periodic function that is 0 at integers, pushes toward them elsewhere
    correction = torch.sin(2 * math.pi * x) / (2 * math.pi * sharpness)
    return x - correction


def progressive_round_loss(model: nn.Module, target_params: List[str] = None) -> torch.Tensor:
    """
    Loss that encourages continuous parameters to approach discrete values.
    
    Instead of hard snapping (which destroys performance), this adds
    a soft penalty for being away from integers/nice values.
    
    Targets: exponents (p), frequencies (omega), blend parameters
    
    Args:
        model: The model
        target_params: Parameter name patterns to target (default: ['p', 'omega'])
    
    Returns:
        Loss encouraging discrete-ish values
    """
    if target_params is None:
        target_params = ['.p', '.omega']
    
    loss = torch.tensor(0.0)
    count = 0
    
    for name, param in model.named_parameters():
        if any(t in name for t in target_params):
            if 'output' in name or 'proj' in name:
                continue
            # Penalty for distance from nearest integer
            # min(x - floor(x), ceil(x) - x)^2
            frac = param - param.floor()
            # Distance to nearest integer: min(frac, 1-frac)
            dist = torch.minimum(frac, 1 - frac)
            loss = loss + (dist ** 2).mean()
            count += 1
    
    return loss / max(count, 1)


class Individual:
    """
    An individual in the evolutionary population.
    
    Contains a model with specific operation configuration.
    """
    
    def __init__(
        self,
        model: nn.Module,
        fitness: float = float('inf'),
        generation: int = 0,
        is_explorer: bool = False,
    ):
        self.model = model
        self.fitness = fitness
        self.generation = generation
        self.is_explorer = is_explorer  # Explorer subpopulation flag
        self.structure_hash = self._compute_structure_hash()
    
    def _compute_structure_hash(self) -> str:
        """Hash the operation selection for diversity tracking."""
        ops = []
        for name, param in self.model.named_parameters():
            if 'logit' in name or 'selector' in name:
                ops.append(param.argmax().item())
        return str(ops)
    
    def clone(self) -> 'Individual':
        """Deep copy this individual."""
        new_model = copy.deepcopy(self.model)
        ind = Individual(
            model=new_model,
            fitness=self.fitness,
            generation=self.generation + 1,
            is_explorer=self.is_explorer,
        )
        return ind


def random_operation_init(model: nn.Module, bias_strength: float = 2.0):
    """
    Initialize with RANDOM operation selections.
    
    Unlike identity init, this explores the full operation space.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'logit' in name or 'selector' in name:
                # Strong random bias toward one operation
                param.normal_(0, bias_strength)
            elif 'p' in name and 'output' not in name:
                # Random power: 0.5 to 3.0
                param.fill_(random.uniform(0.5, 3.0))
            elif 'omega' in name:
                # Random frequency: 0.5 to 2.0
                param.fill_(random.uniform(0.5, 2.0))
            elif 'beta' in name and 'selector' not in name:
                # Random arithmetic blend
                param.fill_(random.uniform(0.5, 2.5))
            elif 'R' in name:
                # Random routing
                param.normal_(0, 1.0)


def mutate_operations(individual: Individual, mutation_rate: float = 0.3) -> Individual:
    """
    Mutate operation selections DISCRETELY.
    
    This is the key difference from gradient-based training:
    - We randomly change which operation is selected
    - Not small continuous nudges, but discrete swaps
    """
    mutant = individual.clone()
    
    with torch.no_grad():
        for name, param in mutant.model.named_parameters():
            if random.random() < mutation_rate:
                if 'logit' in name or 'selector' in name:
                    # DISCRETE mutation: completely re-roll the selection
                    if random.random() < 0.5:
                        # Option 1: Shift to a different option
                        current_best = param.argmax()
                        param.zero_()
                        new_choice = random.randint(0, param.numel() - 1)
                        param.view(-1)[new_choice] = 3.0  # Strong bias to new choice
                    else:
                        # Option 2: Add noise to current selection
                        param.add_(torch.randn_like(param) * 1.0)
                
                elif 'p' in name and 'output' not in name:
                    # Mutate power parameter
                    if random.random() < 0.3:
                        # Discrete jump to common powers
                        param.fill_(random.choice([0.5, 1.0, 2.0, 3.0, -1.0]))
                    else:
                        param.add_(random.uniform(-0.5, 0.5))
                        param.clamp_(0.1, 4.0)
                
                elif 'omega' in name:
                    # Mutate frequency
                    param.mul_(random.uniform(0.8, 1.2))
                    param.clamp_(0.1, 5.0)
                
                elif 'R' in name:
                    # Mutate routing
                    if random.random() < 0.3:
                        # Swap routing connections
                        param.add_(torch.randn_like(param) * 0.5)
    
    return mutant


def mutate_operations_lamarckian(individual: Individual, mutation_rate: float = 0.3) -> Individual:
    """
    Lamarckian mutation: only mutate discrete structure, preserve continuous weights.
    
    Key insight from research (Prellberg & Kramer 2018):
    - Passing optimized weights from parent to child dramatically speeds convergence
    - We only mutate OPERATION SELECTION (discrete), not the parameters
    - Continuous parameters (p, omega, coefficients) are inherited as-is
    - Then we fine-tune with gradient descent
    
    This is "Lamarckian" because learned traits (optimized weights) are inherited.
    """
    mutant = individual.clone()
    
    with torch.no_grad():
        for name, param in mutant.model.named_parameters():
            # ONLY mutate operation selection logits
            # Leave all continuous parameters (p, omega, R, etc.) UNCHANGED
            if 'logit' in name or 'selector' in name:
                if random.random() < mutation_rate:
                    if random.random() < 0.5:
                        # Option 1: Shift to a different operation
                        param.zero_()
                        new_choice = random.randint(0, param.numel() - 1)
                        param.view(-1)[new_choice] = 3.0
                    else:
                        # Option 2: Add noise to soften selection
                        param.add_(torch.randn_like(param) * 0.5)
    
    return mutant


def compute_param_sensitivity(
    model: nn.Module, 
    x: torch.Tensor, 
    y: torch.Tensor
) -> Dict[str, float]:
    """
    Compute gradient-based sensitivity for each parameter.
    
    Tier 2 feature: Use gradients to identify which operations/parameters
    have the most impact on the loss. This can guide mutation toward
    more impactful parts of the model.
    
    Returns:
        Dict mapping parameter names to their sensitivity (|gradient|)
    """
    model.train()
    model.zero_grad()
    
    try:
        pred, _ = model(x, hard=False)  # Soft forward for gradients
        loss = F.mse_loss(pred.squeeze(), y.squeeze())
        loss.backward()
    except Exception:
        return {}
    
    sensitivities = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            sensitivities[name] = param.grad.abs().mean().item()
        else:
            sensitivities[name] = 0.0
    
    model.zero_grad()
    return sensitivities


def mutate_operations_gradient_informed(
    individual: Individual, 
    x: torch.Tensor, 
    y: torch.Tensor,
    mutation_rate: float = 0.3,
    sensitivity_bias: float = 0.5,
) -> Individual:
    """
    Gradient-informed mutation: bias mutations toward less sensitive parameters.
    
    Tier 2 feature: Use gradient information to guide which operations to mutate.
    - Low-sensitivity operations are more likely to be mutated (they're not important)
    - High-sensitivity operations are preserved (they're working)
    
    Args:
        individual: Parent individual
        x, y: Data for computing gradients
        mutation_rate: Base mutation probability
        sensitivity_bias: How much to bias by sensitivity (0=ignore, 1=strong bias)
    
    Returns:
        Mutated individual
    """
    # Compute sensitivities
    sensitivities = compute_param_sensitivity(individual.model, x, y)
    
    if not sensitivities:
        # Fallback to regular mutation if gradient computation fails
        return mutate_operations_lamarckian(individual, mutation_rate)
    
    # Normalize sensitivities to [0, 1]
    max_sens = max(sensitivities.values()) if sensitivities.values() else 1.0
    if max_sens > 0:
        norm_sens = {k: v / max_sens for k, v in sensitivities.items()}
    else:
        norm_sens = sensitivities
    
    mutant = individual.clone()
    
    with torch.no_grad():
        for name, param in mutant.model.named_parameters():
            if 'logit' in name or 'selector' in name:
                # Compute adjusted mutation rate
                # Low sensitivity = higher mutation rate (not important)
                # High sensitivity = lower mutation rate (important)
                sens = norm_sens.get(name, 0.5)
                adjusted_rate = mutation_rate * (1.0 + sensitivity_bias * (1.0 - sens))
                adjusted_rate = min(adjusted_rate, 1.0)
                
                if random.random() < adjusted_rate:
                    if random.random() < 0.5:
                        param.zero_()
                        new_choice = random.randint(0, param.numel() - 1)
                        param.view(-1)[new_choice] = 3.0
                    else:
                        param.add_(torch.randn_like(param) * 0.5)
    
    return mutant


def refine_constants(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int = 50,
    lr: float = 0.01,
    use_lbfgs: bool = False,
    scales_only: bool = False,
    hard: bool = True,
) -> float:
    """
    Use gradient descent to optimize ONLY the constants.
    
    Lock operation selection, only tune parameters.
    
    Args:
        model: The model to refine
        x: Input data
        y: Target data
        steps: Number of optimization steps
        lr: Learning rate (for Adam)
        use_lbfgs: If True, use L-BFGS (better for symbolic regression constants)
        scales_only: If True, only tune output_scale and edge_weights (use after snap)
    """
    import gc
    
    model.train()
    
    # Identify constant parameters (not selection logits)
    constant_params = []
    param_names = []  # DEBUG
    for name, param in model.named_parameters():
        # Skip selection logits
        if 'logit' in name or 'selector' in name:
            continue
        
        # If scales_only, only include output_scale, edge_weights, output_proj
        if scales_only:
            if 'output_scale' in name or 'edge_weights' in name or 'output_proj' in name:
                if param.requires_grad:
                    constant_params.append(param)
                    param_names.append(name)
        else:
            # Skip meta-op core parameters (p, omega, phi, beta) - these are snapped
            if any(x in name for x in ['.p', '.omega', '.phi', '.beta', 'amplitude']):
                continue
            if param.requires_grad:
                constant_params.append(param)
                param_names.append(name)
    
    # DEBUG: Print what we're optimizing
    # print(f"  [refine_constants] Optimizing {len(constant_params)} params: {param_names[:5]}...")
    
    if not constant_params:
        return float('inf')
    
    best_loss = float('inf')
    
    try:
        if use_lbfgs:
            # L-BFGS: Better for finding exact constants (recommended by research)
            y_squeezed = y.squeeze()
            def closure():
                nonlocal best_loss
                optimizer.zero_grad()
                pred, _ = model(x, hard=hard)
                pred = pred.squeeze()
                loss = F.mse_loss(pred, y_squeezed)
                if torch.isnan(loss):
                    return torch.tensor(float('inf'))
                loss.backward()
                best_loss = min(best_loss, loss.item())
                return loss
            
            optimizer = LBFGS(constant_params, lr=1.0, max_iter=steps, line_search_fn='strong_wolfe')
            try:
                optimizer.step(closure)
            except Exception:
                pass  # L-BFGS can fail, that's ok
        else:
            # Adam: Faster but less precise
            optimizer = Adam(constant_params, lr=lr)
            y_squeezed = y.squeeze()
            
            for step in range(steps):
                optimizer.zero_grad()
                
                pred, _ = model(x, hard=hard)
                pred = pred.squeeze()
                loss = F.mse_loss(pred, y_squeezed)
                
                if torch.isnan(loss):
                    del optimizer
                    gc.collect()
                    return float('inf')
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(constant_params, max_norm=1.0)
                optimizer.step()
                
                best_loss = min(best_loss, loss.item())
    except (MemoryError, RuntimeError) as e:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return float('inf')
    except Exception:
        gc.collect()
        return float('inf')
    
    # Cleanup
    del optimizer
    gc.collect()
    
    return best_loss


def calculate_complexity(model: nn.Module) -> float:
    """
    Calculate complexity score for a model (BIC-inspired).
    Higher score = more complex operations.
    
    Based on research recommendation: penalize formula length/operation count
    to guide search toward simpler expressions.
    """
    complexity = 0.0
    n_active_ops = 0
    
    if hasattr(model, 'layers'):
        for layer in model.layers:
            for node in layer.nodes:
                op_str = node.get_selected_operation().lower()
                n_active_ops += 1
                
                # High cost: Transcendental functions (exp, log, trig)
                # These are often incorrectly selected for polynomial targets
                if 'exp' in op_str:
                    complexity += 3.0  # Increased penalty - often wrong choice
                elif 'log' in op_str or 'ln' in op_str:
                    complexity += 2.5
                elif 'sin' in op_str or 'cos' in op_str:
                    complexity += 2.0
                # Medium cost: Powers and roots
                elif 'sqrt' in op_str or 'pow' in op_str or '^' in op_str:
                    complexity += 1.0  # Reduced - these are often correct
                # Low cost: Arithmetic and identity
                elif 'identity' in op_str:
                    complexity += 0.1
                elif 'add' in op_str or 'mul' in op_str or '+' in op_str or '*' in op_str:
                    complexity += 0.5
                # Aggregation - medium-high (often noise)
                elif 'agg' in op_str:
                    complexity += 2.0
                else:
                    complexity += 1.0
    
    # BIC-style penalty: log(n) * k where k = number of operations
    # This encourages simpler formulas
    bic_penalty = 0.5 * n_active_ops
    
    return complexity + bic_penalty


def prune_small_coefficients(
    model: nn.Module, 
    threshold_ratio: float = 0.3,
    absolute_threshold: float = 0.1,
) -> int:
    """
    Prune operations with small output coefficients.
    
    Key insight: If formula is "1.37*x² + 0.74*√(x) + 0.63*exp(x)",
    the x² term dominates (1.37). Terms with coefficients < 0.3*1.37 = 0.41
    are likely noise and can be pruned.
    
    This speeds up convergence by removing spurious operations early.
    
    Args:
        model: The ONN model
        threshold_ratio: Prune terms with |coef| < ratio * max(|coefs|)
        absolute_threshold: Also prune if |coef| < this absolute value
    
    Returns:
        Number of operations pruned
    """
    if not hasattr(model, 'output_proj'):
        return 0
    
    with torch.no_grad():
        weights = model.output_proj.weight.data  # (1, n_features)
        
        if weights.numel() == 0:
            return 0
        
        abs_weights = weights.abs().squeeze()
        if abs_weights.dim() == 0:
            return 0
            
        max_weight = abs_weights.max().item()
        
        if max_weight < 1e-6:
            return 0
        
        # Compute threshold: relative to max OR absolute minimum
        threshold = max(threshold_ratio * max_weight, absolute_threshold)
        
        # Find small coefficients
        small_mask = abs_weights < threshold
        n_pruned = small_mask.sum().item()
        
        # Zero out small coefficients
        if weights.dim() == 2:
            weights[0, small_mask] = 0.0
        else:
            weights[small_mask] = 0.0
    
    return int(n_pruned)


def adaptive_coefficient_pruning(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    prune_ratio: float = 0.3,
) -> Tuple[int, float]:
    """
    Intelligently prune coefficients by measuring their contribution to loss.
    
    For each output coefficient:
    1. Temporarily zero it out
    2. Measure increase in MSE
    3. If MSE increase is small, prune permanently
    
    This is more accurate than just looking at coefficient magnitude.
    
    Args:
        model: The ONN model
        x, y: Data for evaluating contribution
        prune_ratio: Prune if MSE increase < ratio * current_MSE
    
    Returns:
        (n_pruned, new_mse)
    """
    if not hasattr(model, 'output_proj'):
        return 0, float('inf')
    
    model.eval()
    weights = model.output_proj.weight.data
    
    if weights.numel() == 0:
        return 0, float('inf')
    
    with torch.no_grad():
        # Baseline MSE
        pred, _ = model(x, hard=True)
        base_mse = F.mse_loss(pred.squeeze(), y.squeeze()).item()
        
        if math.isnan(base_mse) or math.isinf(base_mse):
            return 0, float('inf')
        
        n_features = weights.shape[1] if weights.dim() == 2 else weights.shape[0]
        pruned_indices = []
        
        for i in range(n_features):
            # Save original weight
            if weights.dim() == 2:
                orig_weight = weights[0, i].clone()
                weights[0, i] = 0.0
            else:
                orig_weight = weights[i].clone()
                weights[i] = 0.0
            
            # Measure MSE without this coefficient
            pred, _ = model(x, hard=True)
            new_mse = F.mse_loss(pred.squeeze(), y.squeeze()).item()
            
            # Restore weight
            if weights.dim() == 2:
                weights[0, i] = orig_weight
            else:
                weights[i] = orig_weight
            
            # If MSE increase is small, mark for pruning
            mse_increase = new_mse - base_mse
            if mse_increase < prune_ratio * base_mse:
                pruned_indices.append(i)
        
        # Prune marked coefficients
        for i in pruned_indices:
            if weights.dim() == 2:
                weights[0, i] = 0.0
            else:
                weights[i] = 0.0
        
        # Final MSE
        pred, _ = model(x, hard=True)
        final_mse = F.mse_loss(pred.squeeze(), y.squeeze()).item()
    
    return len(pruned_indices), final_mse


def check_structure_quality(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    corr_threshold: float = 0.99,
) -> Tuple[bool, float, float]:
    """
    Check if the model has found the correct STRUCTURE (high correlation)
    even if coefficients are wrong (imperfect MSE).
    
    High correlation + imperfect MSE = right structure, wrong coefficients
    
    Args:
        model: The ONN model
        x, y: Data
        corr_threshold: Correlation threshold to consider structure "locked"
    
    Returns:
        (structure_is_good, correlation, mse)
    """
    model.eval()
    with torch.no_grad():
        pred, _ = model(x, hard=True)
        pred = pred.squeeze()
        y_sq = y.squeeze()
        
        mse = F.mse_loss(pred, y_sq).item()
        
        if math.isnan(mse) or math.isinf(mse):
            return False, 0.0, float('inf')
        
        # Compute correlation
        corr_matrix = torch.corrcoef(torch.stack([pred.cpu(), y_sq.cpu()]))
        corr = corr_matrix[0, 1].item()
        
        if math.isnan(corr):
            corr = 0.0
        
        structure_is_good = abs(corr) >= corr_threshold
        
        return structure_is_good, corr, mse


def intensive_coefficient_refinement(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    max_steps: int = 1000,
    target_mse: float = 0.001,
    lr_start: float = 0.01,
    patience: int = 200,
) -> Tuple[float, int]:
    """
    Two-phase coefficient refinement when structure is locked.
    
    Phase 1: Tune output_proj only (safe, won't break structure)
    Phase 2: Very gently tune scale parameters (risky, may break structure)
    
    Returns:
        (final_mse, steps_taken)
    """
    model.eval()
    y_sq = y.squeeze()
    with torch.no_grad():
        pred, _ = model(x, hard=True)
        initial_mse = F.mse_loss(pred.squeeze(), y_sq).item()
    
    if initial_mse < target_mse:
        return initial_mse, 0
    
    total_steps = 0
    best_overall_mse = initial_mse
    best_overall_state = {name: param.detach().clone() for name, param in model.named_parameters()}
    
    # ===== PHASE 1: Output projection only (safe) =====
    output_proj_params = [p for n, p in model.named_parameters() if 'output_proj' in n]
    if output_proj_params:
        model.train()
        optimizer = Adam(output_proj_params, lr=0.05)  # Higher LR ok for output_proj
        
        phase1_best_mse = initial_mse
        phase1_best_state = {n: p.detach().clone() for n, p in model.named_parameters() if 'output_proj' in n}
        no_improve = 0
        
        for step in range(max_steps // 2):
            total_steps += 1
            optimizer.zero_grad()
            pred, _ = model(x, hard=True)
            loss = F.mse_loss(pred.squeeze(), y_sq)
            
            if torch.isnan(loss):
                break
            
            mse = loss.item()
            if mse < phase1_best_mse - 1e-8:
                phase1_best_mse = mse
                phase1_best_state = {n: p.detach().clone() for n, p in model.named_parameters() if 'output_proj' in n}
                no_improve = 0
            else:
                no_improve += 1
            
            if mse < target_mse or no_improve > patience // 2:
                break
            
            loss.backward()
            optimizer.step()
        
        # Restore phase 1 best
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in phase1_best_state:
                    param.copy_(phase1_best_state[name])
        
        if phase1_best_mse < best_overall_mse:
            best_overall_mse = phase1_best_mse
            best_overall_state = {name: param.detach().clone() for name, param in model.named_parameters()}
    
    # ===== PHASE 2: Scale params with TINY LR (risky) =====
    model.eval()
    with torch.no_grad():
        pred, _ = model(x, hard=True)
        current_mse = F.mse_loss(pred.squeeze(), y_sq).item()
    
    if current_mse > target_mse:
        # Only tune scales if we haven't reached target
        scale_params = [p for n, p in model.named_parameters() if 'scale' in n and 'alpha' not in n]
        if scale_params:
            model.train()
            optimizer = Adam(scale_params, lr=0.001)  # Very tiny LR
            
            phase2_best_mse = current_mse
            phase2_best_state = {n: p.detach().clone() for n, p in model.named_parameters() if 'scale' in n}
            no_improve = 0
            
            for step in range(max_steps // 2):
                total_steps += 1
                optimizer.zero_grad()
                pred, _ = model(x, hard=True)
                loss = F.mse_loss(pred.squeeze(), y_sq)
                
                if torch.isnan(loss):
                    break
                
                mse = loss.item()
                if mse < phase2_best_mse - 1e-8:
                    phase2_best_mse = mse
                    phase2_best_state = {n: p.detach().clone() for n, p in model.named_parameters() if 'scale' in n}
                    no_improve = 0
                else:
                    no_improve += 1
                
                if mse < target_mse or no_improve > patience // 2:
                    break
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(scale_params, max_norm=0.1)
                optimizer.step()
            
            # Restore phase 2 best
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in phase2_best_state:
                        param.copy_(phase2_best_state[name])
            
            if phase2_best_mse < best_overall_mse:
                best_overall_mse = phase2_best_mse
                best_overall_state = {name: param.detach().clone() for name, param in model.named_parameters()}
    
    # Restore absolute best
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in best_overall_state:
                param.copy_(best_overall_state[name])
    
    model.eval()
    with torch.no_grad():
        pred, _ = model(x, hard=True)
        final_mse = F.mse_loss(pred.squeeze(), y_sq).item()
    
    return final_mse, total_steps


def finalize_model_coefficients(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    max_steps: int = 500,
    target_mse: float = 0.01,
    sparsity_threshold: float = 0.15,
    l1_weight: float = 0.01,
) -> Tuple[float, str]:
    """
    ROBUST post-evolution coefficient finalization.
    
    Simplified approach: Just do L-BFGS on output_proj weights to find
    the best coefficients. Skip BatchNorm fusion (too error-prone).
    
    Args:
        model: The trained ONN model
        x, y: Training data
        max_steps: Max L-BFGS iterations
        target_mse: Stop when MSE is below this
        sparsity_threshold: Prune coefficients below this ratio of max
        l1_weight: L1 regularization weight for sparsity
        
    Returns:
        (final_mse, formula_string)
    """
    import gc
    
    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device)
    y_sq = y.squeeze()
    
    model.eval()
    with torch.no_grad():
        pred, _ = model(x, hard=True)
        initial_mse = F.mse_loss(pred.squeeze(), y_sq).item()
    
    # Just do L-BFGS refinement on output_proj WITH L1 SPARSITY
    output_params = [p for n, p in model.named_parameters() if 'output_proj' in n and p.requires_grad]
    
    if output_params and initial_mse > target_mse:
        try:
            def closure():
                optimizer.zero_grad()
                pred, _ = model(x, hard=True)
                mse_loss = F.mse_loss(pred.squeeze(), y_sq)
                # Add L1 sparsity penalty
                l1_loss = sum(p.abs().sum() for p in output_params)
                loss = mse_loss + l1_weight * l1_loss
                if not torch.isnan(loss):
                    loss.backward()
                return loss
            
            optimizer = LBFGS(output_params, lr=1.0, max_iter=max_steps, line_search_fn='strong_wolfe')
            optimizer.step(closure)
        except Exception:
            pass
    
    # Prune small coefficients
    prune_small_coefficients(model, threshold_ratio=sparsity_threshold, absolute_threshold=0.1)
    
    # Second L-BFGS pass (pure fitting, no L1)
    model.eval()
    with torch.no_grad():
        pred, _ = model(x, hard=True)
        post_prune_mse = F.mse_loss(pred.squeeze(), y_sq).item()
    
    if output_params and post_prune_mse > target_mse:
        try:
            def closure2():
                optimizer2.zero_grad()
                pred, _ = model(x, hard=True)
                loss = F.mse_loss(pred.squeeze(), y_sq)
                if not torch.isnan(loss):
                    loss.backward()
                return loss
            
            optimizer2 = LBFGS(output_params, lr=1.0, max_iter=max_steps // 2, line_search_fn='strong_wolfe')
            optimizer2.step(closure2)
        except Exception:
            pass
    
    # Final pruning
    prune_small_coefficients(model, threshold_ratio=0.08, absolute_threshold=0.05)
    
    model.eval()
    with torch.no_grad():
        pred, _ = model(x, hard=True)
        final_mse = F.mse_loss(pred.squeeze(), y_sq).item()
    
    formula = model.get_formula() if hasattr(model, 'get_formula') else "?"
    
    gc.collect()
    return final_mse, formula


def ablate_and_select_terms(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mse_tolerance: float = 2.0,
    max_terms: int = 6,
    verbose: bool = True,
) -> Tuple[float, str, List[int]]:
    """
    Combinatorial term selection: try subsets of discovered operations.
    
    Once evolution finds candidate operations (e.g., x², x, sin(x)), this function
    tries all subsets to find the SIMPLEST formula that fits the data well.
    
    Algorithm:
    1. Get current output_proj weights to identify active terms
    2. For each subset of active terms (from smallest to largest):
       a. Zero out non-selected terms
       b. L-BFGS fine-tune the selected coefficients
       c. If MSE is acceptable, prefer this simpler formula
    3. Return the simplest formula that achieves good MSE
    
    Args:
        model: The trained ONN model (should be finalized first)
        x, y: Training data
        mse_tolerance: Accept simpler formula if MSE < baseline_mse * mse_tolerance
        max_terms: Maximum number of terms to consider (for speed)
        verbose: Print progress
        
    Returns:
        (best_mse, best_formula, selected_indices)
    """
    from itertools import combinations
    import copy
    
    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device)
    y_sq = y.squeeze()
    
    # Get current weights and identify significant terms
    if not hasattr(model, 'output_proj'):
        return float('inf'), "?", []
    
    with torch.no_grad():
        weights = model.output_proj.weight.data.clone().squeeze()
        bias = model.output_proj.bias.data.clone() if model.output_proj.bias is not None else torch.zeros(1)
    
    # Find indices of non-zero weights (active terms)
    active_mask = weights.abs() > 0.01
    active_indices = torch.where(active_mask)[0].tolist()
    
    if len(active_indices) == 0:
        return float('inf'), "?", []
    
    if verbose:
        print(f"\n--- TERM ABLATION ---")
        print(f"Active terms: {len(active_indices)} indices: {active_indices}")
    
    # Limit number of terms to try
    if len(active_indices) > max_terms:
        sorted_idx = sorted(active_indices, key=lambda i: abs(weights[i].item()), reverse=True)
        active_indices = sorted_idx[:max_terms]
        if verbose:
            print(f"Limiting to top {max_terms} terms: {active_indices}")
    
    # Save original state
    original_state = copy.deepcopy(model.state_dict())
    
    # Get baseline MSE with all terms
    model.eval()
    with torch.no_grad():
        pred, _ = model(x, hard=True)
        baseline_mse = F.mse_loss(pred.squeeze(), y_sq).item()
    
    if verbose:
        print(f"Baseline MSE (all terms): {baseline_mse:.6f}")
    
    # Track best results - prefer FEWER terms
    best_results = {}  # n_terms -> (mse, formula, indices, state)
    best_results[len(active_indices)] = (baseline_mse, model.get_formula() if hasattr(model, 'get_formula') else "?", active_indices.copy(), copy.deepcopy(model.state_dict()))
    
    # Try subsets from smallest (1 term) to largest (all - 1)
    for n_terms in range(1, len(active_indices)):
        best_for_n = None
        
        for subset in combinations(active_indices, n_terms):
            subset = list(subset)
            
            # Reset to original state
            model.load_state_dict(copy.deepcopy(original_state))
            
            # Zero out non-selected terms
            with torch.no_grad():
                mask = torch.zeros_like(model.output_proj.weight.data)
                for idx in subset:
                    mask[0, idx] = 1.0
                model.output_proj.weight.data *= mask
            
            # L-BFGS fine-tune the selected coefficients
            output_params = [model.output_proj.weight, model.output_proj.bias] if model.output_proj.bias is not None else [model.output_proj.weight]
            try:
                def closure():
                    for p in output_params:
                        if p.grad is not None:
                            p.grad.zero_()
                    pred, _ = model(x, hard=True)
                    loss = F.mse_loss(pred.squeeze(), y_sq)
                    if not torch.isnan(loss):
                        loss.backward()
                    return loss
                
                optimizer = LBFGS(output_params, lr=1.0, max_iter=200, line_search_fn='strong_wolfe')
                optimizer.step(closure)
                
                # Re-apply mask after optimization
                with torch.no_grad():
                    model.output_proj.weight.data *= mask
            except Exception:
                continue
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                pred, _ = model(x, hard=True)
                mse = F.mse_loss(pred.squeeze(), y_sq).item()
            
            if math.isnan(mse) or math.isinf(mse):
                continue
            
            # Track best for this n_terms
            if best_for_n is None or mse < best_for_n[0]:
                best_for_n = (mse, model.get_formula() if hasattr(model, 'get_formula') else "?", subset.copy(), copy.deepcopy(model.state_dict()))
        
        if best_for_n is not None:
            best_results[n_terms] = best_for_n
            if verbose:
                print(f"  [{n_terms} terms] best MSE={best_for_n[0]:.6f} formula={best_for_n[1]}")
    
    # Select the simplest formula with acceptable MSE
    # Acceptable = within mse_tolerance of the baseline
    max_acceptable_mse = baseline_mse * mse_tolerance
    
    best_mse = baseline_mse
    best_formula = best_results[len(active_indices)][1]
    best_indices = active_indices
    best_state = best_results[len(active_indices)][3]
    
    # Try from fewest terms to most
    for n_terms in sorted(best_results.keys()):
        mse, formula, indices, state = best_results[n_terms]
        if mse <= max_acceptable_mse:
            best_mse = mse
            best_formula = formula
            best_indices = indices
            best_state = state
            break  # Take the first (simplest) acceptable solution
    
    # Restore best state
    model.load_state_dict(best_state)
    
    if verbose:
        print(f"\nSelected: {len(best_indices)} terms, MSE={best_mse:.6f}")
        print(f"Formula: {best_formula}")
    
    return best_mse, best_formula, best_indices


def hoyer_sparsity(weights: torch.Tensor) -> torch.Tensor:
    """
    Compute Hoyer sparsity measure (ratio of L1 to L2 norms).
    
    Hoyer = (sqrt(n) - L1/L2) / (sqrt(n) - 1)
    
    Range: [0, 1] where 1 = maximally sparse (one-hot)
    
    Research shows Hoyer regularization induces much sparser weights
    than L1 penalty alone.
    """
    n = weights.numel()
    if n <= 1:
        return torch.tensor(0.0)
    
    l1 = weights.abs().sum()
    l2 = weights.norm(2)
    
    if l2 < 1e-8:
        return torch.tensor(0.0)
    
    sqrt_n = math.sqrt(n)
    # Hoyer sparsity (higher = sparser)
    hoyer = (sqrt_n - l1 / l2) / (sqrt_n - 1 + 1e-8)
    
    return hoyer


def coefficient_sparsity_loss(model: nn.Module) -> torch.Tensor:
    """
    Compute sparsity loss for output projection coefficients.
    
    Encourages the model to use fewer terms in the final formula
    by pushing small coefficients toward zero.
    """
    if not hasattr(model, 'output_proj'):
        return torch.tensor(0.0)
    
    weights = model.output_proj.weight
    
    # L1 penalty on small weights (encourage exact zeros)
    l1_loss = weights.abs().mean()
    
    # Hoyer penalty (encourage sparsity pattern)
    hoyer = hoyer_sparsity(weights.flatten())
    hoyer_loss = 1.0 - hoyer  # Minimize this to maximize sparsity
    
    return 0.5 * l1_loss + 0.5 * hoyer_loss


class EvolutionaryONNTrainer:
    """
    Proper evolutionary training for ONN.
    
    Algorithm:
    1. Initialize population with RANDOM operations
    2. Evaluate each individual's fitness
    3. Select top performers
    4. Create offspring via discrete mutation
    5. Refine constants with gradient descent
    6. Repeat
    
    Key features (from research):
    - Entropy annealing: mild early (exploration), strong late (force discrete)
    - Progressive rounding: soft push toward integers (not hard snap)
    - Lamarckian inheritance: children inherit parent weights
    - Gradient-informed mutations: bias toward impactful changes
    - Coefficient pruning: remove low-impact terms for cleaner formulas
    - Explorer subpopulation: high-mutation scouts that find new basins
    """
    
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        population_size: int = 20,
        elite_size: int = 5,
        mutation_rate: float = 0.3,
        constant_refine_steps: int = 30,
        complexity_penalty: float = 0.02,
        device: Optional[torch.device] = None,
        tau_start: float = 1.0,
        tau_end: float = 0.2,
        entropy_weight: float = 0.01,
        normalize_data: bool = False,
        constant_refine_hard: bool = False,
        # NEW: Entropy annealing schedule (Tier 1)
        entropy_weight_start: float = 0.001,
        entropy_weight_end: float = 0.05,
        # NEW: Progressive rounding (Tier 1)
        progressive_round_weight: float = 0.01,
        # NEW: Lamarckian inheritance (Tier 2)
        lamarckian: bool = True,
        # NEW: Coefficient pruning
        prune_coefficients: bool = True,
        prune_every: int = 5,  # Prune every N generations
        prune_threshold: float = 0.15,  # Prune if |coef| < 15% of max (more aggressive)
        use_adaptive_pruning: bool = True,  # Use MSE-based pruning (smarter)
        # NEW: Explorer subpopulation
        use_explorers: bool = True,
        explorer_fraction: float = 0.2,  # 20% of population are explorers
        explorer_mutation_rate: float = 0.8,  # High mutation for exploration
        # NEW: Risk-seeking selection (Tier 2)
        risk_seeking: bool = False,  # Optimize top-k percentile instead of mean
        risk_seeking_percentile: float = 0.1,  # Top 10% fitness
        # NEW: Visualization
        visualizer: Optional[Any] = None,  # LiveTrainingVisualizer instance
    ):
        self.model_factory = model_factory
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.constant_refine_steps = constant_refine_steps
        self.complexity_penalty = complexity_penalty
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Tau annealing parameters
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.entropy_weight = entropy_weight  # Deprecated: use entropy_weight_start/end
        self.normalize_data = normalize_data
        self.constant_refine_hard = constant_refine_hard
        self.norm_stats: Optional[Dict] = None
        
        # NEW: Entropy annealing (research Tier 1)
        self.entropy_weight_start = entropy_weight_start
        self.entropy_weight_end = entropy_weight_end
        
        # NEW: Progressive rounding (research Tier 1)
        self.progressive_round_weight = progressive_round_weight
        
        # NEW: Lamarckian inheritance (research Tier 2)
        self.lamarckian = lamarckian
        
        # NEW: Coefficient pruning
        self.prune_coefficients = prune_coefficients
        self.prune_every = prune_every
        self.prune_threshold = prune_threshold
        self.use_adaptive_pruning = use_adaptive_pruning
        
        # NEW: Explorer subpopulation
        self.use_explorers = use_explorers
        self.explorer_fraction = explorer_fraction
        self.explorer_mutation_rate = explorer_mutation_rate
        self.n_explorers = max(2, int(population_size * explorer_fraction))
        
        # NEW: Risk-seeking selection (research Tier 2)
        self.risk_seeking = risk_seeking
        self.risk_seeking_percentile = risk_seeking_percentile
        
        # NEW: Visualization
        self.visualizer = visualizer
        
        self.population: List[Individual] = []
        self.explorers: List[Individual] = []  # Separate explorer population
        self.generation = 0
        self.best_ever: Optional[Individual] = None
        self.best_explorer: Optional[Individual] = None  # Best found by explorers
        self.history = []
    
    def initialize_population(self):
        """Create diverse initial population + explorer subpopulation."""
        self.population = []
        self.explorers = []
        
        # Main population
        for i in range(self.population_size):
            model = self.model_factory().to(self.device)
            random_operation_init(model, bias_strength=2.0 + i * 0.1)
            individual = Individual(model, generation=0)
            self.population.append(individual)
        
        # Explorer subpopulation (high mutation scouts)
        if self.use_explorers:
            for i in range(self.n_explorers):
                model = self.model_factory().to(self.device)
                # Initialize explorers with MORE random bias (broader search)
                random_operation_init(model, bias_strength=3.0 + i * 0.2)
                explorer = Individual(model, generation=0)
                explorer.is_explorer = True  # Tag as explorer
                self.explorers.append(explorer)
            print(f"Initialized population of {self.population_size} individuals + {self.n_explorers} explorers")
        else:
            print(f"Initialized population of {self.population_size} individuals")
        
        # Initialize visualizer if provided
        if self.visualizer is not None:
            # Get model dimensions from a sample model
            sample_model = self.population[0].model
            n_inputs = sample_model.n_inputs if hasattr(sample_model, 'n_inputs') else 1
            n_layers = sample_model.n_hidden_layers if hasattr(sample_model, 'n_hidden_layers') else 2
            nodes_per_layer = sample_model.nodes_per_layer if hasattr(sample_model, 'nodes_per_layer') else 4
            self.visualizer.initialize(n_inputs, n_layers, nodes_per_layer)
    
    def evaluate_fitness(self, x: torch.Tensor, y: torch.Tensor, 
                         generation: int = 0, total_generations: int = 50):
        """
        Evaluate fitness of all individuals (main population + explorers).
        
        Includes annealed penalties that change over generations:
        - Entropy: mild early (exploration), strong late (force discrete)
        - Progressive rounding: nudge parameters toward integers
        """
        x = x.to(self.device)
        y = y.to(self.device).squeeze()  # Ensure y is 1D
        
        # Compute annealed weights (Tier 1: entropy annealing)
        current_entropy_weight = anneal_entropy_weight(
            generation, total_generations,
            self.entropy_weight_start, self.entropy_weight_end
        )
        
        # Evaluate all individuals (main + explorers)
        all_individuals = self.population + (self.explorers if self.use_explorers else [])
        
        for ind in all_individuals:
            ind.model.eval()
            try:
                with torch.no_grad():
                    pred, _ = ind.model(x, hard=True)
                    pred = pred.squeeze()  # Ensure pred is 1D
                    mse = F.mse_loss(pred, y).item()
                    
                    if math.isnan(mse) or math.isinf(mse):
                        ind.fitness = float('inf')
                    else:
                        # Add complexity penalty (BIC-inspired parsimony)
                        complexity = calculate_complexity(ind.model)
                        fitness = mse + self.complexity_penalty * complexity
                        
                        # Coefficient sparsity penalty (Hoyer-inspired)
                        # INCREASED: Stronger sparsity encourages fewer output terms
                        sparsity = coefficient_sparsity_loss(ind.model).item()
                        fitness = fitness + 0.05 * sparsity
                        
                        # Entropy regularization with ANNEALED weight (Tier 1)
                        # Mild early (allow exploration), strong late (force discrete)
                        if current_entropy_weight > 0 and hasattr(ind.model, 'entropy_regularization'):
                            try:
                                entropy = ind.model.entropy_regularization().item()
                                fitness = fitness + current_entropy_weight * entropy
                            except Exception:
                                pass
                        
                        # Progressive rounding penalty (Tier 1)
                        # Soft push toward integers for p, omega, etc.
                        if self.progressive_round_weight > 0:
                            try:
                                round_loss = progressive_round_loss(ind.model).item()
                                fitness = fitness + self.progressive_round_weight * round_loss
                            except Exception:
                                pass
                        
                        ind.fitness = fitness
            except Exception:
                ind.fitness = float('inf')
        
        # Track best ever (from main population)
        best_current = min(self.population, key=lambda ind: ind.fitness)
        if self.best_ever is None or best_current.fitness < self.best_ever.fitness:
            self.best_ever = best_current.clone()
        
        # Track best explorer and handle migration
        if self.use_explorers and self.explorers:
            best_explorer = min(self.explorers, key=lambda ind: ind.fitness)
            
            # Check if explorer found something better than main population
            if best_explorer.fitness < best_current.fitness:
                # MIGRATION: Explorer found a better basin!
                # Inject the explorer's solution into main population
                if self.best_explorer is None or best_explorer.fitness < self.best_explorer.fitness:
                    self.best_explorer = best_explorer.clone()
                    
                    # Replace worst individuals in main population with 
                    # copies of the explorer's discovery
                    sorted_pop = sorted(self.population, key=lambda ind: ind.fitness)
                    n_migrate = min(3, len(sorted_pop) // 4)  # Migrate up to 3 or 25%
                    
                    for i in range(n_migrate):
                        # Clone explorer and add slight variation
                        migrant = best_explorer.clone()
                        migrant.is_explorer = False
                        # Replace worst individual
                        self.population[-(i+1)] = migrant
                    
                    # Also update best_ever if explorer beat it
                    if best_explorer.fitness < self.best_ever.fitness:
                        self.best_ever = best_explorer.clone()
    
    def evolve_explorers(self, x: torch.Tensor, y: torch.Tensor):
        """
        Evolve the explorer subpopulation with HIGH mutation.
        
        Explorers are scouts that search for new promising regions.
        They have very high mutation rate and less constant refinement.
        """
        if not self.use_explorers or not self.explorers:
            return
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Sort explorers by fitness
        sorted_explorers = sorted(self.explorers, key=lambda ind: ind.fitness)
        
        new_explorers = []
        
        # Keep best explorer
        new_explorers.append(sorted_explorers[0])
        
        # Generate new explorers through heavy mutation
        while len(new_explorers) < self.n_explorers:
            # Pick a random parent from top half
            parent = random.choice(sorted_explorers[:max(1, len(sorted_explorers)//2)])
            
            # HIGH mutation - explorers move a LOT
            if random.random() < 0.3:
                # Sometimes create completely random explorer (injection)
                model = self.model_factory().to(self.device)
                random_operation_init(model, bias_strength=4.0)
                child = Individual(model, generation=self.generation)
            else:
                # Heavy mutation of parent
                child = mutate_operations(parent, mutation_rate=self.explorer_mutation_rate)
            
            child.is_explorer = True
            
            # Light refinement (explorers don't over-optimize)
            refine_constants(
                child.model, x, y,
                steps=5,  # Very few steps - just enough to evaluate
                lr=0.03,
                hard=self.constant_refine_hard,
            )
            
            new_explorers.append(child)
        
        self.explorers = new_explorers
    
    def select_and_reproduce(self, x: torch.Tensor, y: torch.Tensor, diversity: int = 10):
        """
        Selection and reproduction with Lamarckian inheritance.
        
        Tier 2 features:
        - Lamarckian inheritance: passes optimized weights from parent to child
        - Risk-seeking selection: bias toward top performers (for SR we want 
          the BEST formula, not average performance)
        """
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda ind: ind.fitness)
        
        # Risk-seeking selection: only consider top percentile for reproduction
        if self.risk_seeking:
            top_k = max(2, int(len(sorted_pop) * self.risk_seeking_percentile))
            selection_pool = sorted_pop[:top_k]
        else:
            selection_pool = sorted_pop[:self.population_size//2]
        
        new_population = []
        
        # Mass Mutation condition: Low diversity
        if diversity < 3:
            # Keep only the very best (Elite = 1 or 2)
            for elite in sorted_pop[:2]:
                new_population.append(elite)
            
            # Fill the rest with fresh random individuals + heavy mutation
            while len(new_population) < self.population_size:
                if random.random() < 0.5:
                    # Brand new random individual (Injection)
                    model = self.model_factory().to(self.device)
                    random_operation_init(model, bias_strength=3.0)
                    child = Individual(model, generation=self.generation)
                else:
                    # Heavily mutated elite (Exploration)
                    parent = sorted_pop[0]  # Best one
                    child = mutate_operations(parent, mutation_rate=0.8)
                    # Lamarckian: child inherits parent's refined weights
                
                # Fast refine
                refine_constants(
                    child.model, x, y,
                    steps=10,
                    lr=0.02,
                    hard=self.constant_refine_hard,
                )
                new_population.append(child)
                
        else:
            # Normal Evolution
            
            # Keep elite unchanged
            for elite in sorted_pop[:self.elite_size]:
                new_population.append(elite)
            
            # Fill rest with mutations of top performers
            while len(new_population) < self.population_size:
                # Tournament selection from selection pool
                candidates = random.sample(selection_pool, min(3, len(selection_pool)))
                parent = min(candidates, key=lambda ind: ind.fitness)
                
                # Lamarckian mutation: clone preserves all parent weights
                # Mutation only changes discrete structure (operation selection)
                # Continuous weights (p, omega, etc.) are inherited!
                if self.lamarckian:
                    child = mutate_operations_lamarckian(parent, self.mutation_rate)
                else:
                    child = mutate_operations(parent, self.mutation_rate)
                
                # Refine constants (shorter if Lamarckian since weights are pre-tuned)
                steps = self.constant_refine_steps // 2 if self.lamarckian else self.constant_refine_steps
                refine_constants(
                    child.model, x, y,
                    steps=max(10, steps),
                    lr=0.02,
                    hard=self.constant_refine_hard,
                )
                
                new_population.append(child)
        
        self.population = new_population
        self.generation += 1
    
    def train(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        fitness_x: Optional[torch.Tensor] = None,
        fitness_y: Optional[torch.Tensor] = None,
        generations: int = 50,
        print_every: int = 5,
    ) -> Dict:
        """
        Full evolutionary training.
        
        Args:
            x: Input data
            y: Target data
            generations: Number of generations
            print_every: Print frequency
            
        Returns:
            Training results
        """
        print("="*60)
        print("EVOLUTIONARY ONN TRAINING")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Population: {self.population_size}")
        print(f"Generations: {generations}")
        print("-"*60)
        
        x = x.to(self.device)
        y = y.to(self.device)
        fit_x = fitness_x.to(self.device) if fitness_x is not None else x
        fit_y = fitness_y.to(self.device) if fitness_y is not None else y

        # Optional normalization (improves stability on complex targets)
        if self.normalize_data:
            x_mean = x.mean(dim=0, keepdim=True)
            x_std = x.std(dim=0, keepdim=True).clamp(min=1e-6)
            y_mean = y.mean()
            y_std = y.std().clamp(min=1e-6)
            self.norm_stats = {
                'x_mean': x_mean,
                'x_std': x_std,
                'y_mean': y_mean,
                'y_std': y_std,
            }

            x = (x - x_mean) / x_std
            y = (y - y_mean) / y_std
            fit_x = (fit_x - x_mean) / x_std
            fit_y = (fit_y - y_mean) / y_std
        
        # Initialize
        self.initialize_population()
        
        # Initial constant refinement for main population
        print("Refining initial constants...")
        for ind in self.population:
            refine_constants(
                ind.model, x, y,
                steps=self.constant_refine_steps,
                hard=self.constant_refine_hard,
            )
        
        # Light initial refinement for explorers (they stay mobile)
        if self.use_explorers:
            for explorer in self.explorers:
                refine_constants(
                    explorer.model, x, y,
                    steps=5,  # Very light - explorers stay mobile
                    hard=self.constant_refine_hard,
                )
        
        start_time = time.time()
        structure_locked = False  # Flag to skip mutations once structure is found
        
        for gen in range(generations):
            # Anneal tau for all individuals (early soft, late hard)
            current_tau = anneal_tau(gen, generations, self.tau_start, self.tau_end)
            for ind in self.population:
                set_model_tau(ind.model, current_tau)
            
            # Explorers keep HIGHER tau (stay soft for more exploration)
            if self.use_explorers and not structure_locked:
                explorer_tau = max(current_tau, 0.5)  # Explorers never go below 0.5
                for explorer in self.explorers:
                    set_model_tau(explorer.model, explorer_tau)
            
            # Evaluate with annealed entropy weight (pass generation info)
            self.evaluate_fitness(fit_x, fit_y, generation=gen, total_generations=generations)
            
            # Check for early structure lock (high correlation)
            if not structure_locked and self.best_ever and gen >= 5:
                is_good, corr, _ = check_structure_quality(
                    self.best_ever.model, x, y, corr_threshold=0.995
                )
                if is_good:
                    structure_locked = True
                    print(f"Gen {gen:3d} | STRUCTURE LOCKED (corr={corr:.4f}) - switching to coefficient-only refinement")
            
            # Stats
            fitnesses = [ind.fitness for ind in self.population if ind.fitness < float('inf')]
            if fitnesses:
                best_fit = min(fitnesses)
                mean_fit = sum(fitnesses) / len(fitnesses)
                diversity = len(set(ind.structure_hash for ind in self.population))
            else:
                best_fit = mean_fit = float('inf')
                diversity = 0
            
            # Compute current entropy weight for logging
            current_entropy_weight = anneal_entropy_weight(
                gen, generations, self.entropy_weight_start, self.entropy_weight_end
            )
            
            self.history.append({
                'generation': gen,
                'best_fitness': best_fit,
                'mean_fitness': mean_fit,
                'diversity': diversity,
                'best_ever': self.best_ever.fitness if self.best_ever else float('inf'),
                'tau': current_tau,
                'entropy_weight': current_entropy_weight,
            })
            
            if gen % print_every == 0 or gen == generations - 1:
                explorer_info = ""
                if self.use_explorers and self.explorers:
                    best_explorer_fit = min(e.fitness for e in self.explorers if e.fitness < float('inf'))
                    explorer_info = f" | Explorer: {best_explorer_fit:.4f}"
                print(f"Gen {gen:3d} | Best: {best_fit:.4f} | "
                      f"Mean: {mean_fit:.4f} | Diversity: {diversity} | "
                      f"Best Ever: {self.best_ever.fitness:.4f}{explorer_info}")
            
            # Update visualization if available
            if self.visualizer is not None and self.best_ever is not None:
                # Get formula and correlation for visualization
                formula = ""
                correlation = 0.0
                if hasattr(self.best_ever.model, 'get_formula'):
                    try:
                        formula = self.best_ever.model.get_formula()
                    except Exception:
                        formula = "?"
                
                # Compute correlation
                try:
                    self.best_ever.model.eval()
                    with torch.no_grad():
                        pred, _ = self.best_ever.model(x, hard=True)
                        pred_np = pred.squeeze().cpu().numpy()
                        y_np = y.squeeze().cpu().numpy()
                        import numpy as np
                        correlation = float(np.corrcoef(pred_np.flatten(), y_np.flatten())[0, 1])
                        if math.isnan(correlation):
                            correlation = 0.0
                except Exception:
                    correlation = 0.0
                
                self.visualizer.on_generation(
                    generation=gen,
                    model=self.best_ever.model,
                    x=x,
                    y=y,
                    history=self.history,
                    formula=formula,
                    best_fitness=self.best_ever.fitness,
                    correlation=correlation,
                )
            
            # NEW: Coefficient pruning - remove low-impact terms periodically
            # This cleans up formulas by zeroing small coefficients
            if self.prune_coefficients and gen > 0 and gen % self.prune_every == 0:
                for ind in self.population:
                    if self.use_adaptive_pruning:
                        # Adaptive pruning: uses MSE contribution (smarter)
                        # Conservative: only prune if MSE increase < 2%
                        n_pruned, new_mse = adaptive_coefficient_pruning(
                            ind.model, fit_x, fit_y,
                            prune_ratio=0.02  # Prune if MSE increase < 2% 
                        )
                    else:
                        # Simple threshold pruning
                        n_pruned = prune_small_coefficients(
                            ind.model, 
                            threshold_ratio=self.prune_threshold,
                            absolute_threshold=0.1
                        )
                    # Re-evaluate after pruning
                    if n_pruned > 0:
                        ind.model.eval()
                        with torch.no_grad():
                            pred, _ = ind.model(fit_x, hard=True)
                            mse = F.mse_loss(pred.squeeze(), fit_y.squeeze()).item()
                            if not math.isnan(mse) and not math.isinf(mse):
                                complexity = calculate_complexity(ind.model)
                                ind.fitness = mse + self.complexity_penalty * complexity
            
            # Reproduce (skip on last generation)
            if gen < generations - 1:
                if structure_locked:
                    # Structure locked: STOP all mutations and exploration
                    # Don't modify best_ever here - save refinement for the end
                    # Just skip this generation's reproduction
                    pass
                else:
                    # Normal evolution with mutations
                    self.select_and_reproduce(x, y, diversity=diversity)
                    
                    # Evolve explorers in parallel (if enabled)
                    self.evolve_explorers(x, y)
        
        elapsed = time.time() - start_time
        
        # Final pruning on best model for clean formula
        if self.prune_coefficients and self.best_ever:
            if self.use_adaptive_pruning:
                # Adaptive: prune terms that don't contribute much to MSE
                # Conservative final prune: only 1% MSE tolerance
                n_final_pruned, _ = adaptive_coefficient_pruning(
                    self.best_ever.model, x, y,
                    prune_ratio=0.01  # Conservative final prune (1% MSE tolerance)
                )
            else:
                n_final_pruned = prune_small_coefficients(
                    self.best_ever.model,
                    threshold_ratio=self.prune_threshold,
                    absolute_threshold=0.1
                )
            if n_final_pruned > 0:
                print(f"Final pruning removed {n_final_pruned} small coefficients")
        
        # STRUCTURE-LOCK REFINEMENT
        # If correlation is high but MSE is imperfect, we have the right structure
        # but wrong coefficients. Lock structure and do intensive coefficient tuning.
        if self.best_ever:
            import copy
            
            # Check if structure is good (high correlation)
            structure_good, corr, mse_before = check_structure_quality(
                self.best_ever.model, x, y, corr_threshold=0.99
            )
            
            print(f"Structure check: corr={corr:.4f}, MSE={mse_before:.4f}")
            
            if structure_good and mse_before > 0.01:
                # Structure is good - refinement historically hurts more than helps
                # The evolutionary process with constant_refine_steps already tunes coefficients
                # Additional refinement tends to destabilize the model
                print(f"Structure LOCKED (corr={corr:.4f}). MSE={mse_before:.4f} (skipping refinement - evolution already tuned)")
                
                # Optionally do VERY gentle output_proj-only refinement
                # This is the safest as it only affects the final linear combination
                backup_state = copy.deepcopy(self.best_ever.model.state_dict())
                
                output_proj_params = [p for n, p in self.best_ever.model.named_parameters() if 'output_proj' in n]
                if output_proj_params:
                    self.best_ever.model.train()
                    optimizer = Adam(output_proj_params, lr=0.02)
                    y_sq = y.squeeze()
                    
                    best_mse = mse_before
                    best_out_state = {n: p.detach().clone() for n, p in self.best_ever.model.named_parameters() if 'output_proj' in n}
                    
                    for _ in range(200):
                        optimizer.zero_grad()
                        pred, _ = self.best_ever.model(x, hard=True)
                        loss = F.mse_loss(pred.squeeze(), y_sq)
                        if torch.isnan(loss):
                            break
                        if loss.item() < best_mse:
                            best_mse = loss.item()
                            best_out_state = {n: p.detach().clone() for n, p in self.best_ever.model.named_parameters() if 'output_proj' in n}
                        loss.backward()
                        optimizer.step()
                    
                    # Restore best output_proj state
                    with torch.no_grad():
                        for n, p in self.best_ever.model.named_parameters():
                            if n in best_out_state:
                                p.copy_(best_out_state[n])
                    
                    self.best_ever.model.eval()
                    with torch.no_grad():
                        pred_after, _ = self.best_ever.model(x, hard=True)
                        mse_after = F.mse_loss(pred_after.squeeze(), y.squeeze()).item()
                    
                    if mse_after < mse_before:
                        print(f"  Output-proj refinement: MSE {mse_before:.4f} -> {mse_after:.4f}")
                    else:
                        # Revert entirely
                        self.best_ever.model.load_state_dict(backup_state)
            
            elif mse_before > 0.01:
                # Structure not locked, try gentle refinement
                print("Structure not locked, trying gentle refinement...")
                backup_state = copy.deepcopy(self.best_ever.model.state_dict())
                
                refine_constants(
                    self.best_ever.model, x, y,
                    steps=200,
                    lr=0.005,
                    use_lbfgs=False,
                    scales_only=True,
                    hard=True,
                )
                
                self.best_ever.model.eval()
                with torch.no_grad():
                    pred_after, _ = self.best_ever.model(x, hard=True)
                    mse_after = F.mse_loss(pred_after.squeeze(), y.squeeze()).item()
                
                if mse_after > mse_before * 1.1:
                    print(f"  Refinement hurt (MSE {mse_before:.4f} -> {mse_after:.4f}), reverting...")
                    self.best_ever.model.load_state_dict(backup_state)
                else:
                    print(f"  Refinement: MSE {mse_before:.4f} -> {mse_after:.4f}")
        
        # Final evaluation
        print("-"*60)
        print(f"Training complete in {elapsed:.1f}s")
        
        # Get best model
        best_model = self.best_ever.model if self.best_ever else self.population[0].model
        
        # IMPORTANT: Do NOT snap_to_discrete - it destroys model performance
        # The continuous parameters are more expressive and give better results
        # Formula extraction still works with continuous values
        # 
        # if hasattr(best_model, 'snap_to_discrete'):
        #     best_model.snap_to_discrete()
        
        # CRITICAL: Put model in eval mode for deterministic behavior
        best_model.eval()
        
        # Evaluate on TRAINING data (for debugging)
        with torch.no_grad():
            pred_train, _ = best_model(x, hard=True)
            mse_train = F.mse_loss(pred_train.squeeze(), y.squeeze()).item()
            print(f"  Train MSE: {mse_train:.4f}")
        
        # Also evaluate on VALIDATION data if provided
        if fitness_x is not None and fitness_y is not None:
            with torch.no_grad():
                pred_val, _ = best_model(fit_x, hard=True)
                mse_val = F.mse_loss(pred_val.squeeze(), fit_y.squeeze()).item()
                print(f"  Val MSE: {mse_val:.4f}")
        
        # DISABLED: Compiled inference path (has bugs causing MSE discrepancy)
        use_compiled = False
        # if hasattr(best_model, 'compile_for_inference'):
        #     try:
        #         best_model.compile_for_inference()
        #         use_compiled = True
        #     except Exception:
        #         pass
        
        # Squeeze y for consistent shape
        y_eval = y.squeeze()
        
        with torch.no_grad():
            if use_compiled and hasattr(best_model, 'forward_compiled'):
                pred = best_model.forward_compiled(x)
            else:
                pred, _ = best_model(x, hard=True)
            pred = pred.squeeze()
            
            # DEBUG: Check MSE before denorm
            final_mse_raw = F.mse_loss(pred, y_eval).item()
            print(f"  DEBUG: MSE raw (before denorm): {final_mse_raw:.4f}")
            
            if self.normalize_data and self.norm_stats:
                pred = pred * self.norm_stats['y_std'] + self.norm_stats['y_mean']
                y_eval = y_eval * self.norm_stats['y_std'] + self.norm_stats['y_mean']
            final_mse = F.mse_loss(pred, y_eval).item()
            corr = torch.corrcoef(torch.stack([
                pred.cpu(), y_eval.cpu()
            ]))[0, 1].item()
        
        formula = "N/A"
        if hasattr(best_model, 'get_formula'):
            formula = best_model.get_formula()
        elif hasattr(best_model, 'get_graph_summary'):
            formula = best_model.get_graph_summary()
        
        print(f"Final MSE: {final_mse:.4f}")
        print(f"Correlation: {corr:.4f}")
        print(f"Formula: {formula}")
        
        return {
            'model': best_model,
            'history': self.history,
            'final_mse': final_mse,
            'correlation': corr,
            'formula': formula,
            'training_time': elapsed,
        }


def train_onn_evolutionary(
    model_factory: Callable[[], nn.Module],
    x: torch.Tensor,
    y: torch.Tensor,
    population_size: int = 20,
    generations: int = 50,
    device: Optional[torch.device] = None,
    fitness_x: Optional[torch.Tensor] = None,
    fitness_y: Optional[torch.Tensor] = None,
    normalize_data: bool = False,
    constant_refine_hard: bool = False,
    elite_fraction: float = 0.25,
    mutation_rate: float = 0.3,
    constant_refine_steps: int = 30,
    tau_start: float = 1.0,
    tau_end: float = 0.2,
    prune_coefficients: bool = True,
    prune_threshold: float = 0.15,
    use_adaptive_pruning: bool = True,
    # Explorer subpopulation
    use_explorers: bool = True,
    explorer_fraction: float = 0.2,
    explorer_mutation_rate: float = 0.8,
) -> Dict:
    """
    Convenience function for evolutionary ONN training.
    
    Usage:
        from glassbox.sr import OperationDAG
        from glassbox.sr.evolution import train_onn_evolutionary
        
        def make_model():
            return OperationDAG(n_inputs=1, n_hidden_layers=2, nodes_per_layer=4)
        
        results = train_onn_evolutionary(make_model, x, y, generations=50)
        print(results['formula'])
    
    Args:
        model_factory: Callable that creates a new model instance
        x, y: Training data
        population_size: Number of individuals in population
        generations: Number of evolutionary generations
        device: Torch device
        fitness_x, fitness_y: Optional validation data for fitness evaluation
        normalize_data: If True, normalize x and y during training
        constant_refine_hard: If True, use hard selection during constant refinement
        elite_fraction: Fraction of population to keep as elites (0.0-1.0)
        mutation_rate: Probability of mutating each parameter
        constant_refine_steps: Number of gradient steps for constant refinement
        tau_start, tau_end: Temperature annealing schedule
        prune_coefficients: If True, periodically prune small coefficients
        prune_threshold: Threshold for coefficient pruning (ratio of max)
        use_adaptive_pruning: If True, use MSE-based pruning (smarter)
        use_explorers: If True, maintain a high-mutation explorer subpopulation
        explorer_fraction: Fraction of population size for explorers
        explorer_mutation_rate: Mutation rate for explorers (high for exploration)
    """
    elite_size = max(2, int(population_size * elite_fraction))
    
    trainer = EvolutionaryONNTrainer(
        model_factory=model_factory,
        population_size=population_size,
        elite_size=elite_size,
        mutation_rate=mutation_rate,
        constant_refine_steps=constant_refine_steps,
        device=device,
        normalize_data=normalize_data,
        constant_refine_hard=constant_refine_hard,
        tau_start=tau_start,
        tau_end=tau_end,
        prune_coefficients=prune_coefficients,
        prune_threshold=prune_threshold,
        use_adaptive_pruning=use_adaptive_pruning,
        use_explorers=use_explorers,
        explorer_fraction=explorer_fraction,
        explorer_mutation_rate=explorer_mutation_rate,
    )
    return trainer.train(
        x, y,
        fitness_x=fitness_x,
        fitness_y=fitness_y,
        generations=generations,
    )


def train_onn_hybrid(
    model_factory: Callable[[], nn.Module],
    x: torch.Tensor,
    y: torch.Tensor,
    cycles: int = 5,
    es_generations_per_cycle: int = 10,
    gd_steps_per_cycle: int = 100,
    population_size: int = 15,
    device: Optional[torch.device] = None,
    gd_lr: float = 0.01,
) -> Dict:
    """
    Hybrid ES + Gradient training (Tier 2).
    
    Alternates between:
    1. Evolutionary structure search (discrete mutations)
    2. Gradient descent refinement (continuous optimization)
    
    This combines the exploration ability of ES with the 
    precision of gradient descent.
    
    Args:
        model_factory: Function that creates a new model
        x, y: Training data
        cycles: Number of ES/GD cycles
        es_generations_per_cycle: ES generations per cycle
        gd_steps_per_cycle: Gradient steps per cycle
        population_size: ES population size
        device: Torch device
        gd_lr: Learning rate for gradient descent
    
    Returns:
        Training results dict
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    y = y.to(device)
    
    print("="*60)
    print("HYBRID ES + GRADIENT TRAINING")
    print("="*60)
    print(f"Cycles: {cycles}")
    print(f"ES gens/cycle: {es_generations_per_cycle}")
    print(f"GD steps/cycle: {gd_steps_per_cycle}")
    print("-"*60)
    
    start_time = time.time()
    
    # Initialize trainer
    trainer = EvolutionaryONNTrainer(
        model_factory=model_factory,
        population_size=population_size,
        elite_size=max(2, population_size // 4),
        device=device,
        lamarckian=True,
    )
    trainer.initialize_population()
    
    best_model = None
    best_mse = float('inf')
    history = []
    
    for cycle in range(cycles):
        print(f"\n--- Cycle {cycle+1}/{cycles} ---")
        
        # Phase 1: ES structure search
        print(f"  ES Phase: {es_generations_per_cycle} generations...")
        for gen in range(es_generations_per_cycle):
            current_tau = anneal_tau(
                cycle * es_generations_per_cycle + gen,
                cycles * es_generations_per_cycle,
                1.0, 0.3
            )
            for ind in trainer.population:
                set_model_tau(ind.model, current_tau)
            
            trainer.evaluate_fitness(
                x, y, 
                generation=cycle * es_generations_per_cycle + gen,
                total_generations=cycles * es_generations_per_cycle
            )
            
            diversity = len(set(ind.structure_hash for ind in trainer.population))
            if gen < es_generations_per_cycle - 1:
                trainer.select_and_reproduce(x, y, diversity=diversity)
        
        # Get best from ES
        es_best = trainer.best_ever.model if trainer.best_ever else trainer.population[0].model
        
        # Phase 2: Gradient refinement on best
        print(f"  GD Phase: {gd_steps_per_cycle} steps...")
        es_best.train()
        
        # Only optimize continuous parameters
        continuous_params = [
            p for n, p in es_best.named_parameters() 
            if 'logit' not in n and 'selector' not in n and p.requires_grad
        ]
        
        if continuous_params:
            optimizer = Adam(continuous_params, lr=gd_lr)
            
            for step in range(gd_steps_per_cycle):
                optimizer.zero_grad()
                pred, _ = es_best(x, hard=False)
                loss = F.mse_loss(pred.squeeze(), y.squeeze())
                
                if torch.isnan(loss):
                    break
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(continuous_params, 1.0)
                optimizer.step()
        
        # Evaluate after GD
        es_best.eval()
        with torch.no_grad():
            pred, _ = es_best(x, hard=True)
            cycle_mse = F.mse_loss(pred.squeeze(), y.squeeze()).item()
        
        print(f"  Cycle {cycle+1} MSE: {cycle_mse:.6f}")
        
        history.append({
            'cycle': cycle,
            'mse': cycle_mse,
        })
        
        if cycle_mse < best_mse:
            best_mse = cycle_mse
            best_model = copy.deepcopy(es_best)
        
        # Inject refined model back into population
        if trainer.best_ever:
            trainer.best_ever.model = copy.deepcopy(es_best)
            trainer.best_ever.fitness = cycle_mse
    
    elapsed = time.time() - start_time
    
    # Final evaluation
    best_model.eval()
    with torch.no_grad():
        pred, _ = best_model(x, hard=True)
        final_mse = F.mse_loss(pred.squeeze(), y.squeeze()).item()
        corr = torch.corrcoef(torch.stack([
            pred.squeeze().cpu(), y.squeeze().cpu()
        ]))[0, 1].item()
    
    formula = "N/A"
    if hasattr(best_model, 'get_formula'):
        formula = best_model.get_formula()
    
    print("-"*60)
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Final MSE: {final_mse:.6f}")
    print(f"Correlation: {corr:.4f}")
    print(f"Formula: {formula}")
    
    return {
        'model': best_model,
        'history': history,
        'final_mse': final_mse,
        'correlation': corr,
        'formula': formula,
        'training_time': elapsed,
    }
