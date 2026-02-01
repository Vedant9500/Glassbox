"""
Glassbox SR Tester - Unified Symbolic Regression Testing Tool

A single script with TUI that consolidates all test functionality:
- Visualization mode: Phased regression with live visualization
- Evolution mode: Benchmark on preset formulas
- Pruning mode: Post-training pruning pipeline
- Interactive mode: Custom formula testing
- Single mode: Test a specific formula

Usage:
    python scripts/sr_tester.py                          # Launch TUI
    python scripts/sr_tester.py --mode single --formula "x^2"
    python scripts/sr_tester.py --help                   # Show all flags

Author: Glassbox Project
"""

import sys
import os
import re
import math
import copy
import time
import argparse
from typing import Callable, Dict, Tuple, Optional, List

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np


def sanitize_formula(formula: str) -> str:
    """Sanitize formula for Windows console output (replace Unicode chars)."""
    if not formula:
        return formula
    result = (formula
        .replace('π', 'pi')
        .replace('²', '^2')
        .replace('³', '^3')
        .replace('√', 'sqrt')
        .replace('·', '*')
        .replace('φ', 'phi')
        .replace('ω', 'omega')
    )
    # Remove any other problematic Unicode
    return result.encode('ascii', 'replace').decode('ascii')


# Check for Rich TUI library
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.style import Style
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Glassbox imports
from glassbox.sr.operation_dag import OperationDAG
from glassbox.sr.evolution import EvolutionaryONNTrainer, train_onn_evolutionary, finalize_model_coefficients
from glassbox.sr.visualization import LiveTrainingVisualizer
from glassbox.sr.pruning import PostTrainingPruner, prune_model
from glassbox.sr import generate_polynomial_data, BaselineMLP, get_device

console = Console() if HAS_RICH else None


# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================

class Config:
    """All configurable parameters for SR testing."""
    
    def __init__(self):
        # Mode
        self.mode: str = "interactive"
        
        # Model Architecture
        self.hidden_layers: int = 2
        self.nodes_per_layer: int = 4
        self.simplified_ops: bool = True
        self.fair_mode: bool = True
        
        # Evolution Hyperparameters
        self.population: int = 30
        self.generations: int = 40
        self.elite_size: int = 4
        self.mutation_rate: float = 0.5
        self.explorer_fraction: float = 0.40
        self.explorer_mutation: float = 0.85
        self.constant_refine_steps: int = 50
        self.refine_omega: bool = True
        self.refine_internal: bool = True
        self.lamarckian: bool = True
        self.prune_coefficients: bool = False
        self.risk_seeking: bool = True
        self.risk_seeking_percentile: float = 0.1
        
        # Data Generation
        self.x_min: float = -10.0
        self.x_max: float = 10.0
        self.n_samples: int = 300
        self.noise_std: float = 0.0
        self.precision: int = 32
        self.normalize_data: bool = False
        self.auto_domain: bool = True
        
        # Other
        self.formula: str = ""
        self.lite_mode: bool = False
        self.no_viz: bool = False
        self.device: str = "auto"
        self.seed: Optional[int] = None
        self.compare_mlp: bool = False
        self.enable_pruning: bool = False
        
        # Operator Constraints (None = use default based on simplified_ops)
        self.ops_periodic: Optional[bool] = None  # sin, cos
        self.ops_power: Optional[bool] = None     # x^n, sqrt
        self.ops_exp: Optional[bool] = None       # exp(x)
        self.ops_log: Optional[bool] = None       # log(x)
        self.ops_arithmetic: Optional[bool] = None  # +, *
        self.ops_aggregation: Optional[bool] = None  # sum, mean, max
        
        # Domain-Aware Sampling
        self.sample_avoid: Optional[List[float]] = None  # Singular points to avoid
        self.sample_epsilon: float = 0.1                 # Distance to stay from singular points
        self.use_sample_weights: bool = False            # Use sample weights in fitness

        # Fast-path only mode
        self.fast_path_only: bool = False
    
    def get_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


# =============================================================================
# FORMULA PARSER
# =============================================================================

def parse_formula(formula_str: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Parse a user-entered formula string into a PyTorch function."""
    formula = formula_str.strip().lower()
    formula = re.sub(r'\^(\d+)', r'**\1', formula)
    formula = re.sub(r'\^(\()', r'**\1', formula)
    formula = re.sub(r'\bx(\d+)\b', r'x[:, \1:(\1+1)]', formula)
    formula = re.sub(r'\bx\b', 'x[:, 0:1]', formula)
    formula = formula.replace('sin(', 'torch.sin(')
    formula = formula.replace('cos(', 'torch.cos(')
    formula = formula.replace('tan(', 'torch.tan(')
    formula = formula.replace('exp(', 'torch.exp(')
    formula = formula.replace('log(', 'torch.log(')
    # Handle sqrt of numeric constants before tensor sqrt
    formula = re.sub(r'sqrt\(\s*([0-9\.]+)\s*\)', lambda m: str(math.sqrt(float(m.group(1)))), formula)
    formula = re.sub(r'sqrt\(\s*pi\s*\)', str(math.sqrt(math.pi)), formula)
    formula = re.sub(r'sqrt\(\s*e\s*\)', str(math.sqrt(math.e)), formula)
    formula = formula.replace('sqrt(', 'torch.sqrt(')
    formula = formula.replace('abs(', 'torch.abs(')
    formula = formula.replace('pi', str(math.pi))
    formula = re.sub(r'\be\b', str(math.e), formula)
    
    def target_fn(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.reshape(-1, 1)
        try:
            result = eval(formula)
            if isinstance(result, (int, float)):
                result = torch.full((x.shape[0], 1), result, dtype=x.dtype, device=x.device)
            return result.reshape(-1, 1)
        except Exception as e:
            raise ValueError(f"Error evaluating formula '{formula_str}': {e}")
    
    return target_fn


# =============================================================================
# CORE TESTER CLASS
# =============================================================================

class SRTester:
    """Unified Symbolic Regression Tester."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.get_device()
        
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
    
    def make_model(self) -> OperationDAG:
        """Create model with current config."""
        # Build operator constraints dict
        op_constraints = None
        if any(x is not None for x in [
            self.config.ops_periodic, self.config.ops_power,
            self.config.ops_exp, self.config.ops_log,
            self.config.ops_arithmetic, self.config.ops_aggregation
        ]):
            op_constraints = {
                'periodic': self.config.ops_periodic,
                'power': self.config.ops_power,
                'exp': self.config.ops_exp,
                'log': self.config.ops_log,
                'arithmetic': self.config.ops_arithmetic,
                'aggregation': self.config.ops_aggregation,
            }
            # Remove None values (use default)
            op_constraints = {k: v for k, v in op_constraints.items() if v is not None}
        
        model = OperationDAG(
            n_inputs=1,
            n_hidden_layers=self.config.hidden_layers,
            nodes_per_layer=self.config.nodes_per_layer,
            n_outputs=1,
            simplified_ops=self.config.simplified_ops,
            fair_mode=self.config.fair_mode,
            op_constraints=op_constraints,
        )
        if self.config.precision == 64:
            model = model.double()
        return model
    
    def generate_data(self, formula_str: str) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate training data from formula.
        
        Returns:
            x: Input tensor
            y: Output tensor
            weights: Optional sample weights (1.0 = normal, <1.0 = downweighted near singularities)
        """
        target_fn = parse_formula(formula_str)
        dtype = torch.float64 if self.config.precision == 64 else torch.float32
        
        # Generate x values, optionally avoiding singular points
        if self.config.sample_avoid:
            # Create x values that avoid singular points
            x_list = []
            avoid_points = self.config.sample_avoid
            eps = self.config.sample_epsilon
            
            # Build valid ranges
            sorted_avoid = sorted(avoid_points)
            ranges = []
            current_start = self.config.x_min
            
            for avoid_pt in sorted_avoid:
                if current_start < avoid_pt - eps:
                    ranges.append((current_start, avoid_pt - eps))
                current_start = avoid_pt + eps
            
            if current_start < self.config.x_max:
                ranges.append((current_start, self.config.x_max))
            
            # Calculate samples per range (proportional to range size)
            total_range = sum(r[1] - r[0] for r in ranges)
            if total_range > 0:
                for r_start, r_end in ranges:
                    n_in_range = max(1, int(self.config.n_samples * (r_end - r_start) / total_range))
                    x_list.append(torch.linspace(r_start, r_end, n_in_range, dtype=dtype))
                x = torch.cat(x_list).reshape(-1, 1)
            else:
                # Fallback if all ranges are invalid
                x = torch.linspace(self.config.x_min, self.config.x_max, self.config.n_samples, dtype=dtype).reshape(-1, 1)
        else:
            x = torch.linspace(self.config.x_min, self.config.x_max, self.config.n_samples, dtype=dtype).reshape(-1, 1)
        
        # Auto-domain filtering/shrinking if invalid values
        if self.config.auto_domain:
            x_try = x
            y = None
            for _ in range(4):
                y_try = target_fn(x_try)
                finite_mask = torch.isfinite(y_try).squeeze()
                if finite_mask.float().mean().item() > 0.9:
                    x = x_try[finite_mask].reshape(-1, 1)
                    y = y_try[finite_mask].reshape(-1, 1)
                    break
                mid = (x_try.min() + x_try.max()) / 2
                half = (x_try.max() - x_try.min()) / 2 * 0.75
                x_try = torch.linspace((mid - half).item(), (mid + half).item(), self.config.n_samples, dtype=dtype).reshape(-1, 1)
            if y is None:
                y_try = target_fn(x)
                finite_mask = torch.isfinite(y_try).squeeze()
                x = x[finite_mask].reshape(-1, 1)
                y = y_try[finite_mask].reshape(-1, 1)
        else:
            y = target_fn(x)
        if self.config.noise_std > 0:
            y = y + torch.randn_like(y) * self.config.noise_std
        
        # Compute sample weights if requested
        weights = None
        if self.config.use_sample_weights and self.config.sample_avoid:
            eps = self.config.sample_epsilon
            weights = torch.ones(x.shape[0], dtype=x.dtype)
            for avoid_pt in self.config.sample_avoid:
                # Weight = min(1, |x - avoid_pt| / eps)
                dist = torch.abs(x.squeeze() - avoid_pt)
                weights = torch.minimum(weights, torch.clamp(dist / eps, min=0.1, max=1.0))
        
        return x, y, weights
    
    def run_evolution(self, x: torch.Tensor, y: torch.Tensor, visualizer=None) -> Dict:
        """Run evolutionary training."""
        trainer = EvolutionaryONNTrainer(
            model_factory=self.make_model,
            population_size=self.config.population,
            elite_size=self.config.elite_size,
            mutation_rate=self.config.mutation_rate,
            constant_refine_steps=self.config.constant_refine_steps,
            complexity_penalty=0.01,
            device=self.device,
            lamarckian=self.config.lamarckian,
            use_explorers=True,
            explorer_fraction=self.config.explorer_fraction,
            explorer_mutation_rate=self.config.explorer_mutation,
            prune_coefficients=self.config.prune_coefficients,
            constant_refine_hard=True,
            visualizer=visualizer,
            risk_seeking=self.config.risk_seeking,
            risk_seeking_percentile=self.config.risk_seeking_percentile,
            use_curve_classifier=getattr(self.config, 'use_curve_classifier', False),
            normalize_data=self.config.normalize_data,
        )
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        return trainer.train(x, y, generations=self.config.generations, print_every=5)
    
    def run_phase2_regression(self, x: torch.Tensor, y: torch.Tensor, phase1_formula: str) -> Dict:
        """Phase 2: Pure basis regression for exact coefficients."""
        formula_lower = phase1_formula.lower()
        discovered_ops = set()
        
        if 'sin' in formula_lower or 'cos' in formula_lower:
            discovered_ops.add('periodic')
        if '^2' in formula_lower or '^3' in formula_lower or '**' in formula_lower:
            discovered_ops.add('power')
        # Detect x*sin(x) or x*cos(x) patterns
        if '*' in formula_lower and 'x' in formula_lower:
            if re.search(r'x\d?\s*\*\s*(sin|cos)', formula_lower) or \
               re.search(r'(sin|cos).*\*\s*x', formula_lower):
                discovered_ops.add('x_times_trig')
        
        features = [x]
        feature_names = ['x']
        
        if 'power' in discovered_ops or 'x_times_trig' in discovered_ops:
            features.append(x ** 2)
            feature_names.append('x²')
            features.append(x ** 3)
            feature_names.append('x³')
        
        if 'periodic' in discovered_ops:
            features.append(torch.sin(x))
            feature_names.append('sin(x)')
            features.append(torch.cos(x))
            feature_names.append('cos(x)')
        
        if 'x_times_trig' in discovered_ops:
            features.append(x * torch.sin(x))
            feature_names.append('x·sin(x)')
            features.append(x * torch.cos(x))
            feature_names.append('x·cos(x)')
        
        features_matrix = torch.cat(features, dim=1)
        n_samples = features_matrix.shape[0]
        features_with_bias = torch.cat([features_matrix, torch.ones(n_samples, 1, device=x.device)], dim=1)
        
        try:
            solution = torch.linalg.lstsq(features_with_bias, y)
            weights = solution.solution.squeeze()
            pred = features_with_bias @ weights
            mse = torch.nn.functional.mse_loss(pred.squeeze(), y.squeeze()).item()
            
            # Build formula string
            from glassbox.sr.meta_ops import get_constant_symbol
            formula_parts = []
            for name, w in zip(feature_names, weights[:-1]):
                w_val = w.item()
                if abs(w_val) > 0.01:
                    coef_sym = get_constant_symbol(w_val, threshold=0.05)
                    is_symbolic = coef_sym not in [str(int(round(w_val))), f"{w_val:.2g}", f"{w_val:.4g}"]
                    if abs(w_val - 1.0) < 0.02:
                        formula_parts.append(name)
                    elif abs(w_val + 1.0) < 0.02:
                        formula_parts.append(f"-{name}")
                    elif is_symbolic:
                        formula_parts.append(f"{coef_sym}*{name}")
                    else:
                        formula_parts.append(f"{w_val:.4f}*{name}")
            
            bias_val = weights[-1].item()
            if abs(bias_val) > 0.01:
                bias_sym = get_constant_symbol(bias_val, threshold=0.05)
                is_symbolic = bias_sym not in [str(int(round(bias_val))), f"{bias_val:.2g}", f"{bias_val:.4g}"]
                if is_symbolic:
                    formula_parts.append(bias_sym)
                else:
                    formula_parts.append(f"{bias_val:.4f}")
            
            formula_b = " + ".join(formula_parts) if formula_parts else "0"
            formula_b = formula_b.replace("+ -", "- ")
            
            return {'formula': formula_b, 'mse': mse, 'weights': weights, 'feature_names': feature_names}
        except Exception as e:
            return {'formula': 'ERROR', 'mse': float('inf'), 'error': str(e)}
    
    def run_gradient_refinement(self, model: OperationDAG, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, str]:
        """Gradient-based coefficient refinement."""
        return finalize_model_coefficients(
            model, x, y, 
            refine_internal_constants=self.config.refine_internal
        )
    
    def compare_with_mlp(self, x_train: torch.Tensor, y_train: torch.Tensor, 
                         x_val: torch.Tensor, y_val: torch.Tensor) -> Dict:
        """Train MLP baseline for comparison."""
        mlp = BaselineMLP(n_inputs=1, n_hidden=32, n_outputs=1).to(self.device)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
        
        for _ in range(300):
            mlp.train()
            optimizer.zero_grad()
            pred = mlp(x_train)
            loss = nn.functional.mse_loss(pred, y_train)
            loss.backward()
            optimizer.step()
        
        mlp.eval()
        with torch.no_grad():
            pred = mlp(x_val)
            mse = nn.functional.mse_loss(pred, y_val).item()
            corr = torch.corrcoef(torch.stack([
                pred.squeeze().cpu(), y_val.squeeze().cpu()
            ]))[0, 1].item()
        
        return {'mse': mse, 'corr': corr}


# =============================================================================
# MODE RUNNERS
# =============================================================================

def run_single_mode(config: Config):
    """Test a single formula."""
    start_time = time.time()
    
    print("=" * 70)
    print("SINGLE FORMULA TEST")
    print("=" * 70)
    
    tester = SRTester(config)
    print(f"Device: {tester.device}")
    print(f"Target formula: {config.formula}")
    print("-" * 70)
    
    # Generate data
    x, y, weights = tester.generate_data(config.formula)
    print(f"Data range: x in [{config.x_min:.1f}, {config.x_max:.1f}], {len(x)} samples")
    if weights is not None:
        print(f"Using sample weights (domain-aware)")
    
    # Setup visualizer
    visualizer = None
    if not config.no_viz:
        visualizer = LiveTrainingVisualizer(
            update_every=1,
            figsize=(14, 8),
            lite_mode=config.lite_mode,
        )
    
    # Try Fast Path first (classifier-guided direct regression)
    fast_path_result = None
    if getattr(config, 'use_curve_classifier', False):
        try:
            from scripts.classifier_fast_path import run_fast_path, run_guided_evolution
            from glassbox.sr.evolution import detect_dominant_frequency
            
            x_tensor = x.to(tester.device)
            y_tensor = y.to(tester.device)
            
            # Get FFT frequencies for better basis
            detected_omegas = detect_dominant_frequency(x_tensor, y_tensor, n_frequencies=3)
            
            # Pass operator constraints to fast path (honor --no-ops-* flags)
            op_constraints = None
            if any(x is not None for x in [
                config.ops_periodic, config.ops_power,
                config.ops_exp, config.ops_log,
                config.ops_arithmetic, config.ops_aggregation
            ]):
                op_constraints = {
                    'periodic': config.ops_periodic,
                    'power': config.ops_power,
                    'exp': config.ops_exp,
                    'log': config.ops_log,
                    'arithmetic': config.ops_arithmetic,
                    'aggregation': config.ops_aggregation,
                }
                op_constraints = {k: v for k, v in op_constraints.items() if v is not None}

            fast_path_result = run_fast_path(
                x_tensor, y_tensor,
                classifier_path="models/curve_classifier.pt",
                detected_omegas=detected_omegas,
                op_constraints=op_constraints,
                auto_expand=True,
            )
            
            if fast_path_result and fast_path_result['mse'] < 0.01:
                # Check if this is an EXACT symbolic match or approximation
                is_exact = fast_path_result.get('details', {}).get('exact_match', False)
                
                if is_exact:
                    # Fast path found exact symbolic match - we're done!
                    elapsed = time.time() - start_time
                    print("\n" + "=" * 70)
                    print("FINAL RESULTS (FAST PATH - EXACT MATCH)")
                    print("=" * 70)
                    print(f"TARGET:     {config.formula}")
                    print(f"DISCOVERED: {fast_path_result['formula']}")
                    print(f"FINAL MSE:  {fast_path_result['mse']:.6f}")
                    print()
                    print("=" * 70)
                    print(f"TOTAL TIME: {elapsed:.2f} seconds (FAST PATH)")
                    print("=" * 70)
                    return

                # Fast-path only: return after printing best fast-path result
                if config.fast_path_only:
                    print("\n" + "=" * 70)
                    print("FINAL RESULTS (FAST PATH ONLY)")
                    print("=" * 70)
                    print(f"TARGET:     {config.formula}")
                    print(f"DISCOVERED: {fast_path_result['formula']}")
                    print(f"FINAL MSE:  {fast_path_result['mse']:.6f}")
                    print("(Note: Fast-path only mode; evolution skipped)")
                    print("\n" + "=" * 70)
                    return

                # Fast path found approximation - try guided evolution for exact form
                print(f"\nFast path found APPROXIMATION (MSE={fast_path_result['mse']:.4f})")
                print("Attempting GUIDED EVOLUTION to find exact symbolic form...")
                
                operator_hints = fast_path_result.get('operator_hints', {})
                if operator_hints and operator_hints.get('operators'):
                    guided_result = run_guided_evolution(
                        x_tensor, y_tensor,
                        operator_hints,
                        generations=30,
                        population_size=25,
                        device=str(tester.device),
                        visualizer=visualizer,
                    )
                    
                    if guided_result and guided_result['mse'] < fast_path_result['mse']:
                        # Guided evolution found better result
                        elapsed = time.time() - start_time
                        print("\n" + "=" * 70)
                        print("FINAL RESULTS (GUIDED EVOLUTION)")
                        print("=" * 70)
                        print(f"TARGET:     {config.formula}")
                        print(f"FAST-PATH:  {fast_path_result['formula']}")
                        print(f"EVOLVED:    {guided_result['formula']}")
                        print(f"FINAL MSE:  {guided_result['mse']:.6f}")
                        print()
                        print("=" * 70)
                        print(f"TOTAL TIME: {elapsed:.2f} seconds (GUIDED)")
                        print("=" * 70)
                        return
                    else:
                        # Guided evolution didn't improve - use fast-path result
                        elapsed = time.time() - start_time
                        print("\n" + "=" * 70)
                        print("FINAL RESULTS (FAST PATH - APPROXIMATION)")
                        print("=" * 70)
                        print(f"TARGET:     {config.formula}")
                        print(f"DISCOVERED: {fast_path_result['formula']}")
                        print(f"FINAL MSE:  {fast_path_result['mse']:.6f}")
                        print("(Note: This is an approximation, not exact symbolic form)")
                        print()
                        print("=" * 70)
                        print(f"TOTAL TIME: {elapsed:.2f} seconds (FAST PATH)")
                        print("=" * 70)
                        return
                else:
                    # No operator hints - fall through to full evolution
                    print("No operator hints extracted, falling back to full evolution...")
            elif fast_path_result:
                print(f"\nFast path MSE ({fast_path_result['mse']:.4f}) > 0.01, falling back to evolution...")
        except Exception as e:
            import traceback
            print(f"Fast path error: {e}")
            traceback.print_exc()
            print("Falling back to evolution...")
    
    # Phase 1: Structure Discovery (if fast path didn't succeed)
    print("\n" + "=" * 70)
    print("PHASE 1: STRUCTURE DISCOVERY")
    print("=" * 70)
    
    results = tester.run_evolution(x, y, visualizer=visualizer)
    phase1_model = results['model']
    phase1_formula = results['formula']
    phase1_mse = results['final_mse']
    
    print(f"\\nPhase 1 Result: {sanitize_formula(phase1_formula)}")
    print(f"Phase 1 MSE: {phase1_mse:.4f}")
    
    # Phase 2: Coefficient Refinement (Hybrid)
    print("\\n" + "=" * 70)
    print("PHASE 2: HYBRID COEFFICIENT REFINEMENT")
    print("=" * 70)
    
    # Method A: Gradient
    print("\\n>>> Method A: Gradient Refinement...")
    model_a = copy.deepcopy(phase1_model)
    mse_a, formula_a = tester.run_gradient_refinement(model_a, x.to(tester.device), y.to(tester.device))
    print(f"Method A: MSE={mse_a:.6f}, Formula={sanitize_formula(formula_a)}")
    
    # Method B: Basis Regression
    print("\\n>>> Method B: Pure Basis Regression...")
    result_b = tester.run_phase2_regression(x.to(tester.device), y.to(tester.device), phase1_formula)
    mse_b = result_b['mse']
    formula_b = result_b['formula']
    print(f"Method B: MSE={mse_b:.6f}, Formula={sanitize_formula(formula_b)}")
    
    # Pick winner
    if math.isnan(mse_a): mse_a = float('inf')
    if math.isnan(mse_b): mse_b = float('inf')
    
    print("\n" + "-" * 70)
    if mse_b < mse_a:
        print(f"WINNER: Method B (Pure Basis) - MSE {mse_b:.6f}")
        final_mse, final_formula = mse_b, formula_b
    else:
        print(f"WINNER: Method A (Gradient) - MSE {mse_a:.6f}")
        final_mse, final_formula = mse_a, formula_a
    
    # Optional pruning
    if config.enable_pruning:
        print("\n" + "=" * 70)
        print("POST-TRAINING PRUNING")
        print("=" * 70)
        pruner = PostTrainingPruner(phase1_model, x.to(tester.device), y.to(tester.device))
        pruner.sensitivity_analysis(verbose=True)
        pruner.recursive_graph_prune(importance_threshold=0.01, verbose=True)
        print(f"Pruned formula: {sanitize_formula(pruner.get_formula())}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"TARGET:     {config.formula}")
    print(f"DISCOVERED: {sanitize_formula(final_formula)}")
    print(f"FINAL MSE:  {final_mse:.6f}")
    
    # MLP comparison
    if config.compare_mlp:
        print("\n--- MLP Baseline ---")
        mlp_result = tester.compare_with_mlp(
            x.to(tester.device), y.to(tester.device),
            x.to(tester.device), y.to(tester.device)
        )
        print(f"MLP MSE: {mlp_result['mse']:.6f}")
    
    if not config.no_viz:
        import matplotlib.pyplot as plt
        print("\nClose visualization window to exit.")
        plt.show(block=True)
    
    # Output total time
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"TOTAL TIME: {elapsed_time:.2f} seconds")
    print("=" * 70)
    
    return {'formula': final_formula, 'mse': final_mse, 'time_seconds': elapsed_time}


def run_evolution_mode(config: Config):
    """Run evolution benchmarks on preset formulas."""
    print("=" * 70)
    print("EVOLUTION BENCHMARK MODE")
    print("=" * 70)
    
    test_cases = [
        ('x^2', 'y = x²'),
        ('sin(x)', 'y = sin(x)'),
        ('x^3', 'y = x³'),
        ('x^2+x', 'y = x² + x'),
        ('sin(x)+x^2', 'y = sin(x) + x²'),
    ]
    
    tester = SRTester(config)
    results = []
    
    for formula, name in test_cases:
        print("\n" + "=" * 70)
        print(f"TASK: {name}")
        print("=" * 70)
        
        x, y, weights = tester.generate_data(formula)
        perm = torch.randperm(x.shape[0])
        x, y = x[perm], y[perm]
        if weights is not None:
            weights = weights[perm]
        
        n_train = int(0.8 * len(x))
        x_train, x_val = x[:n_train], x[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        x_train = x_train.to(tester.device)
        y_train = y_train.to(tester.device)
        x_val = x_val.to(tester.device)
        y_val = y_val.to(tester.device)
        
        result = tester.run_evolution(x_train, y_train)
        model = result['model']
        model.eval()
        
        with torch.no_grad():
            pred, _ = model(x_val, hard=True)
            val_mse = nn.functional.mse_loss(pred, y_val).item()
            val_corr = torch.corrcoef(torch.stack([
                pred.squeeze().cpu(), y_val.squeeze().cpu()
            ]))[0, 1].item()
        
        print(f"\nValidation MSE: {val_mse:.4f}, Corr: {val_corr:.4f}")
        print(f"Formula: {result['formula']}")
        
        results.append({
            'task': name,
            'mse': val_mse,
            'corr': val_corr,
            'formula': result['formula'],
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Task':<20} {'MSE':<12} {'Corr':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r['task']:<20} {r['mse']:<12.4f} {r['corr']:<10.4f}")
    
    return results


def run_pruning_mode(config: Config):
    """Run pruning pipeline test."""
    print("=" * 70)
    print("PRUNING PIPELINE TEST")
    print("=" * 70)
    
    tester = SRTester(config)
    formula = config.formula or "sin(x) + x^2"
    
    x, y, weights = tester.generate_data(formula)
    x = x.to(tester.device)
    y = y.to(tester.device)
    
    print(f"Target: {formula}")
    print("\n--- TRAINING ---")
    result = tester.run_evolution(x, y)
    pre_formula = result['formula']
    pre_mse = result['final_mse']
    
    print(f"\nPre-pruning: {pre_formula}")
    print(f"Pre-pruning MSE: {pre_mse:.6f}")
    
    print("\n--- PRUNING ---")
    pruner = PostTrainingPruner(result['model'], x, y)
    pruner.sensitivity_analysis(verbose=True)
    pruner.recursive_graph_prune(importance_threshold=0.01, verbose=True)
    pruner.mask_and_finetune(weight_threshold=0.05, verbose=True)
    
    post_formula = pruner.get_formula()
    post_mse = pruner.get_mse()
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Before: {pre_formula} (MSE: {pre_mse:.6f})")
    print(f"After:  {post_formula} (MSE: {post_mse:.6f})")
    
    return {'pre': pre_formula, 'post': post_formula}


def run_viz_mode(config: Config):
    """Visualization mode with phased regression."""
    config.formula = config.formula or "pi*sin(x) + x^2/pi"
    return run_single_mode(config)


# =============================================================================
# TUI (Rich-based)
# =============================================================================

def run_tui():
    """Run interactive TUI."""
    if not HAS_RICH:
        print("Rich library required for TUI. Install with: pip install rich")
        print("Falling back to CLI mode.")
        formula = input("Enter formula: ").strip()
        config = Config()
        config.formula = formula
        config.mode = "single"
        return run_single_mode(config)
    
    console.print(Panel.fit(
        "[bold cyan]🔬 GLASSBOX SR TESTER[/bold cyan]\n"
        "[dim]Unified Symbolic Regression Testing Tool[/dim]",
        border_style="cyan"
    ))
    
    config = Config()
    
    # Mode selection
    mode_table = Table(show_header=False, box=None)
    mode_table.add_row("[1]", "single", "Test a single formula")
    mode_table.add_row("[2]", "viz", "Visualization with phased regression")
    mode_table.add_row("[3]", "evolution", "Benchmark on preset formulas")
    mode_table.add_row("[4]", "pruning", "Post-training pruning test")
    
    console.print("\n[bold]Select Mode:[/bold]")
    console.print(mode_table)
    
    mode_choice = Prompt.ask("Enter choice", choices=["1", "2", "3", "4"], default="1")
    mode_map = {"1": "single", "2": "viz", "3": "evolution", "4": "pruning"}
    config.mode = mode_map[mode_choice]
    
    # Formula input for single/viz/pruning
    if config.mode in ["single", "viz", "pruning"]:
        default_formula = "sin(x) + x^2" if config.mode != "viz" else "pi*sin(x) + x^2/pi"
        config.formula = Prompt.ask("Target formula", default=default_formula)
    
    # Architecture settings
    if Confirm.ask("\nCustomize architecture?", default=False):
        config.hidden_layers = IntPrompt.ask("Hidden layers", default=config.hidden_layers)
        config.nodes_per_layer = IntPrompt.ask("Nodes per layer", default=config.nodes_per_layer)
        config.simplified_ops = Confirm.ask("Simplified ops (no exp/log)?", default=config.simplified_ops)
        config.fair_mode = Confirm.ask("FairDARTS mode?", default=config.fair_mode)
    
    # Training settings
    if Confirm.ask("Customize training?", default=False):
        config.generations = IntPrompt.ask("Generations", default=config.generations)
        config.population = IntPrompt.ask("Population size", default=config.population)
        config.mutation_rate = FloatPrompt.ask("Mutation rate", default=config.mutation_rate)
        config.explorer_fraction = FloatPrompt.ask("Explorer fraction", default=config.explorer_fraction)
        config.refine_internal = Confirm.ask("Refine internal constants (omega)?", default=config.refine_internal)
    
    # Other options
    config.lite_mode = Confirm.ask("Lite mode (no network diagram)?", default=False)
    config.compare_mlp = Confirm.ask("Compare with MLP baseline?", default=False)
    
    # Show config summary
    console.print("\n[bold]Configuration:[/bold]")
    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_row("Mode", config.mode)
    if config.formula:
        summary_table.add_row("Formula", config.formula)
    summary_table.add_row("Architecture", f"{config.hidden_layers} layers × {config.nodes_per_layer} nodes")
    summary_table.add_row("Training", f"{config.generations} gen, pop={config.population}")
    console.print(summary_table)
    
    if not Confirm.ask("\nStart training?", default=True):
        console.print("[yellow]Cancelled.[/yellow]")
        return
    
    console.print("\n[bold green]Starting...[/bold green]\n")
    
    # Run selected mode
    if config.mode == "single":
        return run_single_mode(config)
    elif config.mode == "viz":
        return run_viz_mode(config)
    elif config.mode == "evolution":
        return run_evolution_mode(config)
    elif config.mode == "pruning":
        return run_pruning_mode(config)


# =============================================================================
# CLI ARGUMENT PARSER
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Glassbox SR Tester - Unified Symbolic Regression Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sr_tester.py                                    # Launch TUI
  python sr_tester.py --mode single --formula "x^2"      # Test x²
  python sr_tester.py --mode viz --formula "sin(x)+x^2"  # Visualize
  python sr_tester.py --mode evolution                   # Run benchmarks
  python sr_tester.py --mode pruning                     # Pruning test
        """
    )
    
    # Mode
    parser.add_argument('--mode', '-m', choices=['single', 'viz', 'evolution', 'pruning', 'interactive'],
                        default=None, help='Test mode')
    parser.add_argument('--formula', '-f', type=str, default='', help='Target formula to test')
    
    # Architecture
    arch = parser.add_argument_group('Model Architecture')
    arch.add_argument('--hidden-layers', type=int, default=2, help='Number of hidden layers (1-4)')
    arch.add_argument('--nodes-per-layer', type=int, default=4, help='Nodes per layer (2-12)')
    arch.add_argument('--simplified-ops', dest='simplified_ops', action='store_true', default=True,
                      help='Use simplified ops (power/periodic/arithmetic only)')
    arch.add_argument('--full-ops', dest='simplified_ops', action='store_false',
                      help='Use full operation set (includes exp/log)')
    arch.add_argument('--fair-mode', dest='fair_mode', action='store_true', default=True,
                      help='Enable FairDARTS independent sigmoids')
    arch.add_argument('--no-fair-mode', dest='fair_mode', action='store_false')
    
    # Operator Constraints
    ops = parser.add_argument_group('Operator Constraints (override defaults)')
    ops.add_argument('--ops-periodic', dest='ops_periodic', action='store_true', default=None,
                     help='Enable periodic ops (sin, cos)')
    ops.add_argument('--no-ops-periodic', dest='ops_periodic', action='store_false',
                     help='Disable periodic ops')
    ops.add_argument('--ops-power', dest='ops_power', action='store_true', default=None,
                     help='Enable power ops (x^n, sqrt)')
    ops.add_argument('--no-ops-power', dest='ops_power', action='store_false',
                     help='Disable power ops')
    ops.add_argument('--ops-exp', dest='ops_exp', action='store_true', default=None,
                     help='Enable exp op')
    ops.add_argument('--no-ops-exp', dest='ops_exp', action='store_false',
                     help='Disable exp op')
    ops.add_argument('--ops-log', dest='ops_log', action='store_true', default=None,
                     help='Enable log op')
    ops.add_argument('--no-ops-log', dest='ops_log', action='store_false',
                     help='Disable log op')
    ops.add_argument('--ops-arithmetic', dest='ops_arithmetic', action='store_true', default=None,
                     help='Enable arithmetic ops (+, *)')
    ops.add_argument('--no-ops-arithmetic', dest='ops_arithmetic', action='store_false',
                     help='Disable arithmetic ops')
    ops.add_argument('--ops-aggregation', dest='ops_aggregation', action='store_true', default=None,
                     help='Enable aggregation ops (sum, mean, max)')
    ops.add_argument('--no-ops-aggregation', dest='ops_aggregation', action='store_false',
                     help='Disable aggregation ops')

    # Fast Path Control
    fp = parser.add_argument_group('Fast Path Control')
    fp.add_argument('--fast-path-only', action='store_true', default=False,
                    help='Only run fast-path; do not run evolution fallback')
    
    # Evolution
    evo = parser.add_argument_group('Evolution Hyperparameters')
    evo.add_argument('--population', '-p', type=int, default=30, help='Population size (10-100)')
    evo.add_argument('--generations', '-g', type=int, default=40, help='Number of generations (10-200)')
    evo.add_argument('--elite-size', type=int, default=4, help='Elite individuals to preserve')
    evo.add_argument('--mutation-rate', type=float, default=0.5, help='Mutation probability (0.1-0.9)')
    evo.add_argument('--explorer-fraction', type=float, default=0.40, help='Fraction of explorers (0-0.6)')
    evo.add_argument('--explorer-mutation', type=float, default=0.85, help='Explorer mutation rate')
    evo.add_argument('--constant-refine-steps', type=int, default=50, help='Gradient steps for constants')
    evo.add_argument('--refine-omega', dest='refine_omega', action='store_true', default=True,
                     help='Enable omega (frequency) optimization')
    evo.add_argument('--no-refine-omega', dest='refine_omega', action='store_false')
    evo.add_argument('--refine-internal', dest='refine_internal', action='store_true', default=True,
                     help='Enable internal constant refinement')
    evo.add_argument('--no-refine-internal', dest='refine_internal', action='store_false')
    evo.add_argument('--lamarckian', dest='lamarckian', action='store_true', default=True,
                     help='Use Lamarckian evolution')
    evo.add_argument('--no-lamarckian', dest='lamarckian', action='store_false')
    evo.add_argument('--risk-seeking', dest='risk_seeking', action='store_true', default=True,
                     help='Enable risk-seeking policy gradient for stuck detection')
    evo.add_argument('--no-risk-seeking', dest='risk_seeking', action='store_false')
    evo.add_argument('--curve-classifier', dest='use_curve_classifier', action='store_true', default=False,
                     help='Use curve classifier to warm-start operator selection')
    evo.add_argument('--no-curve-classifier', dest='use_curve_classifier', action='store_false')
    
    # Data
    data = parser.add_argument_group('Data Generation')
    data.add_argument('--x-min', type=float, default=-6.0, help='Minimum x value')
    data.add_argument('--x-max', type=float, default=6.0, help='Maximum x value')
    data.add_argument('--n-samples', type=int, default=300, help='Number of data points')
    data.add_argument('--noise-std', type=float, default=0.0, help='Noise standard deviation')
    data.add_argument('--precision', type=int, choices=[32, 64], default=32,
                      help='Numeric precision for data (32 or 64)')
    data.add_argument('--normalize-data', dest='normalize_data', action='store_true', default=False,
                      help='Normalize data before evolution training')
    data.add_argument('--no-normalize-data', dest='normalize_data', action='store_false')
    data.add_argument('--auto-domain', dest='auto_domain', action='store_true', default=True,
                      help='Auto-shrink domain if formula invalid in range')
    data.add_argument('--no-auto-domain', dest='auto_domain', action='store_false')
    
    # Domain-Aware Sampling
    domain = parser.add_argument_group('Domain-Aware Sampling')
    domain.add_argument('--sample-avoid', type=float, nargs='*', default=None,
                        help='Singular points to avoid (e.g., --sample-avoid 1 -1 for 1/(1-x^2))')
    domain.add_argument('--sample-epsilon', type=float, default=0.1,
                        help='Distance to stay from singular points (default: 0.1)')
    domain.add_argument('--use-sample-weights', action='store_true', default=False,
                        help='Use sample weights to downweight points near singularities')
    
    # Other
    other = parser.add_argument_group('Other Options')
    other.add_argument('--lite', action='store_true', help='Lite mode (no network diagram)')
    other.add_argument('--no-viz', action='store_true', help='Disable visualization')
    other.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto', help='Device')
    other.add_argument('--seed', type=int, default=None, help='Random seed')
    other.add_argument('--compare-mlp', action='store_true', help='Compare with MLP baseline')
    other.add_argument('--pruning', action='store_true', help='Enable post-training pruning')
    
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # If no mode specified, launch TUI
    if args.mode is None:
        run_tui()
        return
    
    # Build config from args
    config = Config()
    config.mode = args.mode
    config.formula = args.formula
    config.hidden_layers = args.hidden_layers
    config.nodes_per_layer = args.nodes_per_layer
    config.simplified_ops = args.simplified_ops
    config.fair_mode = args.fair_mode
    config.population = args.population
    config.generations = args.generations
    config.elite_size = args.elite_size
    config.mutation_rate = args.mutation_rate
    config.explorer_fraction = args.explorer_fraction
    config.explorer_mutation = args.explorer_mutation
    config.constant_refine_steps = args.constant_refine_steps
    config.refine_omega = args.refine_omega
    config.refine_internal = args.refine_internal
    config.lamarckian = args.lamarckian
    config.risk_seeking = args.risk_seeking
    config.use_curve_classifier = args.use_curve_classifier
    config.x_min = args.x_min
    config.x_max = args.x_max
    config.n_samples = args.n_samples
    config.noise_std = args.noise_std
    config.precision = args.precision
    config.normalize_data = args.normalize_data
    config.auto_domain = args.auto_domain
    config.lite_mode = args.lite
    config.no_viz = args.no_viz
    config.device = args.device
    config.seed = args.seed
    config.compare_mlp = args.compare_mlp
    config.enable_pruning = args.pruning
    
    # Operator constraints
    config.ops_periodic = args.ops_periodic
    config.ops_power = args.ops_power
    config.ops_exp = args.ops_exp
    config.ops_log = args.ops_log
    config.ops_arithmetic = args.ops_arithmetic
    config.ops_aggregation = args.ops_aggregation
    
    # Domain-aware sampling
    config.sample_avoid = args.sample_avoid
    config.sample_epsilon = args.sample_epsilon
    config.use_sample_weights = args.use_sample_weights

    # Fast path only
    config.fast_path_only = args.fast_path_only
    
    # Run selected mode
    if config.mode == "single":
        if not config.formula:
            print("Error: --formula required for single mode")
            return 1
        run_single_mode(config)
    elif config.mode == "viz":
        run_viz_mode(config)
    elif config.mode == "evolution":
        run_evolution_mode(config)
    elif config.mode == "pruning":
        run_pruning_mode(config)
    elif config.mode == "interactive":
        run_tui()


if __name__ == "__main__":
    main()
