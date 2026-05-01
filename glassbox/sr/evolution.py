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
import logging
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np


DEFAULT_CURVE_CLASSIFIER_PATH = "models/curve_classifier_v3.1.pt"

# Configure module-level logger
logger = logging.getLogger(__name__)

# ============================================================================
# Constants (replacing magic numbers for maintainability)
# ============================================================================

# Population initialization
DEFAULT_BIAS_STRENGTH = 2.0
EXPLORER_BIAS_STRENGTH = 3.0
BIAS_INCREMENT = 0.1
EXPLORER_BIAS_INCREMENT = 0.2

# Mutation and selection
MUTATION_LOGIT_STRENGTH = 3.0
MUTATION_NOISE_SCALE = 0.5
LAMARCKIAN_NOISE_SCALE = 0.5

# Optimization
OUTPUT_PROJ_LR = 0.05
SCALE_PARAMS_LR = 0.001
GENTLE_REFINEMENT_LR = 0.005
POST_LOCK_LR = 0.02
GRADIENT_CLIP_NORM = 1.0
SCALE_GRADIENT_CLIP_NORM = 0.1

# Temperature and annealing
MIN_EXPLORER_TAU = 0.5
STRUCTURE_LOCK_CORR_THRESHOLD = 0.995
FINAL_STRUCTURE_CHECK_CORR = 0.99

# Pruning
FINAL_PRUNE_RATIO = 0.01
FINAL_PRUNE_THRESHOLD_RATIO = 0.08
FINAL_PRUNE_ABSOLUTE = 0.05
PRUNE_MSE_TOLERANCE = 0.02

# Refinement
POST_LOCK_REFINEMENT_STEPS = 200
GENTLE_REFINEMENT_STEPS = 200
MSE_IMPROVEMENT_THRESHOLD = 1e-8
MSE_DEGRADATION_TOLERANCE = 1.1
TARGET_MSE_THRESHOLD = 0.01

# Risk-Seeking Policy Gradient (RSPG)
try:
    from .risk_seeking_policy_gradient import (
        RiskSeekingEvolutionMixin,
        GradientMonitor,
        compute_risk_seeking_fitness,
        compute_selection_probabilities_rspg,
    )
    RSPG_AVAILABLE = True
except ImportError:
    RSPG_AVAILABLE = False
    RiskSeekingEvolutionMixin = object  # Dummy base class


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


def detect_dominant_frequency(
    x: torch.Tensor, 
    y: torch.Tensor,
    n_frequencies: int = 3,
    return_phase_info: bool = False,
) -> 'List[float] | Dict[str, Any]':
    """
    Use FFT to detect dominant frequencies in the target data.
    
    This helps seed omega values for individuals, enabling faster
    discovery of formulas like sin(3.2*x).
    
    Args:
        x: Input tensor (N,) or (N, 1)
        y: Target tensor (N,) or (N, 1)
        n_frequencies: Number of top frequencies to return
        return_phase_info: If True, return a dict with omegas, phases,
            and harmonic-consistency metadata instead of a plain list.
        
    Returns:
        List of detected omega values (angular frequencies), or if
        return_phase_info is True a dict with keys:
            omegas: List[float]
            phases: List[float]  (radians, range [-π, π])
            harmonic_fundamental: Optional[float]
            is_harmonic_series: bool
    """
    _fallback_omegas: List[float] = [1.0]
    _fallback_info: Dict[str, Any] = {
        'omegas': [1.0],
        'phases': [0.0],
        'harmonic_fundamental': None,
        'is_harmonic_series': False,
    }

    try:
        x_np = x.squeeze().cpu().numpy()
        y_np = y.squeeze().cpu().numpy()
        
        # Sort by x for interpolation
        sort_idx = np.argsort(x_np)
        x_sorted = x_np[sort_idx]
        y_sorted = y_np[sort_idx]
        
        # Interpolate to uniform grid for FFT
        n_samples = len(x_sorted)
        x_min, x_max = x_sorted.min(), x_sorted.max()
        x_range = x_max - x_min
        
        if x_range < 1e-6:
            return _fallback_info if return_phase_info else _fallback_omegas

        # Use refined sampling for better resolution (2x original points if possible)
        n_interp = max(n_samples, 1024)
        x_uniform = np.linspace(x_min, x_max, n_interp)
        y_uniform = np.interp(x_uniform, x_sorted, y_sorted)
        
        # Remove DC offset (mean)
        y_centered = y_uniform - np.mean(y_uniform)
        
        # Compute FFT
        fft_result = np.fft.rfft(y_centered)
        
        # Sample spacing
        dx = x_range / (n_interp - 1)
        freqs = np.fft.rfftfreq(n_interp, dx)
        
        # Get magnitudes and phases (skip DC component at index 0)
        magnitudes = np.abs(fft_result[1:])
        phase_angles = np.angle(fft_result[1:])  # Preserve phase spectrum
        freqs = freqs[1:]
        
        if len(magnitudes) == 0:
            return _fallback_info if return_phase_info else _fallback_omegas
        
        # Find top N frequency peaks
        top_indices = np.argsort(magnitudes)[-n_frequencies:][::-1]
        
        # Convert frequencies to angular frequencies (omega = 2π*f)
        detected_omegas: List[float] = []
        detected_phases: List[float] = []
        for idx in top_indices:
            if idx < len(freqs):
                freq = freqs[idx]
                omega = 2 * np.pi * freq
                # Only accept reasonable omega values
                if 0.1 < omega < 50.0:
                    detected_omegas.append(float(omega))
                    detected_phases.append(float(phase_angles[idx]))
        
        # Ensure we return at least one value
        if not detected_omegas:
            detected_omegas = [1.0]
            detected_phases = [0.0]

        if not return_phase_info:
            return detected_omegas

        # ── Harmonic consistency check ──
        # Check if detected peaks are integer multiples of a fundamental.
        # This flags likely square / triangle wave targets (sum-of-harmonics)
        # vs. single-frequency targets.
        harmonic_fundamental: Optional[float] = None
        is_harmonic_series = False

        if len(detected_omegas) >= 2:
            candidate_fund = detected_omegas[0]  # strongest peak
            if candidate_fund > 0.15:
                ratios = [omega / candidate_fund for omega in detected_omegas[1:]]
                int_ratios = [abs(r - round(r)) < 0.12 and round(r) >= 2 for r in ratios]
                if sum(int_ratios) >= max(1, len(ratios) // 2):
                    harmonic_fundamental = candidate_fund
                    is_harmonic_series = True

        return {
            'omegas': detected_omegas,
            'phases': detected_phases,
            'harmonic_fundamental': harmonic_fundamental,
            'is_harmonic_series': is_harmonic_series,
        }
        
    except Exception as e:
        logger.debug(f"FFT frequency detection failed: {e}")
        return _fallback_info if return_phase_info else _fallback_omegas


def seed_omega_from_fft(
    model: nn.Module,
    detected_omegas: List[float],
    individual_idx: int = 0,
):
    """
    Seed omega parameters in a model with FFT-detected frequencies.
    
    Different individuals get different detected frequencies for diversity.
    
    Args:
        model: The model to seed
        detected_omegas: List of detected omega values from FFT
        individual_idx: Which individual this is (for diversity)
    """
    if not detected_omegas:
        return
        
    # Pick which omega to use based on individual index
    omega_to_use = detected_omegas[individual_idx % len(detected_omegas)]
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'omega' in name:
                # 75% chance to use detected frequency (aggressive seeding)
                if random.random() < 0.75:
                    param.fill_(omega_to_use)
                # Otherwise keep the random initialization


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
    
    # Get device from model parameters to avoid device mismatch
    device = next((p.device for p in model.parameters()), torch.device('cpu'))
    loss = torch.tensor(0.0, device=device)
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
        self.raw_mse = float('inf')
        self.complexity = float('inf')
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
        ind.raw_mse = self.raw_mse
        ind.complexity = self.complexity
        ind.structure_hash = self.structure_hash
        return ind

    def refresh_structure_hash(self) -> None:
        """Recompute the discrete-structure hash after mutations."""
        self.structure_hash = self._compute_structure_hash()


class StructureConfidenceTracker:
    """
    Graduated confidence-based structure locking.
    
    Instead of a binary "locked/unlocked" state, this tracks confidence
    in the current structure based on multiple criteria:
    
    1. Correlation stability - high correlation maintained over generations
    2. MSE quality - low MSE indicates good fit
    3. Improvement trend - still improving or plateaued
    4. Validation consistency - if validation data available
    
    The confidence score (0.0 to 1.0) is used to:
    - Gradually reduce mutation rate (more refinement, less exploration)
    - Switch to refinement-only mode at very high confidence
    - Provide escape hatch if confidence drops (bad lock detected)
    
    Benefits over binary lock:
    - Smooth transition from exploration to exploitation
    - Can partially recover from premature locking
    - Uses multiple criteria, not just correlation
    """
    
    # Constants for confidence calculation
    CORR_THRESHOLD_LOW = 0.95       # Start gaining confidence
    CORR_THRESHOLD_HIGH = 0.995     # High confidence threshold
    MSE_THRESHOLD_GOOD = 0.01       # Good MSE (adjust per problem)
    STABILITY_GENERATIONS = 3       # Generations to be stable for lock
    REFINEMENT_ONLY_CONFIDENCE = 0.95  # Confidence to switch to refinement-only
    MIN_MUTATION_FRACTION = 0.1     # Never reduce mutation below 10%
    
    def __init__(
        self,
        mse_threshold: float = 0.01,
        stability_generations: int = 3,
        min_mutation_fraction: float = 0.1,
    ):
        """
        Args:
            mse_threshold: MSE below this is considered "good"
            stability_generations: Consecutive good generations needed for high confidence
            min_mutation_fraction: Minimum mutation rate fraction (0.1 = 10% of original)
        """
        self.mse_threshold = mse_threshold
        self.stability_generations = stability_generations
        self.min_mutation_fraction = min_mutation_fraction
        
        # Tracking state
        self.history: List[Dict[str, float]] = []
        self.confidence = 0.0
        self.best_mse_seen = float('inf')
        self.generations_stable = 0
        self.last_improvement_gen = 0
        
        # Escape hatch state
        self.refinement_mse_before = None
        self.refinement_failures = 0
    
    def update(
        self,
        generation: int,
        correlation: float,
        mse: float,
        validation_corr: Optional[float] = None,
    ) -> float:
        """
        Update confidence based on current generation metrics.
        
        Args:
            generation: Current generation number
            correlation: Training correlation
            mse: Training MSE
            validation_corr: Validation correlation (optional)
        
        Returns:
            Updated confidence score (0.0 to 1.0)
        """
        # Track history
        self.history.append({
            'gen': generation,
            'corr': correlation,
            'mse': mse,
            'val_corr': validation_corr,
        })
        
        # Track improvement
        if mse < self.best_mse_seen * 0.99:  # 1% improvement
            self.best_mse_seen = mse
            self.last_improvement_gen = generation
        
        # Compute individual confidence components
        corr_confidence = self._correlation_confidence(correlation)
        mse_confidence = self._mse_confidence(mse)
        stability_confidence = self._stability_confidence(generation)
        
        # Validation bonus (if available)
        val_bonus = 0.0
        if validation_corr is not None and validation_corr > 0.98:
            val_bonus = 0.1  # Extra confidence if validation is good
        
        # Combined confidence (weighted average)
        # Correlation is most important, MSE and stability are bonuses
        self.confidence = (
            0.5 * corr_confidence +
            0.25 * mse_confidence +
            0.25 * stability_confidence +
            val_bonus
        )
        self.confidence = min(1.0, self.confidence)  # Cap at 1.0
        
        # Update stability counter
        if correlation > self.CORR_THRESHOLD_HIGH and mse < self.mse_threshold:
            self.generations_stable += 1
        else:
            self.generations_stable = max(0, self.generations_stable - 1)
        
        return self.confidence
    
    def _correlation_confidence(self, corr: float) -> float:
        """Map correlation to confidence. 0.95-0.995 maps to 0.0-1.0."""
        if corr < self.CORR_THRESHOLD_LOW:
            return 0.0
        if corr >= self.CORR_THRESHOLD_HIGH:
            return 1.0
        # Linear interpolation
        return (corr - self.CORR_THRESHOLD_LOW) / (self.CORR_THRESHOLD_HIGH - self.CORR_THRESHOLD_LOW)
    
    def _mse_confidence(self, mse: float) -> float:
        """Map MSE to confidence. Lower is better."""
        if mse >= self.mse_threshold * 10:
            return 0.0
        if mse <= self.mse_threshold:
            return 1.0
        # Log scale for MSE (since it varies by orders of magnitude)
        log_mse = math.log10(mse + 1e-10)
        log_threshold = math.log10(self.mse_threshold)
        log_high = math.log10(self.mse_threshold * 10)
        return (log_high - log_mse) / (log_high - log_threshold)
    
    def _stability_confidence(self, generation: int) -> float:
        """Confidence from stability (consecutive good generations)."""
        if self.generations_stable >= self.stability_generations:
            return 1.0
        return self.generations_stable / self.stability_generations
    
    def get_effective_mutation_rate(self, base_mutation_rate: float) -> float:
        """
        Get effective mutation rate based on confidence.
        
        High confidence = low mutation (more refinement)
        Low confidence = normal mutation (more exploration)
        """
        # Scale mutation: 1.0 at confidence=0, min_mutation_fraction at confidence=1
        scale = 1.0 - self.confidence * (1.0 - self.min_mutation_fraction)
        return base_mutation_rate * scale
    
    def should_refine_only(self) -> bool:
        """Check if we should switch to refinement-only mode (no mutations)."""
        return (
            self.confidence >= self.REFINEMENT_ONLY_CONFIDENCE and
            self.generations_stable >= self.stability_generations
        )
    
    def start_refinement_tracking(self, current_mse: float):
        """Call before intensive refinement to track if it helps."""
        self.refinement_mse_before = current_mse
    
    def end_refinement_tracking(self, new_mse: float) -> bool:
        """
        Call after intensive refinement. Returns True if refinement helped.
        
        If refinement makes things worse, reduces confidence (escape hatch).
        """
        if self.refinement_mse_before is None:
            return True
        
        improvement = self.refinement_mse_before - new_mse
        improvement_pct = improvement / (self.refinement_mse_before + 1e-10)
        
        if improvement_pct < -0.2:  # MSE got 20% worse
            # Refinement hurt - reduce confidence
            self.refinement_failures += 1
            self.confidence *= 0.5  # Halve confidence
            self.generations_stable = 0
            return False
        
        self.refinement_mse_before = None
        return True
    
    def get_status(self) -> str:
        """Get human-readable status string."""
        if self.should_refine_only():
            return f"REFINEMENT-ONLY (conf={self.confidence:.2f}, stable={self.generations_stable})"
        elif self.confidence > 0.5:
            return f"HIGH-CONFIDENCE (conf={self.confidence:.2f}, stable={self.generations_stable})"
        elif self.confidence > 0.2:
            return f"BUILDING-CONFIDENCE (conf={self.confidence:.2f})"
        else:
            return f"EXPLORING (conf={self.confidence:.2f})"


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
                # Random frequency: 0.3 to 6.0 (wide range for frequency discovery)
                param.fill_(random.uniform(0.3, 6.0))
            elif 'beta' in name and 'selector' not in name:
                # Random arithmetic blend
                param.fill_(random.uniform(0.5, 2.5))
            elif 'R' in name:
                # Random routing
                param.normal_(0, 1.0)


def seed_population_from_classifier(
    model: nn.Module,
    predictions: Dict[str, float],
    detected_omegas: Optional[List[float]] = None,
    bias_strength: float = 3.0,
    individual_idx: int = 0,
    seed_fraction: float = 0.9,
):
    """
    Probabilistically bias model toward operators predicted by curve classifier.
    
    Key insight: Instead of biasing ALL individuals the same way, we use
    the classifier probabilities to decide PER-INDIVIDUAL whether to bias.
    
    If sin=0.95, then ~95% of individuals start biased toward sin.
    If exp=0.10, then only ~10% get exp-biased.
    
    This maintains diversity while focusing search on likely operators.
    
    Args:
        model: The ONN model to seed
        predictions: Dict from predict_operators() {op_name: probability}
        detected_omegas: FFT-detected frequencies for omega seeding
        bias_strength: How strongly to bias logits (higher = more deterministic)
        individual_idx: Index of this individual (for omega diversity)
        seed_fraction: Overall fraction of population to seed (default 90%)
    """
    # First check if we should seed this individual at all
    if random.random() > seed_fraction:
        # Leave as random initialization for diversity
        return
    
    # Mapping from classifier classes to ONN meta-op indices
    # For simplified_ops=True: n_unary=2 [MetaPeriodic(0), MetaPower(1)]
    # For simplified_ops=False: n_unary=4 [MetaPeriodic(0), MetaPower(1), MetaExp(2), MetaLog(3)]
    simplified = getattr(model, 'simplified_ops', True)
    
    if simplified:
        unary_map = {
            'sin': 0, 'cos': 0, 'periodic': 0,  # MetaPeriodic
            'power': 1, 'identity': 1, 'polynomial': 1, 'rational': 1,  # MetaPower
        }
    else:
        unary_map = {
            'sin': 0, 'cos': 0, 'periodic': 0,  # MetaPeriodic
            'power': 1, 'identity': 1, 'polynomial': 1, 'rational': 1,  # MetaPower
            'exp': 2, 'exponential': 2,  # MetaExp
            'log': 3,  # MetaLog
        }
    
    with torch.no_grad():
        # Bias operation selectors based on probabilistic sampling
        for layer in model.layers:
            for node in layer.nodes:
                if not hasattr(node, 'op_selector'):
                    continue
                
                selector = node.op_selector
                
                # HardConcreteOperationSelector layout: [type(2), unary(n), binary(m)]
                if hasattr(selector, 'logits') and hasattr(selector, '_type_end'):
                    # For each predicted operator, sample whether to bias this node
                    for op_name, prob in predictions.items():
                        if op_name in unary_map:
                            # Probabilistic: bias node if random < probability
                            if random.random() < prob:
                                unary_idx = unary_map[op_name]
                                logit_idx = 2 + unary_idx  # Skip 2 type logits
                                if logit_idx < len(selector.logits):
                                    selector.logits.data[logit_idx] += bias_strength
                                    # Also bias toward unary type (index 0)
                                    selector.logits.data[0] += bias_strength * 0.5
        
        # Seed omega parameters with FFT-detected frequencies
        if detected_omegas:
            seed_omega_from_fft(model, detected_omegas, individual_idx)


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
                    # Mutate frequency - discrete jumps or continuous perturbation
                    if random.random() < 0.3:
                        # Discrete jump to common frequencies (expanded list)
                        param.fill_(random.choice([0.5, 1.0, 2.0, math.pi, 3.0, 3.5, 4.0, 5.0, 6.0]))
                    else:
                        # Continuous perturbation with wider range
                        param.add_(random.uniform(-0.8, 0.8))
                        param.clamp_(0.1, 8.0)
                
                elif 'R' in name:
                    # Mutate routing
                    if random.random() < 0.3:
                        # Swap routing connections
                        param.add_(torch.randn_like(param) * 0.5)
    
    mutant.refresh_structure_hash()
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
    
    mutant.refresh_structure_hash()
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
    except Exception as e:
        logger.debug(f"compute_param_sensitivity failed: {e}")
        return {}
    
    sensitivities = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            sensitivities[name] = param.grad.abs().mean().item()
        else:
            sensitivities[name] = 0.0
    
    model.zero_grad()
    return sensitivities


def normalize_param_sensitivities(sensitivities: Dict[str, float]) -> Dict[str, float]:
    """Normalize raw sensitivities to [0, 1] for mutation scheduling."""
    if not sensitivities:
        return {}

    max_sens = max(sensitivities.values())
    if max_sens <= 0:
        return {name: 0.0 for name in sensitivities}

    return {name: min(1.0, max(0.0, value / max_sens)) for name, value in sensitivities.items()}


def _mutation_profile_from_sensitivity(
    sensitivity: float,
    mutation_rate: float,
    sensitivity_bias: float,
) -> Tuple[float, float, bool]:
    """Map sensitivity to mutation rate, noise scale, and freeze flag."""
    sensitivity = min(1.0, max(0.0, sensitivity))

    # High-sensitivity parameters should be protected aggressively.
    adjusted_rate = mutation_rate * (1.0 - sensitivity_bias * sensitivity)
    adjusted_rate = max(0.05 * mutation_rate, min(1.0, adjusted_rate))

    # Low-sensitivity parameters may absorb stronger perturbations.
    noise_scale = 0.5 * (1.0 + (1.0 - sensitivity) * sensitivity_bias)
    freeze = sensitivity >= 0.85
    return adjusted_rate, noise_scale, freeze


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
    
    norm_sens = normalize_param_sensitivities(sensitivities)
    
    mutant = individual.clone()
    
    with torch.no_grad():
        for name, param in mutant.model.named_parameters():
            if 'logit' in name or 'selector' in name:
                # Compute adjusted mutation rate
                sens = norm_sens.get(name, 0.5)
                adjusted_rate, noise_scale, freeze = _mutation_profile_from_sensitivity(
                    sens,
                    mutation_rate,
                    sensitivity_bias,
                )

                if freeze:
                    continue
                
                if random.random() < adjusted_rate:
                    if random.random() < 0.5:
                        param.zero_()
                        new_choice = random.randint(0, param.numel() - 1)
                        param.view(-1)[new_choice] = 3.0
                    else:
                        param.add_(torch.randn_like(param) * noise_scale)
    
    mutant.refresh_structure_hash()
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
    refine_internal: bool = False, # NEW: Allow refining p, omega
    use_amp: bool = True,  # NEW: Use automatic mixed precision (FP16)
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
        refine_internal: If True, ALSO tune p, omega, phi, etc. (use for final polish)
        use_amp: If True, use FP16 mixed precision for ~2x speedup (CUDA only)
    """
    model.train()
    
    # Check if AMP is available (CUDA only)
    amp_enabled = use_amp and x.is_cuda
    scaler = torch.amp.GradScaler('cuda') if amp_enabled else None
    
    # Identify constant parameters (not selection logits)
    constant_params = []
    for name, param in model.named_parameters():
        # Skip selection logits
        if 'logit' in name or 'selector' in name:
            continue
        
        # If scales_only, only include output_scale, edge_weights, output_proj
        if scales_only:
            if 'output_scale' in name or 'edge_weights' in name or 'output_proj' in name:
                if param.requires_grad:
                    constant_params.append(param)
        else:
            # Check for meta-op core parameters
            is_internal = any(x in name for x in ['.p', '.omega', '.phi', '.beta', 'amplitude'])
            
            if is_internal and not refine_internal:
                continue # Skip internal if not requested
                
            if param.requires_grad:
                constant_params.append(param)
    
    if not constant_params:
        return float('inf')
    
    best_loss = float('inf')
    
    try:
        if use_lbfgs:
            # L-BFGS: Better for finding exact constants (recommended by research)
            # Note: L-BFGS doesn't support AMP well, so we skip it here
            y_squeezed = y.squeeze()
            # Create optimizer first, then define closure
            optimizer = LBFGS(constant_params, lr=1.0, max_iter=steps, line_search_fn='strong_wolfe')
            
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
            
            try:
                optimizer.step(closure)
            except Exception as e:
                logger.debug(f"L-BFGS step failed: {e}")
        else:
            # Adam with optional AMP
            optimizer = Adam(constant_params, lr=lr)
            y_squeezed = y.squeeze()
            
            for step in range(steps):
                optimizer.zero_grad()
                
                # Forward pass with autocast for FP16
                if amp_enabled:
                    with torch.amp.autocast('cuda'):
                        pred, _ = model(x, hard=hard)
                        pred = pred.squeeze()
                        loss = F.mse_loss(pred, y_squeezed)
                else:
                    pred, _ = model(x, hard=hard)
                    pred = pred.squeeze()
                    loss = F.mse_loss(pred, y_squeezed)
                
                if torch.isnan(loss):
                    return float('inf')
                
                # Backward pass with scaler for FP16
                if amp_enabled and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(constant_params, max_norm=GRADIENT_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(constant_params, max_norm=GRADIENT_CLIP_NORM)
                    optimizer.step()
                
                best_loss = min(best_loss, loss.item())
    except (MemoryError, RuntimeError) as e:
        logger.debug(f"refine_constants memory/runtime error: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return float('inf')
    except Exception as e:
        logger.warning(f"refine_constants unexpected error: {e}")
        return float('inf')
    
    return best_loss


def quick_refine_internal(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int = 5,
) -> float:
    """
    Quick L-BFGS refinement on INTERNAL constants only (omega, phi, p).
    
    This is a lightweight version of refine_constants() designed for
    per-candidate optimization during evolutionary fitness evaluation.
    
    Key differences from refine_constants():
    - Only optimizes internal parameters (omega, phi, p, amplitude)
    - Does NOT optimize output_proj, edge_weights, or output_scale
    - Uses fewer steps (default 5) for speed
    - Always uses L-BFGS
    
    Args:
        model: The model to refine
        x: Input data
        y: Target data (should match x shape expectations)
        steps: Number of L-BFGS iterations (keep low for speed)
        
    Returns:
        Best MSE achieved (float('inf') on failure)
    """
    model.train()
    
    # Collect ONLY internal constant parameters
    internal_params = []
    for name, param in model.named_parameters():
        # Skip selection logits
        if 'logit' in name or 'selector' in name:
            continue
        
        # Only include internal meta-op parameters
        is_internal = any(x in name for x in ['.p', '.omega', '.phi', 'amplitude'])
        
        if is_internal and param.requires_grad:
            internal_params.append(param)
    
    if not internal_params:
        # No internal params to optimize
        return float('inf')
    
    best_loss = float('inf')
    y_squeezed = y.squeeze()
    
    try:
        optimizer = LBFGS(internal_params, lr=1.0, max_iter=steps, line_search_fn='strong_wolfe')
        
        def closure():
            nonlocal best_loss
            optimizer.zero_grad()
            pred, _ = model(x, hard=True)
            pred = pred.squeeze()
            loss = F.mse_loss(pred, y_squeezed)
            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(float('inf'), requires_grad=True, device=x.device)
            loss.backward()
            best_loss = min(best_loss, loss.item())
            return loss
        
        optimizer.step(closure)
        
    except Exception as e:
        logger.debug(f"quick_refine_internal failed: {e}")
        return float('inf')
    
    model.eval()
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
            # NaN correlation usually means constant predictions (zero variance).
            # Flag it so degenerate models don't silently pass structure checks.
            import warnings
            warnings.warn(
                "check_structure_locked: NaN correlation detected (likely constant "
                f"predictions, pred std={pred.std().item():.2e}). Treating as corr=0.0.",
                RuntimeWarning,
                stacklevel=2,
            )
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
    use_amp: bool = True,  # NEW: Use FP16 mixed precision
) -> Tuple[float, int]:
    """
    Two-phase coefficient refinement when structure is locked.
    
    Phase 1: Tune output_proj only (safe, won't break structure)
    Phase 2: Very gently tune scale parameters (risky, may break structure)
    
    Returns:
        (final_mse, steps_taken)
    """
    y_sq = y.squeeze()
    
    # Setup AMP if available (CUDA only)
    amp_enabled = use_amp and x.is_cuda
    scaler = torch.amp.GradScaler('cuda') if amp_enabled else None
    
    # Get initial MSE
    model.eval()
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
        optimizer = Adam(output_proj_params, lr=OUTPUT_PROJ_LR)
        
        phase1_best_mse = initial_mse
        phase1_best_state = {n: p.detach().clone() for n, p in model.named_parameters() if 'output_proj' in n}
        no_improve = 0
        
        for step in range(max_steps // 2):
            total_steps += 1
            optimizer.zero_grad()
            
            # AMP forward pass
            if amp_enabled:
                with torch.amp.autocast('cuda'):
                    pred, _ = model(x, hard=True)
                    loss = F.mse_loss(pred.squeeze(), y_sq)
            else:
                pred, _ = model(x, hard=True)
                loss = F.mse_loss(pred.squeeze(), y_sq)
            
            if torch.isnan(loss):
                break
            
            # AMP backward pass
            if amp_enabled and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # THEN check improvement and break conditions
            mse = loss.item()
            if mse < phase1_best_mse - MSE_IMPROVEMENT_THRESHOLD:
                phase1_best_mse = mse
                phase1_best_state = {n: p.detach().clone() for n, p in model.named_parameters() if 'output_proj' in n}
                no_improve = 0
            else:
                no_improve += 1
            
            if mse < target_mse or no_improve > patience // 2:
                break
        
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
            optimizer = Adam(scale_params, lr=SCALE_PARAMS_LR)
            
            phase2_best_mse = current_mse
            phase2_best_state = {n: p.detach().clone() for n, p in model.named_parameters() if 'scale' in n}
            no_improve = 0
            
            for step in range(max_steps // 2):
                total_steps += 1
                optimizer.zero_grad()
                
                # AMP forward pass
                if amp_enabled:
                    with torch.amp.autocast('cuda'):
                        pred, _ = model(x, hard=True)
                        loss = F.mse_loss(pred.squeeze(), y_sq)
                else:
                    pred, _ = model(x, hard=True)
                    loss = F.mse_loss(pred.squeeze(), y_sq)
                
                if torch.isnan(loss):
                    break
                
                # AMP backward pass
                if amp_enabled and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(scale_params, max_norm=SCALE_GRADIENT_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(scale_params, max_norm=SCALE_GRADIENT_CLIP_NORM)
                    optimizer.step()
                
                # THEN check improvement and break conditions
                mse = loss.item()
                if mse < phase2_best_mse - MSE_IMPROVEMENT_THRESHOLD:
                    phase2_best_mse = mse
                    phase2_best_state = {n: p.detach().clone() for n, p in model.named_parameters() if 'scale' in n}
                    no_improve = 0
                else:
                    no_improve += 1
                
                if mse < target_mse or no_improve > patience // 2:
                    break
            
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
    
    # Return model in eval mode
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
    refine_internal_constants: bool = True,  # NEW: Enable internal constant refinement
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
        refine_internal_constants: If True, run L-BFGS on ALL parameters (p, omega) first
        
    Returns:
        (final_mse, formula_string)
    """
    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device)
    y_sq = y.squeeze()
    
    model.eval()
    with torch.no_grad():
        pred, _ = model(x, hard=True)
        initial_mse = F.mse_loss(pred.squeeze(), y_sq).item()
    
    # 0. Refine INTERNAL constants first (p, omega, etc.)
    # This fixes problems like sin(3.5x) where 3.5 was missed
    if refine_internal_constants:
        print("Refining internal constants (frequencies, powers)...")
        refine_constants(
            model, x, y,
            steps=50,  # Short burst of L-BFGS
            use_lbfgs=True,
            scales_only=False,
            hard=True,  # Optimize the HARD discrete structure
            refine_internal=True # Enable tuning of p, omega, etc.
        )
        
        # Re-evaluate MSE
        with torch.no_grad():
            pred, _ = model(x, hard=True)
            initial_mse = F.mse_loss(pred.squeeze(), y_sq).item()
        print(f"MSE after internal refinement: {initial_mse:.6f}")
    
    # Just do L-BFGS refinement on output_proj WITH L1 SPARSITY
    output_params = [p for n, p in model.named_parameters() if 'output_proj' in n and p.requires_grad]
    
    if output_params and initial_mse > target_mse:
        try:
            # Create optimizer first, then define closure
            optimizer = LBFGS(output_params, lr=1.0, max_iter=max_steps, line_search_fn='strong_wolfe')
            
            def closure():
                optimizer.zero_grad()
                pred, _ = model(x, hard=True)
                mse_loss = F.mse_loss(pred.squeeze(), y_sq)
                # Add L1 sparsity penalty
                l1_loss = sum(p.abs().sum() for p in output_params)
                loss = mse_loss + l1_weight * l1_loss
                if not torch.isnan(loss):
                    loss.backward()
                else:
                    return torch.tensor(float('inf'), requires_grad=True)
                return loss
            
            optimizer.step(closure)
        except (RuntimeError, ValueError) as e:
            logger.debug(f"finalize_model_coefficients L-BFGS pass 1 failed (optimization error): {e}")
        except Exception as e:
            logger.warning(f"finalize_model_coefficients L-BFGS pass 1 failed (unexpected): {e}")
    
    # Prune small coefficients
    prune_small_coefficients(model, threshold_ratio=sparsity_threshold, absolute_threshold=0.1)
    
    # Second L-BFGS pass (pure fitting, no L1)
    model.eval()
    with torch.no_grad():
        pred, _ = model(x, hard=True)
        post_prune_mse = F.mse_loss(pred.squeeze(), y_sq).item()
    
    if output_params and post_prune_mse > target_mse:
        try:
            # Create optimizer first, then define closure
            optimizer2 = LBFGS(output_params, lr=1.0, max_iter=max_steps // 2, line_search_fn='strong_wolfe')
            
            def closure2():
                optimizer2.zero_grad()
                pred, _ = model(x, hard=True)
                loss = F.mse_loss(pred.squeeze(), y_sq)
                if not torch.isnan(loss):
                    loss.backward()
                return loss
            
            optimizer2.step(closure2)
        except (RuntimeError, ValueError) as e:
             logger.debug(f"finalize_model_coefficients L-BFGS pass 2 failed (optimization error): {e}")
        except Exception as e:
            logger.warning(f"finalize_model_coefficients L-BFGS pass 2 failed (unexpected): {e}")
    
    # Final pruning
    prune_small_coefficients(model, threshold_ratio=FINAL_PRUNE_THRESHOLD_RATIO, absolute_threshold=FINAL_PRUNE_ABSOLUTE)
    
    model.eval()
    with torch.no_grad():
        pred, _ = model(x, hard=True)
        final_mse = F.mse_loss(pred.squeeze(), y_sq).item()
    
    # Safeguard: never return NaN - it breaks comparisons
    if math.isnan(final_mse) or math.isinf(final_mse):
        final_mse = float('inf')
    
    formula = model.get_formula() if hasattr(model, 'get_formula') else "?"
    
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
    # Note: itertools.combinations and copy are imported at module level
    
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
            
            # Reset to original state (no need for deepcopy - load_state_dict handles copying)
            model.load_state_dict(original_state)
            
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
                    optimizer.zero_grad()
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
    device = weights.device
    
    if n <= 1:
        return torch.tensor(0.0, device=device)
    
    l1 = weights.abs().sum()
    l2 = weights.norm(2)
    
    if l2 < 1e-8:
        return torch.tensor(0.0, device=device)
    
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
        device = next((p.device for p in model.parameters()), torch.device('cpu'))
        return torch.tensor(0.0, device=device)
    
    weights = model.output_proj.weight
    
    # L1 penalty on small weights (encourage exact zeros)
    l1_loss = weights.abs().mean()
    
    # Hoyer penalty (encourage sparsity pattern)
    hoyer = hoyer_sparsity(weights.flatten())
    hoyer_loss = 1.0 - hoyer  # Minimize this to maximize sparsity
    
    return 0.5 * l1_loss + 0.5 * hoyer_loss


class EvolutionaryONNTrainer(RiskSeekingEvolutionMixin):
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
        # NEW: Nested BFGS for internal constants (omega, phi, p)
        nested_bfgs: bool = True,           # Enable per-candidate internal refinement
        nested_bfgs_steps: int = 5,         # L-BFGS iterations per candidate
        nested_bfgs_every: int = 1,         # Refine every N generations
        # NEW: Curve classifier warm-start
        use_curve_classifier: bool = False,  # Use curve shape to predict operators
        curve_classifier_path: str = DEFAULT_CURVE_CLASSIFIER_PATH,
        # NEW: Early stopping on exact match
        early_stop_mse: float = 1e-6,  # Stop if MSE below this (exact match)
        early_stop_corr: float = 0.9999,  # Stop if correlation above this
        # C++ backend controls (parity with sklearn_wrapper)
        p_min: float = -2.0,  # Minimum power exponent bound for C++ evolution
        p_max: float = 3.0,   # Maximum power exponent bound for C++ evolution
        timeout_seconds: int = 120,  # Timeout for C++ evolution in seconds
        random_seed: int = -1,  # Random seed for C++ evolution (-1 = nondeterministic)
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
        
        # Initialize RSPG if available and enabled
        if RSPG_AVAILABLE and risk_seeking:
            self.init_risk_seeking(
                enable_rspg=True,
                rspg_percentile=risk_seeking_percentile * 100,  # Convert to percentage
                rspg_temperature=0.5,
                monitor_window_size=10,
            )
        else:
            self.gradient_monitor = None
        
        # NEW: Visualization
        self.visualizer = visualizer
        
        # NEW: Nested BFGS for internal constants
        self.nested_bfgs = nested_bfgs
        self.nested_bfgs_steps = nested_bfgs_steps
        self.nested_bfgs_every = nested_bfgs_every
        
        # NEW: Curve classifier warm-start
        self.use_curve_classifier = use_curve_classifier
        self.curve_classifier_path = curve_classifier_path
        
        # NEW: Early stopping thresholds
        self.early_stop_mse = early_stop_mse
        self.early_stop_corr = early_stop_corr
        
        # C++ backend controls
        self.p_min = p_min
        self.p_max = p_max
        self.timeout_seconds = timeout_seconds
        self.random_seed = random_seed
        
        # NEW: Graduated structure confidence tracker (replaces binary lock)
        self.confidence_tracker = StructureConfidenceTracker(
            mse_threshold=0.01,
            stability_generations=3,
            min_mutation_fraction=0.1,
        )
        
        self.population: List[Individual] = []
        self.explorers: List[Individual] = []  # Separate explorer population
        self.generation = 0
        self.best_ever: Optional[Individual] = None
        self.best_explorer: Optional[Individual] = None  # Best found by explorers
        self.stuck_generations = 0  # Track consecutive stuck generations
        self.last_best_ever_fitness = float('inf')  # Track if best_ever improved
        self.history = []
    
    def initialize_population(self, x: torch.Tensor = None, y: torch.Tensor = None):
        """
        Create diverse initial population + explorer subpopulation.
        
        If use_curve_classifier is True and x, y data provided, uses CNN predictions
        to probabilistically seed the population toward high-confidence operators.
        
        Args:
            x: Optional input data for classifier prediction
            y: Optional target data for classifier prediction
        """
        self.population = []
        self.explorers = []
        
        # Get classifier predictions if enabled and data provided
        classifier_predictions = None
        detected_omegas = None
        
        if self.use_curve_classifier and x is not None and y is not None:
            try:
                # Import here to avoid circular dependency
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))
                from curve_classifier_integration import predict_operators
                
                # Get classifier predictions
                x_np = x.cpu().numpy().flatten() if hasattr(x, 'cpu') else np.asarray(x).flatten()
                y_np = y.cpu().numpy().flatten() if hasattr(y, 'cpu') else y.flatten()
                
                classifier_predictions = predict_operators(
                    x_np, y_np, 
                    model_path=self.curve_classifier_path,
                    threshold=0.3,
                )
                
                if classifier_predictions:
                    logger.info(f"CNN seeding with predictions: {classifier_predictions}")
                    print(f"🧠 CNN-seeded initialization: {list(classifier_predictions.keys())}")
                
                # Also get FFT frequencies for omega seeding
                detected_omegas = detect_dominant_frequency(x, y, n_frequencies=3)
                
            except Exception as e:
                logger.warning(f"Classifier prediction failed, using random init: {e}")
                classifier_predictions = None
        
        # Main population
        for i in range(self.population_size):
            model = self.model_factory().to(self.device)
            
            if classifier_predictions:
                # CNN-seeded initialization: probabilistically bias toward predictions
                random_operation_init(model, bias_strength=DEFAULT_BIAS_STRENGTH + i * BIAS_INCREMENT)
                seed_population_from_classifier(
                    model, 
                    classifier_predictions, 
                    detected_omegas=detected_omegas,
                    individual_idx=i,
                    seed_fraction=0.9,  # 90% of population gets seeded
                )
            else:
                # Standard random initialization
                random_operation_init(model, bias_strength=DEFAULT_BIAS_STRENGTH + i * BIAS_INCREMENT)
            
            individual = Individual(model, generation=0)
            self.population.append(individual)
        
        # Explorer subpopulation (high mutation scouts)
        # Note: Explorers are NOT CNN-seeded - they explore randomly
        if self.use_explorers:
            for i in range(self.n_explorers):
                model = self.model_factory().to(self.device)
                # Initialize explorers with MORE random bias (broader search)
                random_operation_init(model, bias_strength=EXPLORER_BIAS_STRENGTH + i * EXPLORER_BIAS_INCREMENT)
                explorer = Individual(model, generation=0)
                explorer.is_explorer = True  # Tag as explorer
                self.explorers.append(explorer)
            
            if classifier_predictions:
                print(f"Initialized CNN-seeded population of {self.population_size} individuals + {self.n_explorers} explorers (random)")
            else:
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

        def _add_if_finite(base: float, value: float, weight: float = 1.0) -> float:
            """Safely add a weighted scalar term only when it is finite."""
            term = weight * value
            if math.isfinite(term):
                return base + term
            return base
        
        for ind in all_individuals:
            # TIER 3: Skip re-evaluation of elite individuals that weren't mutated
            # Elites keep their fitness from previous generation (small speedup)
            if hasattr(ind, '_is_elite') and ind._is_elite and ind.fitness < float('inf'):
                continue
            # NESTED BFGS: Refine internal constants (omega, phi, p) before evaluation
            # This enables discovering formulas like sin(3.2*x) or x^2.5
            if self.nested_bfgs and generation % self.nested_bfgs_every == 0:
                # Only refine if not already refined this generation
                if not getattr(ind, '_refined_this_gen', False):
                    # Need y with proper shape for quick_refine_internal
                    y_for_refine = y.unsqueeze(-1) if y.dim() == 1 else y
                    refine_loss = quick_refine_internal(
                        ind.model,
                        x,
                        y_for_refine,
                        steps=self.nested_bfgs_steps,
                    )
                    # Deterministic fallback path when internal L-BFGS fails.
                    if not math.isfinite(refine_loss):
                        refine_constants(
                            ind.model,
                            x,
                            y_for_refine,
                            steps=max(5, self.nested_bfgs_steps),
                            lr=0.01,
                            use_lbfgs=False,
                            hard=True,
                        )
                    ind._refined_this_gen = True
            
            ind.model.eval()
            try:
                with torch.no_grad():
                    pred, _ = ind.model(x, hard=True)
                    pred = pred.squeeze()  # Ensure pred is 1D
                    mse = F.mse_loss(pred, y).item()
                    ind.raw_mse = mse
                    
                    if math.isnan(mse) or math.isinf(mse):
                        ind.fitness = float('inf')
                        ind.complexity = float('inf')
                    else:
                        # Add complexity penalty (BIC-inspired parsimony)
                        complexity = calculate_complexity(ind.model)
                        ind.complexity = complexity
                        if not math.isfinite(complexity):
                            ind.fitness = float('inf')
                            continue

                        fitness = _add_if_finite(mse, complexity, self.complexity_penalty)
                        
                        # Coefficient sparsity penalty (Hoyer-inspired)
                        # INCREASED: Stronger sparsity encourages fewer output terms
                        sparsity = coefficient_sparsity_loss(ind.model).item()
                        fitness = _add_if_finite(fitness, sparsity, 0.05)
                        
                        # Entropy regularization with ANNEALED weight (Tier 1)
                        # Mild early (allow exploration), strong late (force discrete)
                        if current_entropy_weight > 0 and hasattr(ind.model, 'entropy_regularization'):
                            try:
                                entropy = ind.model.entropy_regularization().item()
                                fitness = _add_if_finite(fitness, entropy, current_entropy_weight)
                            except Exception:
                                pass
                        
                        # Progressive rounding penalty (Tier 1)
                        # Soft push toward integers for p, omega, etc.
                        if self.progressive_round_weight > 0:
                            try:
                                round_loss = progressive_round_loss(ind.model).item()
                                fitness = _add_if_finite(fitness, round_loss, self.progressive_round_weight)
                            except Exception:
                                pass

                        ind.fitness = fitness if math.isfinite(fitness) else float('inf')
            except Exception:
                ind.fitness = float('inf')
                ind.raw_mse = float('inf')
                ind.complexity = float('inf')
        
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
    
    def select_and_reproduce(self, x: torch.Tensor, y: torch.Tensor, diversity: int = 10, mutation_rate: float = None):
        """
        Selection and reproduction with Lamarckian inheritance.
        
        Tier 2 features:
        - Lamarckian inheritance: passes optimized weights from parent to child
        - Risk-seeking selection: bias toward top performers (for SR we want 
          the BEST formula, not average performance)
        
        Args:
            x, y: Training data
            diversity: Current population diversity (used for mass mutation)
            mutation_rate: Override mutation rate (None = use self.mutation_rate)
        """
        # Use provided mutation rate or default to self.mutation_rate
        effective_rate = mutation_rate if mutation_rate is not None else self.mutation_rate
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
            
            # Keep elite unchanged and mark them to skip re-evaluation
            for elite in sorted_pop[:self.elite_size]:
                elite._is_elite = True  # TIER 3: Mark elite for evaluation skip
                new_population.append(elite)
            
            # Fill rest with mutations of top performers
            while len(new_population) < self.population_size:
                # Check if RSPG should be used
                use_rspg = self.gradient_monitor is not None and hasattr(self, 'should_use_rspg') and self.should_use_rspg()
                
                if use_rspg:
                    # Risk-seeking selection: use probabilities that favor top performers
                    fitnesses = [ind.fitness for ind in selection_pool]
                    parents = self.select_parents_rspg(selection_pool, fitnesses, n_parents=1)
                    parent = parents[0]
                else:
                    # Normal tournament selection
                    candidates = random.sample(selection_pool, min(3, len(selection_pool)))
                    parent = min(candidates, key=lambda ind: ind.fitness)
                
                # Lamarckian mutation: clone preserves all parent weights
                # Mutation only changes discrete structure (operation selection)
                # Continuous weights (p, omega, etc.) are inherited!
                if self.lamarckian:
                    child = mutate_operations_lamarckian(parent, effective_rate)
                else:
                    child = mutate_operations(parent, effective_rate)
                
                # TIER 3: Progressive refinement reduction
                # Early: full refinement (finding structure)
                # Late: minimal refinement (structure stable, just tune coefficients)
                base_steps = self.constant_refine_steps // 2 if self.lamarckian else self.constant_refine_steps
                
                # Reduce steps based on confidence (0.0->1.0 maps to 1.0->0.3)
                confidence_factor = 1.0 - 0.7 * self.confidence_tracker.confidence
                adaptive_steps = max(15, int(base_steps * confidence_factor))
                
                # TIER 3: Use L-BFGS when confidence is high (faster convergence)
                use_lbfgs = self.confidence_tracker.confidence > 0.4
                
                refine_constants(
                    child.model, x, y,
                    steps=adaptive_steps,
                    lr=0.02,
                    hard=self.constant_refine_hard,
                    use_lbfgs=use_lbfgs,
                )
                
                child._is_elite = False  # Mark as new individual, needs evaluation
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
        trace_path: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        random_seed: Optional[int] = None,
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
        
        # Store original data for curve classifier (before normalization)
        x_original = x.clone()
        y_original = y.clone()

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
            fit_y = (fit_y - y_mean) / y_std
            
        # ---------------------------------------------------------------------
        # NEW C++ BACKEND INTEGRATION
        # If the C++ core is available, we run the evolution natively in C++ 
        # for a massive speedup (100x+), skipping the PyTorch loop.
        # ---------------------------------------------------------------------
        try:
            import sys
            import os
            from pathlib import Path
            cpp_dir = Path(__file__).parent / 'cpp'
            if str(cpp_dir) not in sys.path:
                sys.path.insert(0, str(cpp_dir))
            import _core
            
            print("\n" + "*"*60)
            print("🚀 USING BLAZING FAST C++ EVOLUTION BACKEND 🚀")
            print("*"*60 + "\n")
            
            # FFT warm-start is only well-defined for univariate inputs.
            detected_omegas = []
            if x.dim() == 1 or (x.dim() > 1 and x.shape[1] == 1):
                detected_omegas = detect_dominant_frequency(x, y, n_frequencies=3)
                if detected_omegas and detected_omegas[0] != 1.0:
                    print(f"FFT detected frequencies (omega) for C++ seeding: {[f'{o:.2f}' for o in detected_omegas]}")
                else:
                    detected_omegas = []

            # Prepare inputs for C++
            X_list = [x[:, i].cpu().numpy() for i in range(x.shape[1])] if x.dim() > 1 else [x.cpu().numpy()]
            y_arr = y.reshape(-1).cpu().numpy()
            
            # Resolve C++ parameters: train() overrides take priority, then __init__ defaults
            effective_timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
            effective_seed = random_seed if random_seed is not None else self.random_seed
            
            start_time = time.time()
            
            # Run C++ Loop (matching sklearn_wrapper parameter parity)
            result = _core.run_evolution(
                X_list=X_list,
                y=y_arr,
                pop_size=self.population_size,
                generations=generations,
                early_stop_mse=self.early_stop_mse,
                seed_omegas=detected_omegas,
                trace_path=trace_path if trace_path else "",
                timeout_seconds=effective_timeout,
                random_seed=effective_seed,
                p_min=self.p_min,
                p_max=self.p_max,
            )
            
            end_time = time.time()
            print(f"✅ C++ Evolution completed in {end_time - start_time:.2f} seconds!")
            print(f"C++ Best Target MSE: {result['best_mse']:.6f}")
            
            # Parse the C++ AST back into a PyTorch model
            model = self.model_factory().to(self.device)
            
            # Helper to map C++ ops to PyTorch meta_ops indices
            def set_cpp_weights_to_model(cpp_nodes, out_weights, out_bias, dummy_model):
                # NOTE: For now, we just construct a mock-up of the best model 
                # so the visualizer and tester don't crash. 
                # Converting a jagged DAG strictly back to the rigid PyTorch 
                # block-matrix format requires structural projection.
                # Since the C++ native run is our final output, we return 
                # the string formula of the best graph directly.
                pass
                
            print("\nC++ BEST FORMULA (AST NODES):")
            unary_ops = ["Periodic", "Power", "IntPow", "Exp", "Log"]
            binary_ops = ["Arithmetic", "Division", "Aggregation"]
            
            for idx, node in enumerate(result['nodes']):
                op_type = node.get('type', 0)
                op = ["Input", "Constant", "Unary", "Binary"][op_type] if op_type < 4 else "Unknown"
                
                if op_type == 2:  # Unary
                    op += f"[{unary_ops[node.get('unary_op', 0)]}]"
                elif op_type == 3:  # Binary
                    op += f"[{binary_ops[node.get('binary_op', 0)]}]"
                    
                val = node.get('value', 0)
                feat = node.get('feature_idx', 0)
                print(f"  Node {idx}: {op} (val={val:.2f}, feat={feat}) left={node.get('left_child')}, right={node.get('right_child')}")
            
            print(f"Output Weights: {result['output_weights']}")
            print(f"Output Bias: {result['output_bias']:.4f}")
            
            # P8: Create a real runnable PyTorch module from the C++ AST
            try:
                from glassbox.sr.cpp.export_pytorch import CppGraphModule
                cpp_model = CppGraphModule(result).to(self.device)
                self.best_ever = Individual(cpp_model, fitness=result['best_mse'])
                self.best_ever.raw_mse = result['best_mse']
                print("✅ Created CppGraphModule (runnable nn.Module)")
            except Exception as e:
                print(f"⚠️ CppGraphModule failed ({e}), using dummy model")
                dummy_model = self.model_factory().to(self.device)
                self.best_ever = Individual(dummy_model, fitness=result['best_mse'])
                self.best_ever.raw_mse = result['best_mse']
            
            # We return early. The PyTorch loop is deprecated by the C++ core.
            
            formula = result.get('formula', '0')
            display_mse = None
            try:
                from sympy import Symbol, sympify
                from sympy.utilities.lambdify import lambdify
                from glassbox.sr.meta_ops import safe_numpy_power

                x_np = x.detach().cpu().numpy()
                y_np = y.detach().cpu().numpy().reshape(-1)
                if x_np.ndim == 1:
                    x_np = x_np.reshape(-1, 1)

                n_features = x_np.shape[1]
                local_symbols = {"x": Symbol("x")}
                for i in range(n_features):
                    local_symbols[f"x{i}"] = Symbol(f"x{i}")

                expr = sympify(formula.replace("^", "**"), locals=local_symbols)
                
                # Inject safe power into lambdify context
                modules = [{"pow": safe_numpy_power, "Pow": safe_numpy_power}, "numpy"]
                
                if n_features == 1:
                    fn = lambdify(local_symbols["x"], expr, modules=modules)
                    y_pred = fn(x_np[:, 0])
                else:
                    args = [local_symbols[f"x{i}"] for i in range(n_features)]
                    fn = lambdify(args, expr, modules=modules)
                    y_pred = fn(*[x_np[:, i] for i in range(n_features)])

                y_pred = np.asarray(y_pred, dtype=np.float64).reshape(y_np.shape)
                mask = np.isfinite(y_pred) & np.isfinite(y_np)
                if mask.sum() >= 10:
                    candidate = float(np.mean((y_pred[mask] - y_np[mask]) ** 2))
                    if math.isfinite(candidate):
                        display_mse = candidate
            except Exception:
                display_mse = None

            score_mse = display_mse if display_mse is not None else float(result['best_mse'])
            self.best_ever.fitness = score_mse
            elapsed = end_time - start_time
            
            return {
                'model': self.best_ever.model,
                'history': [{'generation': i, 'best_fitness': 0} for i in range(generations)],
                'final_mse': score_mse,
                'final_mse_display': display_mse,
                'final_mse_raw': float(result['best_mse']),
                'correlation': 0.0,
                'formula': formula,
                'training_time': elapsed,
                'cpp_ast': result,
            }
            
        except ImportError as e:
            print(f"⚠️ C++ backend not available ({e}), falling back to PyTorch...")
        except Exception as e:
            print(f"⚠️ C++ backend failed at runtime ({e}), falling back to PyTorch...")
        # ---------------------------------------------------------------------
        
        # Initialize population (with CNN seeding if curve classifier enabled)
        self.initialize_population(x_original, y_original)
        
        # FFT-based frequency detection only applies to univariate inputs.
        detected_omegas = []
        if x.dim() == 1 or (x.dim() > 1 and x.shape[1] == 1):
            detected_omegas = detect_dominant_frequency(x, y, n_frequencies=3)
            if detected_omegas and detected_omegas[0] != 1.0:
                print(f"FFT detected frequencies (omega): {[f'{o:.2f}' for o in detected_omegas]}")
                # Seed some individuals with detected frequencies
                for i, ind in enumerate(self.population):
                    seed_omega_from_fft(ind.model, detected_omegas, individual_idx=i)
                if self.use_explorers:
                    for i, explorer in enumerate(self.explorers):
                        seed_omega_from_fft(explorer.model, detected_omegas, individual_idx=i)
        
        # Curve-classifier warm-start is applied in initialize_population()
        # to avoid duplicate prediction and bias passes.
        
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
        # Note: structure_locked is now handled by self.confidence_tracker
        
        for gen in range(generations):
            # Anneal tau for all individuals (early soft, late hard)
            current_tau = anneal_tau(gen, generations, self.tau_start, self.tau_end)
            for ind in self.population:
                set_model_tau(ind.model, current_tau)
            
            # Explorers keep HIGHER tau (stay soft for more exploration)
            # Only apply when not in refinement-only mode
            if self.use_explorers and not self.confidence_tracker.should_refine_only():
                explorer_tau = max(current_tau, MIN_EXPLORER_TAU)  # Explorers never go below min
                for explorer in self.explorers:
                    set_model_tau(explorer.model, explorer_tau)
            
            # Reset nested BFGS refinement flags for all individuals
            # This ensures each individual is refined once per generation
            if self.nested_bfgs:
                for ind in self.population:
                    ind._refined_this_gen = False
                if self.use_explorers:
                    for explorer in self.explorers:
                        explorer._refined_this_gen = False
            
            # Evaluate with annealed entropy weight (pass generation info)
            self.evaluate_fitness(fit_x, fit_y, generation=gen, total_generations=generations)
            
            # Update gradient monitor if RSPG is enabled
            if self.gradient_monitor is not None and self.best_ever is not None:
                self.update_gradient_monitor(
                    loss=self.best_ever.fitness,
                    model=self.best_ever.model,
                )
            
            # Update structure confidence tracker (replaces binary lock)
            if self.best_ever and gen >= 3:
                # Compute correlation for confidence tracking
                self.best_ever.model.eval()
                with torch.no_grad():
                    pred, _ = self.best_ever.model(x, hard=True)
                    pred_sq = pred.squeeze()
                    y_sq = y.squeeze()
                    corr = torch.corrcoef(torch.stack([pred_sq.cpu(), y_sq.cpu()]))[0, 1].item()
                    mse = F.mse_loss(pred_sq, y_sq).item()
                    if math.isnan(corr):
                        import warnings
                        warnings.warn(
                            f"Gen {gen}: NaN correlation on best_ever (likely constant "
                            f"predictions, pred std={pred_sq.std().item():.2e}). "
                            "Treating as corr=0.0.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        corr = 0.0
                
                # Update confidence tracker
                self.confidence_tracker.update(gen, corr, mse)
                
                # Log status changes
                if self.confidence_tracker.should_refine_only() and gen % 5 == 0:
                    print(f"Gen {gen:3d} | {self.confidence_tracker.get_status()}")
            
            # Stats
            fitnesses = [ind.fitness for ind in self.population if ind.fitness < float('inf')]
            if fitnesses:
                best_fit = min(fitnesses)
                mean_fit = sum(fitnesses) / len(fitnesses)
                diversity = len(set(ind.structure_hash for ind in self.population))
            else:
                best_fit = mean_fit = float('inf')
                diversity = 0
            
            # Track stuck generations and trigger population restart
            if self.best_ever is not None:
                # Check if best_ever improved this generation
                if self.best_ever.fitness < self.last_best_ever_fitness - 0.001:
                    # Improved! Reset stuck counter
                    self.stuck_generations = 0
                    self.last_best_ever_fitness = self.best_ever.fitness
                else:
                    # No improvement
                    self.stuck_generations += 1
                
                # POPULATION RESTART: If stuck for too long, reinject fresh individuals
                if self.stuck_generations >= 5 and diversity <= 3:
                    print(f"POPULATION RESTART: Stuck for {self.stuck_generations} gens, injecting fresh individuals")
                    
                    # Keep only top 2 elites
                    sorted_pop = sorted(self.population, key=lambda ind: ind.fitness)
                    new_population = sorted_pop[:2]
                    
                    # Fill rest with fresh random individuals
                    while len(new_population) < self.population_size:
                        model = self.model_factory().to(self.device)
                        random_operation_init(model, bias_strength=random.uniform(1.0, 5.0))
                        child = Individual(model, generation=gen)
                        # Quick refine
                        refine_constants(child.model, x, y, steps=20, lr=0.02, hard=self.constant_refine_hard)
                        new_population.append(child)
                    
                    self.population = new_population
                    self.stuck_generations = 0
                    # Re-evaluate fitness
                    self.evaluate_fitness(fit_x, fit_y, generation=gen, total_generations=generations)
            
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
                    valid_explorer_fits = [e.fitness for e in self.explorers if e.fitness < float('inf')]
                    if valid_explorer_fits:
                        best_explorer_fit = min(valid_explorer_fits)
                        explorer_info = f" | Explorer: {best_explorer_fit:.4f}"
                    else:
                        explorer_info = " | Explorer: inf"
                
                # Add RSPG status
                rspg_info = ""
                if self.gradient_monitor is not None:
                    rspg_stats = self.get_rspg_stats()
                    if rspg_stats.get('is_rspg_active', 0.0) > 0.5:
                        stuck = "STUCK" if rspg_stats.get('is_stuck', 0.0) > 0.5 else ""
                        explode = "EXPLODE" if rspg_stats.get('is_exploding', 0.0) > 0.5 else ""
                        status = f"{stuck}{explode}".strip() or "ACTIVE"
                        rspg_info = f" | RSPG:{status}"
                
                # Add confidence tracker info
                conf_info = ""
                if hasattr(self, 'confidence_tracker') and self.confidence_tracker.confidence > 0:
                    eff_rate = self.confidence_tracker.get_effective_mutation_rate(self.mutation_rate)
                    conf_info = f" | Conf:{self.confidence_tracker.confidence:.2f} MutRate:{eff_rate:.2f}"
                
                print(f"Gen {gen:3d} | Best: {best_fit:.4f} | "
                      f"Mean: {mean_fit:.4f} | Diversity: {diversity} | "
                      f"Best Ever: {self.best_ever.fitness:.4f}{explorer_info}{rspg_info}{conf_info}")

            # EARLY STOPPING CHECK
            # Check for exact match (very low MSE)
            best_raw_mse = getattr(self.best_ever, 'raw_mse', float('inf')) if self.best_ever else float('inf')
            if self.best_ever and best_raw_mse < self.early_stop_mse:
                print(f"\n*** EXACT MATCH FOUND! MSE={best_raw_mse:.2e} at Gen {gen} ***")
                break
            
            # Check for near-perfect correlation (structure is correct)
            if hasattr(self, 'confidence_tracker') and self.confidence_tracker.confidence > 0:
                # Get current correlation
                self.best_ever.model.eval()
                with torch.no_grad():
                    pred, _ = self.best_ever.model(x, hard=True)
                    pred_sq = pred.squeeze()
                    y_sq = y.squeeze()
                    current_corr = torch.corrcoef(torch.stack([pred_sq.cpu(), y_sq.cpu()]))[0, 1].item()
                    if not math.isnan(current_corr) and current_corr > self.early_stop_corr:
                        print(f"\n*** NEAR-EXACT MATCH! Corr={current_corr:.6f} at Gen {gen} ***")
                        break
            
            # Phase 1 doesn't need perfect MSE - Phase 2 refinement will fix coefficients
            # Stop if we have a good structure (high correlation) and stable MSE
            if self.best_ever and best_raw_mse < 0.01:
                # MSE below 0.01 is good enough for Phase 1
                if hasattr(self, 'confidence_tracker') and self.confidence_tracker.generations_stable > 2:
                    print(f"\nEARLY STOPPING: Reached target accuracy (MSE < 0.01) at Gen {gen}")
                    break
            
            # Also stop if correlation is very high (structure is correct, coefficients need polish)
            if hasattr(self, 'confidence_tracker') and self.confidence_tracker.confidence > 0.95:
                if self.confidence_tracker.generations_stable > 5:
                    print(f"\nEARLY STOPPING: High confidence structure found at Gen {gen}")
                    break
            
            # Stop if we are "Refining Only" and haven't improved for a long time
            if hasattr(self, 'confidence_tracker') and self.confidence_tracker.should_refine_only() and self.confidence_tracker.generations_stable > 8:
                 print(f"\nEARLY STOPPING: Converged (Structure Locked for 8+ gens) at Gen {gen}")
                 break
            
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
                        # Conservative: only prune if MSE increase < tolerance
                        n_pruned, new_mse = adaptive_coefficient_pruning(
                            ind.model, fit_x, fit_y,
                            prune_ratio=PRUNE_MSE_TOLERANCE
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
                if self.confidence_tracker.should_refine_only():
                    # High confidence: refinement-only mode, skip mutations
                    pass
                else:
                    # Get effective mutation rate based on confidence
                    effective_mutation_rate = self.confidence_tracker.get_effective_mutation_rate(self.mutation_rate)
                    
                    # STUCK BOOST: If RSPG detects stuck, boost mutation significantly!
                    if self.gradient_monitor is not None:
                        rspg_stats = self.get_rspg_stats()
                        if rspg_stats.get('is_stuck', 0.0) > 0.5:
                            # Boost mutation to escape local minima
                            effective_mutation_rate = min(0.9, effective_mutation_rate * 2.5)
                            # Also reset confidence to allow more exploration
                            if self.confidence_tracker.confidence > 0.3:
                                self.confidence_tracker.confidence *= 0.5
                                self.confidence_tracker.generations_stable = 0
                    
                    # Normal evolution with (possibly boosted) mutations
                    self.select_and_reproduce(x, y, diversity=diversity, mutation_rate=effective_mutation_rate)
                    
                    # Evolve explorers in parallel (if enabled and not high confidence)
                    if self.confidence_tracker.confidence < 0.8:
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

            # Check if structure is good (high correlation)
            structure_good, corr, mse_before = check_structure_quality(
                self.best_ever.model, x, y, corr_threshold=0.99
            )
            
            print(f"Structure check: corr={corr:.4f}, MSE={mse_before:.4f}")
            
            if structure_good and mse_before > 0.01:
                # Structure is good but coefficients need work
                # Phase 2: Run intensive coefficient refinement
                print(f"Structure LOCKED (corr={corr:.4f}). MSE={mse_before:.4f} - running intensive coefficient refinement...")
                
                backup_state = copy.deepcopy(self.best_ever.model.state_dict())
                
                # Run intensive coefficient refinement (Phase 2)
                final_mse, steps = intensive_coefficient_refinement(
                    self.best_ever.model, x, y,
                    target_mse=0.001,  # Aim for very low MSE
                    max_steps=500,
                    patience=100,
                )
                
                # Check if refinement helped or hurt
                self.best_ever.model.eval()
                with torch.no_grad():
                    pred_after, _ = self.best_ever.model(x, hard=True)
                    mse_after = F.mse_loss(pred_after.squeeze(), y.squeeze()).item()
                
                if mse_after < mse_before * 0.8:  # Improved by at least 20%
                    print(f"  Phase 2 refinement: MSE {mse_before:.4f} -> {mse_after:.4f} ({steps} steps)")
                else:
                    # Refinement didn't help much, try output_proj only as fallback
                    print(f"  Phase 2 refinement didn't improve enough ({mse_before:.4f} -> {mse_after:.4f})")
                    self.best_ever.model.load_state_dict(backup_state)
                    
                    # Fallback: gentle output_proj-only refinement
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
                            pred_fallback, _ = self.best_ever.model(x, hard=True)
                            mse_fallback = F.mse_loss(pred_fallback.squeeze(), y.squeeze()).item()
                        
                        if mse_fallback < mse_before:
                            print(f"  Fallback output-proj refinement: MSE {mse_before:.4f} -> {mse_fallback:.4f}")
                        else:
                            # Revert entirely
                            self.best_ever.model.load_state_dict(backup_state)
            
            elif mse_before > TARGET_MSE_THRESHOLD:
                # Structure not locked, try gentle refinement
                print("Structure not locked, trying gentle refinement...")
                backup_state = copy.deepcopy(self.best_ever.model.state_dict())
                
                refine_constants(
                    self.best_ever.model, x, y,
                    steps=GENTLE_REFINEMENT_STEPS,
                    lr=GENTLE_REFINEMENT_LR,
                    use_lbfgs=False,
                    scales_only=True,
                    hard=True,
                )
                
                self.best_ever.model.eval()
                with torch.no_grad():
                    pred_after, _ = self.best_ever.model(x, hard=True)
                    mse_after = F.mse_loss(pred_after.squeeze(), y.squeeze()).item()
                
                if mse_after > mse_before * MSE_DEGRADATION_TOLERANCE:
                    logger.info(f"  Refinement hurt (MSE {mse_before:.4f} -> {mse_after:.4f}), reverting...")
                    self.best_ever.model.load_state_dict(backup_state)
                else:
                    logger.info(f"  Refinement: MSE {mse_before:.4f} -> {mse_after:.4f}")
        
        # Final evaluation
        logger.info("-"*60)
        logger.info(f"Training complete in {elapsed:.1f}s")
        
        # Get best model
        best_model = self.best_ever.model if self.best_ever else self.population[0].model
        
        # Note: snap_to_discrete is intentionally NOT called here.
        # The continuous parameters are more expressive and give better results.
        # Formula extraction still works correctly with continuous values.
        
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
        
        # Note: Compiled inference path is disabled due to MSE discrepancy bugs.
        use_compiled = False
        
        # Squeeze y for consistent shape
        y_eval = y.squeeze()
        
        with torch.no_grad():
            if use_compiled and hasattr(best_model, 'forward_compiled'):
                pred = best_model.forward_compiled(x)
            else:
                pred, _ = best_model(x, hard=True)
            pred = pred.squeeze()
            
            # Log MSE before denormalization for debugging
            final_mse_raw = F.mse_loss(pred, y_eval).item()
            logger.debug(f"MSE raw (before denorm): {final_mse_raw:.4f}")
            
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
        
        # Sanitize formula for Windows console (replace Unicode chars)
        formula_display = (formula
            .replace('π', 'pi')
            .replace('²', '^2')
            .replace('³', '^3')
            .replace('√', 'sqrt')
            .replace('·', '*')
            .replace('φ', 'phi')
            .replace('ω', 'omega')
        )
        # Remove any other problematic Unicode
        formula_display = formula_display.encode('ascii', 'replace').decode('ascii')
        
        print(f"Final MSE: {final_mse:.4f}")
        print(f"Correlation: {corr:.4f}")
        print(f"Formula: {formula_display}")
        
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
    
    # Sanitize formula for Windows console
    formula_display = (formula
        .replace('π', 'pi')
        .replace('²', '^2')
        .replace('³', '^3')
        .replace('√', 'sqrt')
        .replace('·', '*')
        .replace('φ', 'phi')
        .replace('ω', 'omega')
    )
    formula_display = formula_display.encode('ascii', 'replace').decode('ascii')
    
    print("-"*60)
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Final MSE: {final_mse:.6f}")
    print(f"Correlation: {corr:.4f}")
    print(f"Formula: {formula_display}")
    
    return {
        'model': best_model,
        'history': history,
        'final_mse': final_mse,
        'correlation': corr,
        'formula': formula,
        'training_time': elapsed,
    }
