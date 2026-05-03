"""
Evolution module for Glassbox.

This module contains the evolutionary training algorithms for Operation-Based Neural Networks.
"""

from .evolution import (
    EvolutionaryONNTrainer,
    train_onn_evolutionary,
    train_onn_hybrid,
    finalize_model_coefficients,
    detect_dominant_frequency,
    seed_omega_from_fft,
    anneal_tau,
    anneal_entropy_weight,
    set_model_tau,
    Individual,
    StructureConfidenceTracker,
    random_operation_init,
    seed_population_from_classifier,
    mutate_operations,
    mutate_operations_lamarckian,
    mutate_operations_gradient_informed,
    refine_constants,
    quick_refine_internal,
    calculate_complexity,
    prune_small_coefficients,
    adaptive_coefficient_pruning,
    check_structure_quality,
    intensive_coefficient_refinement,
    ablate_and_select_terms,
)

__all__ = [
    'EvolutionaryONNTrainer',
    'train_onn_evolutionary',
    'train_onn_hybrid',
    'finalize_model_coefficients',
    'detect_dominant_frequency',
    'seed_omega_from_fft',
    'anneal_tau',
    'anneal_entropy_weight',
    'set_model_tau',
    'Individual',
    'StructureConfidenceTracker',
    'random_operation_init',
    'seed_population_from_classifier',
    'mutate_operations',
    'mutate_operations_lamarckian',
    'mutate_operations_gradient_informed',
    'refine_constants',
    'quick_refine_internal',
    'calculate_complexity',
    'prune_small_coefficients',
    'adaptive_coefficient_pruning',
    'check_structure_quality',
    'intensive_coefficient_refinement',
    'ablate_and_select_terms',
]
