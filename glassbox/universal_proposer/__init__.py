"""
Universal Proposer module for Glassbox.

This module contains the universal proposer for fast-path skeleton generation.
"""

from .universal_proposer import (
    UniversalProposerConfig,
    UniversalProposer,
    load_universal_proposer_checkpoint,
    propose_fpip_v2_from_xy,
    propose_from_xy,
    proposer_output_to_fpip_v2,
    grammar_decode_topk_skeletons,
    grammar_decode_multivariate_skeletons,
    build_search_plan,
    build_multivariate_search_plan,
    validate_fpip_v2_payload,
    validate_universal_proposer_checkpoint_metadata,
    DEFAULT_OPERATOR_VOCAB,
    DEFAULT_SKELETON_VOCAB,
    DEFAULT_UNIVARIATE_SKELETON_VOCAB,
    DEFAULT_MULTIVARIATE_SKELETON_VOCAB,
    UNIVERSAL_PROPOSER_ARCHITECTURE_VERSION,
    UNIVERSAL_PROPOSER_CONTRACT_VERSION,
    UNIVERSAL_PROPOSER_MULTIVARIATE_CONTRACT_VERSION,
    UNIVERSAL_PROPOSER_MULTIVARIATE_NEURAL_MODE,
    UNIVERSAL_PROPOSER_ROLE,
    UNIVERSAL_PROPOSER_UNIVARIATE_NEURAL_MODE,
    normalize_formula_key,
)

__all__ = [
    'UniversalProposerConfig',
    'UniversalProposer',
    'load_universal_proposer_checkpoint',
    'propose_fpip_v2_from_xy',
    'propose_from_xy',
    'proposer_output_to_fpip_v2',
    'grammar_decode_topk_skeletons',
    'grammar_decode_multivariate_skeletons',
    'build_search_plan',
    'build_multivariate_search_plan',
    'validate_fpip_v2_payload',
    'validate_universal_proposer_checkpoint_metadata',
    'DEFAULT_OPERATOR_VOCAB',
    'DEFAULT_SKELETON_VOCAB',
    'DEFAULT_UNIVARIATE_SKELETON_VOCAB',
    'DEFAULT_MULTIVARIATE_SKELETON_VOCAB',
    'UNIVERSAL_PROPOSER_ARCHITECTURE_VERSION',
    'UNIVERSAL_PROPOSER_CONTRACT_VERSION',
    'UNIVERSAL_PROPOSER_MULTIVARIATE_CONTRACT_VERSION',
    'UNIVERSAL_PROPOSER_MULTIVARIATE_NEURAL_MODE',
    'UNIVERSAL_PROPOSER_ROLE',
    'UNIVERSAL_PROPOSER_UNIVARIATE_NEURAL_MODE',
    'normalize_formula_key',
]
