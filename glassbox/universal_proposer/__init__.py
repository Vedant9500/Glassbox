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
    validate_fpip_v2_payload,
    DEFAULT_OPERATOR_VOCAB,
    DEFAULT_SKELETON_VOCAB,
)

__all__ = [
    'UniversalProposerConfig',
    'UniversalProposer',
    'load_universal_proposer_checkpoint',
    'propose_fpip_v2_from_xy',
    'propose_from_xy',
    'proposer_output_to_fpip_v2',
    'validate_fpip_v2_payload',
    'DEFAULT_OPERATOR_VOCAB',
    'DEFAULT_SKELETON_VOCAB',
]
