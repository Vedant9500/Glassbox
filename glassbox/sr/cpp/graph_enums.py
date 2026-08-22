"""Graph type IDs shared by Python seed/export code.

Must match declaration order of NodeType / UnaryOp / BinaryOp in ast.h.
Single source of truth for seed_graph_builder and export_pytorch (P6-013).
"""

from __future__ import annotations

# NodeType
TYPE_INPUT = 0
TYPE_CONSTANT = 1
TYPE_UNARY = 2
TYPE_BINARY = 3

# UnaryOp
UNARY_PERIODIC = 0
UNARY_POWER = 1
UNARY_INTPOW = 2
UNARY_EXP = 3
UNARY_LOG = 4
UNARY_ABS = 5

# BinaryOp
BINARY_ARITHMETIC = 0
BINARY_DIVISION = 1
BINARY_AGGREGATION = 2

__all__ = [
    "BINARY_AGGREGATION",
    "BINARY_ARITHMETIC",
    "BINARY_DIVISION",
    "TYPE_BINARY",
    "TYPE_CONSTANT",
    "TYPE_INPUT",
    "TYPE_UNARY",
    "UNARY_ABS",
    "UNARY_EXP",
    "UNARY_INTPOW",
    "UNARY_LOG",
    "UNARY_PERIODIC",
    "UNARY_POWER",
]
