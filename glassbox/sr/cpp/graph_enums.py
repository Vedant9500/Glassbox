"""Graph type IDs shared by Python seed/export code.

Must match declaration order of NodeType / UnaryOp / BinaryOp in ast.h.
Single source of truth for seed_graph_builder and export_pytorch (P6-013).
"""

from __future__ import annotations

from enum import IntEnum


class NodeType(IntEnum):
    INPUT = 0
    CONSTANT = 1
    UNARY = 2
    BINARY = 3


class UnaryOp(IntEnum):
    PERIODIC = 0
    POWER = 1
    INTPOW = 2
    EXP = 3
    LOG = 4
    ABS = 5


class BinaryOp(IntEnum):
    ARITHMETIC = 0
    DIVISION = 1
    AGGREGATION = 2


# NodeType
TYPE_INPUT = NodeType.INPUT
TYPE_CONSTANT = NodeType.CONSTANT
TYPE_UNARY = NodeType.UNARY
TYPE_BINARY = NodeType.BINARY

# UnaryOp
UNARY_PERIODIC = UnaryOp.PERIODIC
UNARY_POWER = UnaryOp.POWER
UNARY_INTPOW = UnaryOp.INTPOW
UNARY_EXP = UnaryOp.EXP
UNARY_LOG = UnaryOp.LOG
UNARY_ABS = UnaryOp.ABS

# BinaryOp
BINARY_ARITHMETIC = BinaryOp.ARITHMETIC
BINARY_DIVISION = BinaryOp.DIVISION
BINARY_AGGREGATION = BinaryOp.AGGREGATION


def validate_node_type(value: int) -> NodeType:
    """M-156: membership-checked NodeType (typos fail loud, not silent int)."""
    try:
        return NodeType(int(value))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid NodeType: {value!r}") from exc


def validate_unary_op(value: int) -> UnaryOp:
    """M-156: membership-checked UnaryOp."""
    try:
        return UnaryOp(int(value))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid UnaryOp: {value!r}") from exc


def validate_binary_op(value: int) -> BinaryOp:
    """M-156: membership-checked BinaryOp."""
    try:
        return BinaryOp(int(value))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid BinaryOp: {value!r}") from exc

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
    "BinaryOp",
    "NodeType",
    "UnaryOp",
    "validate_binary_op",
    "validate_node_type",
    "validate_unary_op",
]
