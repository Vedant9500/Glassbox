import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from scripts.teacher_generator import (
    TeacherRNN, TOKENS, TOKEN_TO_IDX, IDX_TO_TOKEN, 
    parse_prefix, evaluate_ast, tokens_to_curve, get_ground_truth_labels
)

def test_ast_evaluation():
    # Test valid expression
    # sin ( + x pi )
    tokens = ['sin', '+', 'x', 'pi']
    x_val = np.array([0, np.pi/2, np.pi])
    
    ast, remaining = parse_prefix(tokens)
    assert len(remaining) == 0
    
    y = evaluate_ast(ast, x_val)
    # sin(0 + pi) = 0
    # sin(pi/2 + pi) = -1
    # sin(pi + pi) = 0
    np.testing.assert_almost_equal(y[0], 0.0)
    np.testing.assert_almost_equal(y[1], -1.0)
    np.testing.assert_almost_equal(y[2], 0.0)
    
def test_invalid_ast():
    tokens = ['+'] # Missing operands
    try:
        y, success = tokens_to_curve(tokens)
        assert not success
    except Exception:
        pass
        
    tokens = ['+', 'x'] # Missing second operand
    try:
        y, success = tokens_to_curve(tokens)
        assert not success
    except Exception:
        pass

def test_teacher_sampling():
    teacher = TeacherRNN(vocab_size=len(TOKENS), max_len=10)
    sequences, log_probs, is_valid_arr = teacher.sample(batch_size=5)
    
    assert len(sequences) == 5
    assert log_probs.shape[0] == 5
    assert log_probs.shape[1] == 10
    
def test_ground_truth_labels():
    operator_classes = {'identity': 0, 'sin': 1, 'addition': 2, 'multiplication': 3}
    
    tokens = ['+', 'sin', 'x', 'x']
    multi_hot = get_ground_truth_labels(tokens, operator_classes)
    
    # identity (x), sin (sin), addition (+) should be 1
    # multiplication (*) should be 0
    assert multi_hot[0] == 1.0
    assert multi_hot[1] == 1.0
    assert multi_hot[2] == 1.0
    assert multi_hot[3] == 0.0

def test_pow_evaluation():
    # Test valid expression
    # pow x 2.0
    tokens = ['pow', 'x', '2.0']
    x_val = np.array([-2, 0, 3])
    
    ast, remaining = parse_prefix(tokens)
    assert len(remaining) == 0
    
    y = evaluate_ast(ast, x_val)
    # (-2)^2 = 4
    # 0^2 = 0
    # 3^2 = 9
    np.testing.assert_almost_equal(y[0], 4.0)
    np.testing.assert_almost_equal(y[1], 0.0)
    np.testing.assert_almost_equal(y[2], 9.0)

def test_safe_division_sign_preservation():
    # / 1.0 -2.0
    tokens = ['/', '1.0', '-2.0']
    x_val = np.array([0])
    
    ast, remaining = parse_prefix(tokens)
    assert len(remaining) == 0
    
    y = evaluate_ast(ast, x_val)
    # 1.0 / -2.0 = -0.5
    np.testing.assert_almost_equal(y[0], -0.5)

def test_power_label_includes_pow():
    operator_classes = {'power': 0, 'identity': 1}
    tokens = ['pow', 'x', '2.0']
    multi_hot = get_ground_truth_labels(tokens, operator_classes)
    
    assert multi_hot[0] == 1.0 # 'power'
    assert multi_hot[1] == 1.0 # 'identity'
