"""
Teacher Equation Generator for Adversarial Self-Play

This module defines an RNN-based reinforcement learning agent (the Teacher)
that generates increasingly complex mathematical formulas as sequences of tokens.
The token sequences are parsed into callable Python functions which are used
to generate training data (X, Y curves) for the Curve Classifier (the Student).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PrefixState:
    """Tracks the required number of children to complete a prefix expression."""
    def __init__(self, batch_size, device):
        # Outstanding children required. 
        # Start at 1 because we need 1 root node to complete an expression.
        self.needed = torch.ones(batch_size, dtype=torch.long, device=device)
        self.finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def update(self, tokens_idx):
        """Update the required children based on the arity of the sampled token."""
        # Arity dictionary mapping token indices to how many children they need
        arity_map = {
            TOKEN_TO_IDX['+']: 2, TOKEN_TO_IDX['-']: 2, 
            TOKEN_TO_IDX['*']: 2, TOKEN_TO_IDX['/']: 2,
            TOKEN_TO_IDX['pow']: 2,
            TOKEN_TO_IDX['sin']: 1, TOKEN_TO_IDX['cos']: 1, 
            TOKEN_TO_IDX['exp']: 1, TOKEN_TO_IDX['log']: 1, 
            TOKEN_TO_IDX['sqrt']: 1, TOKEN_TO_IDX['abs']: 1,
            # Everything else (constants, variables, EOS, PAD) has arity 0
        }
        
        # Default arity is 0
        arities = torch.zeros_like(tokens_idx)
        for token_idx, arity in arity_map.items():
            arities[tokens_idx == token_idx] = arity
            
        # We consumed one requirement by placing a token.
        # Then we added 'arity' new requirements.
        self.needed = self.needed - 1 + arities
        
        # If needed == 0, the expression is syntactically complete.
        self.finished = self.finished | (self.needed <= 0)

# Define the Token Vocabulary
# We map token strings to indices.
TOKENS = [
    '<PAD>', '<EOS>', # Control tokens
    '+', '-', '*', '/', 'pow', # Binary operators
    'sin', 'cos', 'exp', 'log', 'sqrt', 'abs', # Unary operators
    'x', # The variable
    # We include a few common constants. The teacher can combine them.
    '1.0', '-1.0', '2.0', '0.5', '3.0', 'pi', 'e' 
]
TOKEN_TO_IDX = {t: i for i, t in enumerate(TOKENS)}
IDX_TO_TOKEN = {i: t for t, i in TOKEN_TO_IDX.items()}

# Define which tokens belong to which operator class for ground truth labeling
OPERATOR_LABELS = {
    'identity': ['x'],
    'sin': ['sin'],
    'cos': ['cos'],
    'power': ['sqrt', 'pow'], # x^p includes x^2, x^3, sqrt(x), etc.
    'exp': ['exp'],
    'log': ['log'],
    'addition': ['+', '-'],
    'multiplication': ['*'],
    'rational': ['/'],
    'const_pi': ['pi'],
    'const_e': ['e'],
    'const_1': ['1.0', '-1.0'],
    'const_2': ['2.0'],
    'const_half': ['0.5']
}

class TeacherRNN(nn.Module):
    """
    An autoregressive RNN that generates a sequence of tokens representing a formula.
    Uses REINFORCE (Policy Gradient) to optimize generation based on environmental rewards.
    """
    def __init__(self, vocab_size=len(TOKENS), embed_size=64, hidden_size=128, max_len=20):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=TOKEN_TO_IDX['<PAD>'])
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden):
        # x: (batch_size, 1) - the previous token
        embedded = self.embedding(x) # (batch_size, 1, embed_size)
        output, hidden = self.gru(embedded, hidden)
        logits = self.fc(output.squeeze(1)) # (batch_size, vocab_size)
        return logits, hidden
        
    def sample(self, batch_size=1, device='cpu', temperature=1.0):
        """
        Sample a batch of token sequences using ancestral sampling.
        Returns:
            sequences: List of lists of token indices
            log_probs: Tensor of shape (batch_size, max_len) containing log_probs of chosen tokens.
            entropies: Tensor of shape (batch_size, max_len) containing the entropy of the action distribution.
        """
        sequences = []
        
        # Initial input is <EOS> (used as START token here for simplicity)
        curr_token = torch.full((batch_size, 1), TOKEN_TO_IDX['<EOS>'], dtype=torch.long, device=device)
        hidden = torch.zeros(1, batch_size, self.hidden_size, device=device)
        
        seqs_tensor = torch.zeros(batch_size, self.max_len, dtype=torch.long, device=device)
        log_probs_tensor = torch.zeros(batch_size, self.max_len, device=device)
        entropies_tensor = torch.zeros(batch_size, self.max_len, device=device)
        
        # Track which sequences have hit EOS
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        prefix_state = PrefixState(batch_size, device)
        
        for step in range(self.max_len):
            logits, hidden = self.forward(curr_token, hidden)
            
            # --- Syntax Masking ---
            # If we STILL need children, we cannot output <EOS> or <PAD>
            # If we need exactly 0 children (finished), we MUST output <EOS>
            # If we are at the last step (max_len-1) and need more than 0 children, 
            # we must output something with arity 0 to close tree.
            
            mask = torch.zeros_like(logits, dtype=torch.bool)
            
            # Case 1: Finished -> MUST output EOS
            finished_mask = prefix_state.finished
            mask[finished_mask, :] = True
            mask[finished_mask, TOKEN_TO_IDX['<EOS>']] = False # Allow EOS
            
            # Case 2: Unfinished -> Cannot output EOS/PAD
            unfinished_mask = ~prefix_state.finished
            mask[unfinished_mask, TOKEN_TO_IDX['<EOS>']] = True
            mask[unfinished_mask, TOKEN_TO_IDX['<PAD>']] = True
            
            # Case 3: Running out of space -> Force terminals (arity 0)
            if step >= self.max_len - 1:
               # Block everything except terminals AND EOS/PAD
               for tok_idx, tok_str in IDX_TO_TOKEN.items():
                   if tok_str in ['+', '-', '*', '/', 'sin', 'cos', 'exp', 'log', 'sqrt', 'abs']:
                       mask[unfinished_mask, tok_idx] = True
            
            # Apply mask (set logits to -inf)
            logits.masked_fill_(mask, -1e9)
            # ---------------------
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
                
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            
            # Sample next token
            next_token = dist.sample()
            
            # If the expression is syntactically finished but EOS wasn't sampled yet, force PAD
            # Actually, our masking forces EOS on the step *after* finishing, 
            # and then they will be finished = True for the rest.
            
            log_prob = dist.log_prob(next_token)
            entropy = dist.entropy()
            
            # If a sequence already finished, zero its log-prob and entropy
            # so post-finish forced tokens don't pollute the policy gradient
            log_prob = torch.where(finished, torch.tensor(0.0, device=device), log_prob)
            entropy = torch.where(finished, torch.tensor(0.0, device=device), entropy)
            
            seqs_tensor[:, step] = next_token
            log_probs_tensor[:, step] = log_prob
            entropies_tensor[:, step] = entropy
            
            curr_token = next_token.unsqueeze(1)
            
            # Update prefix state
            prefix_state.update(next_token)
            
            # Update finished mask
            finished = finished | (next_token == TOKEN_TO_IDX['<EOS>'])
            
            if finished.all() and prefix_state.finished.all():
                break
                
        # Convert tensor to lists of strings
        seqs_np = seqs_tensor.cpu().numpy()
        for i in range(batch_size):
            seq = []
            for idx in seqs_np[i]:
                if idx == TOKEN_TO_IDX['<EOS>']:
                    break
                if idx != TOKEN_TO_IDX['<PAD>']:
                    seq.append(IDX_TO_TOKEN[idx])
            sequences.append(seq)
            
        return sequences, log_probs_tensor, entropies_tensor

# --- Prefix to Infix/Evaluation parsing ---

def parse_prefix(tokens):
    """
    Parses a sequence of tokens in prefix notation into an intermediate AST.
    Tokens must form a valid prefix expression. Returns (ast, remaining_tokens).
    """
    if not tokens:
        raise ValueError("Unexpected end of expression")
        
    token = tokens.pop(0)
    
    if token in ['+', '-', '*', '/', 'pow']:
        # Binary operator: needs two children
        left_ast, tokens = parse_prefix(tokens)
        right_ast, tokens = parse_prefix(tokens)
        return (token, left_ast, right_ast), tokens
        
    elif token in ['sin', 'cos', 'exp', 'log', 'sqrt', 'abs']:
        # Unary operator: needs one child
        child_ast, tokens = parse_prefix(tokens)
        return (token, child_ast), tokens
        
    else:
        # Terminal (variable or constant)
        return token, tokens

def evaluate_ast(ast_node, x_val):
    """
    Recursively evaluate the AST over a numpy array X.
    """
    if isinstance(ast_node, tuple):
        op = ast_node[0]
        if len(ast_node) == 3:
            # Binary
            left = evaluate_ast(ast_node[1], x_val)
            right = evaluate_ast(ast_node[2], x_val)
            
            if op == '+': return left + right
            elif op == '-': return left - right
            elif op == '*': return left * right
            elif op == '/':
                # Safe division — preserve sign of denominator
                denom = np.where(np.abs(right) < 1e-6, np.sign(right + 1e-12) * 1e-6, right)
                return left / denom
            elif op == 'pow':
                # Safe power: clip exponent to prevent overflow
                return np.power(np.abs(left) + 1e-8, np.clip(right, -5, 5))
        elif len(ast_node) == 2:
            # Unary
            child = evaluate_ast(ast_node[1], x_val)
            
            if op == 'sin': return np.sin(child)
            elif op == 'cos': return np.cos(child)
            elif op == 'exp': return np.exp(np.clip(child, -10, 10)) # Safe exp
            elif op == 'log': return np.log(np.abs(child) + 1e-6) # Safe log
            elif op == 'sqrt': return np.sqrt(np.abs(child)) # Safe sqrt
            elif op == 'abs': return np.abs(child)
    else:
        # Terminal
        if ast_node == 'x': return np.copy(x_val) # Avoid mutating x
        elif ast_node == 'pi': return np.full_like(x_val, np.pi, dtype=float)
        elif ast_node == 'e': return np.full_like(x_val, np.e, dtype=float)
        else:
            # Hopefully a float constant
            try:
                val = float(ast_node)
                return np.full_like(x_val, val, dtype=float)
            except ValueError:
                return np.zeros_like(x_val) # Failed to parse

def tokens_to_curve(tokens, x_range=(-5, 5), n_points=256):
    """
    Takes a list of tokens, parses them as a prefix expression, and returns the Y curve.
    Returns (y_curve, success_bool)
    """
    tokens_copy = list(tokens)
    
    # We require the sequence to be a valid, complete prefix expression.
    try:
        ast, remaining = parse_prefix(tokens_copy)
        if len(remaining) > 0:
            # Didn't consume all tokens - invalid formula format
            return None, False
            
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = evaluate_ast(ast, x)
        
        # Check for NaNs or Infs
        if not np.all(np.isfinite(y)):
            return None, False
            
        # Prevent extreme values that cause float64 overflow in standard deviation
        # np.std squares values internally, so y > 1e154 overflows to inf.
        if np.max(np.abs(y)) > 1e20:
            return None, False
            
        # Check for constant curves (e.g., generated '1.0 + 1.0') -> Student learns nothing
        if np.std(y) < 1e-8:
            return None, False
            
        return y, True
        
    except (ValueError, RecursionError):
        # RecursionError: Too deep / invalid format
        # ValueError: Unexpected end of expression
        return None, False

def get_ground_truth_labels(tokens, operator_classes_dict):
    """
    Given an accepted token list, determine which generic operator classes are present.
    Matches the schema expected by CurveClassifier.
    Returns a multi-hot numpy array of shape (N_CLASSES,)
    """
    # Create the multi-hot vector based on the operator_classes mapping
    # e.g. operator_classes_dict might be {'identity': 0, 'sin': 1, ... }
    multi_hot = np.zeros(len(operator_classes_dict), dtype=np.float32)
    
    # Check for presence of tokens that trigger a class
    present_tokens = set(tokens)
    
    for class_name, triggers in OPERATOR_LABELS.items():
        if class_name in operator_classes_dict:
            idx = operator_classes_dict[class_name]
            for trigger in triggers:
                if trigger in present_tokens:
                    multi_hot[idx] = 1.0
                    break
                    
    return multi_hot
