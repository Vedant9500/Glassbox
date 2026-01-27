"""
Glassbox Formula Tester - Interactive UI + Full Visualization

A tool for testing if the ONN can discover mathematical formulas.
Uses a Tkinter control panel + the beautiful standalone visualization window.

Usage:
    python scripts/formula_tester.py

Author: Glassbox Project
"""

import sys
import os
import re
import threading
from typing import Callable, Dict, Tuple, Optional, List, Set
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np

from glassbox.sr.operation_dag import OperationDAG
from glassbox.sr.evolution import EvolutionaryONNTrainer
from glassbox.sr.visualization import LiveTrainingVisualizer

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    HAS_TK = True
except ImportError:
    HAS_TK = False

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


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
# FORMULA TESTER (with Phase 2 regression)
# =============================================================================

class FormulaTester:
    """Two-phase formula testing: Structure Discovery + Exact Coefficients."""
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        generations: int = 80,
        population_size: int = 20,
        n_hidden_layers: int = 1,
        nodes_per_layer: int = 6,
        x_range: Tuple[float, float] = (-6, 6),
        n_samples: int = 300,
        explorer_fraction: float = 0.40,
        mutation_rate: float = 0.4,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generations = generations
        self.population_size = population_size
        self.n_hidden_layers = n_hidden_layers
        self.nodes_per_layer = nodes_per_layer
        self.x_range = x_range
        self.n_samples = n_samples
        self.explorer_fraction = explorer_fraction
        self.mutation_rate = mutation_rate
        self.use_simplified_ops = False  # Set to False for formulas with exp/log
        
        self.phase1_model = None
        self.phase1_formula = ""
        self.phase1_mse = float('inf')
        self.phase2_formula = ""
        self.phase2_mse = float('inf')
        self.coefficients: Dict[str, float] = {}
    
    def make_model(self) -> OperationDAG:
        return OperationDAG(
            n_inputs=1,
            n_hidden_layers=self.n_hidden_layers,
            nodes_per_layer=self.nodes_per_layer,
            n_outputs=1,
            simplified_ops=self.use_simplified_ops,
            fair_mode=True,
        )
    
    def generate_data(self, target_fn: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.linspace(self.x_range[0], self.x_range[1], self.n_samples).reshape(-1, 1)
        y = target_fn(x)
        return x, y
    
    def run_phase1(self, x: torch.Tensor, y: torch.Tensor, visualizer=None) -> Dict:
        """Phase 1: Structure Discovery using evolutionary training."""
        trainer = EvolutionaryONNTrainer(
            model_factory=self.make_model,
            population_size=self.population_size,
            elite_size=4,
            mutation_rate=self.mutation_rate,
            constant_refine_steps=30,
            complexity_penalty=0.01,
            device=self.device,
            lamarckian=True,
            use_explorers=True,
            explorer_fraction=self.explorer_fraction,
            explorer_mutation_rate=0.85,
            prune_coefficients=False,
            constant_refine_hard=True,
            visualizer=visualizer,
            # Enable risk-seeking policy gradient for stuck detection
            risk_seeking=True,
            risk_seeking_percentile=0.1,  # Focus on top 10%
        )
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        return trainer.train(x, y, generations=self.generations, print_every=5)

    def run_phase2(self, x: torch.Tensor, y: torch.Tensor, phase1_formula: str) -> Dict:
        """Phase 2: Pure Basis Regression for exact coefficients."""
        formula_lower = phase1_formula.lower()
        discovered_ops = set()
        
        if 'sin' in formula_lower or 'cos' in formula_lower:
            discovered_ops.add('periodic')
        if '^2' in formula_lower or '^3' in formula_lower or '**' in formula_lower:
            discovered_ops.add('power')
        if '*' in formula_lower and 'x' in formula_lower:
            if re.search(r'x\d?\s*\*\s*(sin|cos)', formula_lower) or \
               re.search(r'(sin|cos).*\*\s*x', formula_lower):
                discovered_ops.add('x_times_trig')
        discovered_ops.add('linear')
        
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
        features_with_bias = torch.cat([features_matrix, torch.ones(n_samples, 1)], dim=1)
        
        solution = torch.linalg.lstsq(features_with_bias, y)
        weights = solution.solution.squeeze()
        
        pred = features_with_bias @ weights
        phase2_mse = torch.nn.functional.mse_loss(pred.squeeze(), y.squeeze()).item()
        
        self.coefficients = {}
        for name, w in zip(feature_names, weights[:-1]):
            if abs(w.item()) > 0.01:
                self.coefficients[name] = w.item()
        bias = weights[-1].item()
        if abs(bias) > 0.01:
            self.coefficients['bias'] = bias
        
        formula_parts = []
        for name, w in zip(feature_names, weights[:-1]):
            w_val = w.item()
            if abs(w_val) > 0.01:
                formula_parts.append(f"{w_val:.2f}*{name}")
        if abs(bias) > 0.01:
            formula_parts.append(f"{bias:.2f}")
        
        phase2_formula = " + ".join(formula_parts) if formula_parts else "0"
        
        return {'formula': phase2_formula, 'mse': phase2_mse, 'coefficients': self.coefficients}
    
    def run_full_pipeline(self, target_formula: str, visualizer=None) -> Dict:
        target_fn = parse_formula(target_formula)
        x, y = self.generate_data(target_fn)
        
        phase1_results = self.run_phase1(x, y, visualizer=visualizer)
        
        self.phase1_model = phase1_results['model']
        self.phase1_formula = phase1_results['formula']
        self.phase1_mse = phase1_results['final_mse']
        
        print("\n" + "="*60)
        print("PHASE 2: EXACT COEFFICIENT EXTRACTION")
        print("="*60)
        
        phase2_results = self.run_phase2(x, y, self.phase1_formula)
        self.phase2_formula = phase2_results['formula']
        self.phase2_mse = phase2_results['mse']
        
        print(f"Phase 2 Formula: {self.phase2_formula}")
        print(f"Phase 2 MSE: {self.phase2_mse:.6f}")
        
        validation = self._validate_results(target_formula)
        
        return {
            'target_formula': target_formula,
            'phase1': {'formula': self.phase1_formula, 'mse': self.phase1_mse},
            'phase2': {'formula': self.phase2_formula, 'mse': self.phase2_mse, 'coefficients': self.coefficients},
            'validation': validation,
        }
    
    def _validate_results(self, target_formula: str) -> Dict:
        target_fn = parse_formula(target_formula)
        test_points = [0.0, 1.0, 2.0, 3.0, 5.0]
        validation = {'points': [], 'success': False}
        
        for x_val in test_points:
            x_test = torch.tensor([[x_val]])
            y_true = target_fn(x_test).item()
            
            y_pred = 0.0
            for name, coef in self.coefficients.items():
                if name == 'x': y_pred += coef * x_val
                elif name == 'x²': y_pred += coef * x_val**2
                elif name == 'x³': y_pred += coef * x_val**3
                elif name == 'sin(x)': y_pred += coef * math.sin(x_val)
                elif name == 'cos(x)': y_pred += coef * math.cos(x_val)
                elif name == 'x·sin(x)': y_pred += coef * x_val * math.sin(x_val)
                elif name == 'x·cos(x)': y_pred += coef * x_val * math.cos(x_val)
                elif name == 'bias': y_pred += coef
            
            error = abs(y_true - y_pred)
            validation['points'].append({'x': x_val, 'y_true': y_true, 'y_pred': y_pred, 'error': error})
        
        max_error = max(p['error'] for p in validation['points'])
        validation['max_error'] = max_error
        validation['success'] = max_error < 0.1
        return validation


# =============================================================================
# CONTROL PANEL GUI (launches visualization in separate window)
# =============================================================================

class ControlPanelGUI:
    """
    Simple control panel for settings.
    Launches the full visualization in a separate matplotlib window.
    """
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Glassbox Formula Tester")
        self.root.geometry("320x550")
        self.root.configure(bg='#0a0a0a')
        self.root.resizable(False, False)
        
        self.is_training = False
        self._setup_styles()
        self._setup_ui()
    
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        bg, fg = '#0a0a0a', '#00ff00'
        style.configure('TFrame', background=bg)
        style.configure('TLabel', background=bg, foreground=fg, font=('Consolas', 10))
        style.configure('Header.TLabel', font=('Consolas', 16, 'bold'), foreground=fg, background=bg)
        style.configure('TLabelframe', background=bg, foreground=fg)
        style.configure('TLabelframe.Label', background=bg, foreground=fg, font=('Consolas', 11, 'bold'))
    
    def _setup_ui(self):
        main = ttk.Frame(self.root, padding="10")
        main.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main, text="🔬 GLASSBOX", style='Header.TLabel').pack(pady=(0, 5))
        ttk.Label(main, text="Formula Tester", font=('Consolas', 12)).pack(pady=(0, 15))
        
        # Formula
        f = ttk.LabelFrame(main, text="Target Formula", padding="8")
        f.pack(fill=tk.X, pady=5)
        self.formula_var = tk.StringVar(value="sin(x) + x^2")
        tk.Entry(f, textvariable=self.formula_var, font=('Consolas', 12),
            bg='#16213e', fg='#00ff00', insertbackground='#00ff00').pack(fill=tk.X)
        ttk.Label(f, text="Examples: x^2, sin(x)+x^2, x*cos(x)", font=('Consolas', 8)).pack(pady=2)
        
        # Architecture
        f = ttk.LabelFrame(main, text="Network", padding="8")
        f.pack(fill=tk.X, pady=5)
        
        row = ttk.Frame(f)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Hidden Layers:").pack(side=tk.LEFT)
        self.layers_var = tk.IntVar(value=1)
        tk.Spinbox(row, from_=1, to=3, textvariable=self.layers_var, width=5,
            bg='#16213e', fg='#00ff00', font=('Consolas', 10)).pack(side=tk.RIGHT)
        
        row = ttk.Frame(f)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Nodes/Layer:").pack(side=tk.LEFT)
        self.nodes_var = tk.IntVar(value=6)
        tk.Spinbox(row, from_=4, to=12, textvariable=self.nodes_var, width=5,
            bg='#16213e', fg='#00ff00', font=('Consolas', 10)).pack(side=tk.RIGHT)
        
        # Training
        f = ttk.LabelFrame(main, text="Training", padding="8")
        f.pack(fill=tk.X, pady=5)
        
        row = ttk.Frame(f)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Generations:").pack(side=tk.LEFT)
        self.gen_var = tk.IntVar(value=80)
        tk.Spinbox(row, from_=20, to=200, increment=10, textvariable=self.gen_var, width=5,
            bg='#16213e', fg='#00ff00', font=('Consolas', 10)).pack(side=tk.RIGHT)
        
        row = ttk.Frame(f)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Population:").pack(side=tk.LEFT)
        self.pop_var = tk.IntVar(value=20)
        tk.Spinbox(row, from_=10, to=50, increment=5, textvariable=self.pop_var, width=5,
            bg='#16213e', fg='#00ff00', font=('Consolas', 10)).pack(side=tk.RIGHT)
        
        row = ttk.Frame(f)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Explorer %:").pack(side=tk.LEFT)
        self.explorer_var = tk.IntVar(value=40)
        tk.Spinbox(row, from_=10, to=60, increment=5, textvariable=self.explorer_var, width=5,
            bg='#16213e', fg='#00ff00', font=('Consolas', 10)).pack(side=tk.RIGHT)
        
        row = ttk.Frame(f)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Mutation:").pack(side=tk.LEFT)
        self.mutation_var = tk.DoubleVar(value=0.4)
        tk.Spinbox(row, from_=0.1, to=0.8, increment=0.1, textvariable=self.mutation_var, width=5,
            bg='#16213e', fg='#00ff00', font=('Consolas', 10), format="%.1f").pack(side=tk.RIGHT)
        
        # Button
        self.start_btn = tk.Button(main, text="▶ START TRAINING", command=self._start_training,
            font=('Consolas', 14, 'bold'), bg='#0f3460', fg='#00ff00',
            activebackground='#16213e', activeforeground='#00ff00', height=2)
        self.start_btn.pack(fill=tk.X, pady=15)
        
        # Results
        f = ttk.LabelFrame(main, text="Results", padding="8")
        f.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.results_text = tk.Text(f, height=8, font=('Consolas', 9),
            bg='#16213e', fg='#00ff00', wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        self.results_text.insert(tk.END, "Click START to begin.\n\nA visualization window will open showing the network architecture, training progress, and discovered formula.")
        self.results_text.config(state=tk.DISABLED)
    
    def _start_training(self):
        if self.is_training:
            messagebox.showinfo("Info", "Training already in progress")
            return
        
        formula = self.formula_var.get().strip()
        if not formula:
            messagebox.showerror("Error", "Please enter a formula")
            return
        
        try:
            fn = parse_formula(formula)
            _ = fn(torch.tensor([[0.0]]))
        except Exception as e:
            messagebox.showerror("Error", f"Invalid formula: {e}")
            return
        
        self.is_training = True
        self.start_btn.config(state=tk.DISABLED, text="Training...")
        
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Starting training...\n\nVisualization window opening...")
        self.results_text.config(state=tk.DISABLED)
        
        # Run training in background thread
        training_thread = threading.Thread(target=self._run_training, args=(formula,), daemon=True)
        training_thread.start()
    
    def _run_training(self, formula: str):
        try:
            # Create visualizer (opens in separate window)
            visualizer = LiveTrainingVisualizer(
                update_every=1,
                figsize=(14, 8),
                lite_mode=False,
            )
            
            # Create tester with current settings
            tester = FormulaTester(
                generations=self.gen_var.get(),
                population_size=self.pop_var.get(),
                n_hidden_layers=self.layers_var.get(),
                nodes_per_layer=self.nodes_var.get(),
                explorer_fraction=self.explorer_var.get() / 100.0,
                mutation_rate=self.mutation_var.get(),
            )
            
            # Run training with visualization
            results = tester.run_full_pipeline(formula, visualizer=visualizer)
            
            # Show results in main thread
            self.root.after(0, lambda: self._show_results(results))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: (
                setattr(self, 'is_training', False),
                self.start_btn.config(state=tk.NORMAL, text="▶ START TRAINING")
            ))
    
    def _show_results(self, results: Dict):
        v = results['validation']
        status = "✓ SUCCESS" if v['success'] else "✗ PARTIAL"
        
        text = f"""{'='*30}
TARGET: {results['target_formula']}
{'='*30}

PHASE 1 (Structure):
  {results['phase1']['formula'][:40]}
  MSE: {results['phase1']['mse']:.4f}

PHASE 2 (Exact):
  {results['phase2']['formula']}
  MSE: {results['phase2']['mse']:.6f}

{status}
Max error: {v['max_error']:.6f}
"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.config(state=tk.DISABLED)


# =============================================================================
# CLI
# =============================================================================

def run_cli(formula: str, generations: int = 80, visualize: bool = True):
    print("="*60)
    print("GLASSBOX FORMULA TESTER")
    print("="*60)
    print(f"Target: {formula}")
    print("-"*60)
    
    visualizer = None
    if visualize:
        visualizer = LiveTrainingVisualizer(update_every=1, figsize=(14, 8), lite_mode=False)
    
    tester = FormulaTester(generations=generations)
    results = tester.run_full_pipeline(formula, visualizer=visualizer)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Phase 1:    {results['phase1']['formula']}")
    print(f"Phase 1 MSE: {results['phase1']['mse']:.4f}")
    print(f"Phase 2:    {results['phase2']['formula']}")
    print(f"Phase 2 MSE: {results['phase2']['mse']:.6f}")
    print(f"\nStatus: {'✓ SUCCESS' if results['validation']['success'] else '✗ PARTIAL'}")
    print(f"Max error: {results['validation']['max_error']:.6f}")
    
    if visualize:
        print("\nClose the visualization window to exit.")
        plt.show(block=True)
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Glassbox Formula Tester")
    parser.add_argument('--formula', '-f', type=str, help='Formula to test')
    parser.add_argument('--no-gui', action='store_true', help='CLI mode')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    parser.add_argument('--generations', '-g', type=int, default=80)
    args = parser.parse_args()
    
    if args.formula:
        run_cli(args.formula, generations=args.generations, visualize=not args.no_viz)
    elif args.no_gui:
        formula = input("Enter formula: ").strip()
        if formula:
            run_cli(formula, generations=args.generations, visualize=not args.no_viz)
    else:
        if not HAS_TK:
            print("GUI requires tkinter. Use --no-gui for CLI mode.")
            sys.exit(1)
        root = tk.Tk()
        app = ControlPanelGUI(root)
        root.mainloop()


if __name__ == "__main__":
    main()
