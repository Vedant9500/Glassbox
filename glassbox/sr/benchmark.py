"""
Benchmarking Framework for Operation-Based Neural Networks.

Compare ONN against:
- CNN (Convolutional Neural Network)
- LSTM (Long Short-Term Memory)
- MLP (Multi-Layer Perceptron)

Provides standardized:
- Dataset loading (synthetic + real)
- Training loops
- Evaluation metrics
- Comparison reports
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, List, Tuple, Callable
import time
import copy
import math

# Import our ONN
from .operation_dag import OperationDAG, OperationDAGSimple
from .hybrid_optimizer import HybridOptimizer, LBFGSConstantOptimizer


# ============================================================================
# GPU Utilities
# ============================================================================

def get_device(prefer_gpu: bool = True) -> torch.device:
    """Get the best available device (GPU if available, else CPU)."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def to_device(tensor_or_model, device: torch.device):
    """Move tensor or model to device."""
    return tensor_or_model.to(device)



# ============================================================================
# Baseline Models
# ============================================================================

class BaselineMLP(nn.Module):
    """Simple MLP baseline."""
    
    def __init__(
        self,
        n_inputs: int,
        n_hidden: int = 32,
        n_layers: int = 2,
        n_outputs: int = 1,
    ):
        super().__init__()
        
        layers = [nn.Linear(n_inputs, n_hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(n_hidden, n_hidden), nn.ReLU()])
        layers.append(nn.Linear(n_hidden, n_outputs))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class BaselineLSTM(nn.Module):
    """LSTM baseline for sequence data."""
    
    def __init__(
        self,
        n_inputs: int,
        hidden_size: int = 32,
        n_layers: int = 1,
        n_outputs: int = 1,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=n_inputs,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, n_outputs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_inputs) or (batch, n_inputs)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Last timestep


class BaselineCNN(nn.Module):
    """1D CNN baseline (good for time series and sequences)."""
    
    def __init__(
        self,
        n_inputs: int,
        n_channels: int = 32,
        kernel_size: int = 3,
        n_outputs: int = 1,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(1, n_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(n_channels, n_channels, kernel_size, padding=kernel_size // 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_channels, n_outputs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_inputs) -> (batch, 1, n_inputs)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


# ============================================================================
# Benchmark Datasets
# ============================================================================

def generate_polynomial_data(
    n_samples: int = 500,
    noise_std: float = 0.1,
    formula: str = 'x^2',
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """Generate polynomial function data."""
    x = torch.linspace(-3, 3, n_samples).unsqueeze(-1)
    
    if formula == 'x^2':
        y = x ** 2
        true_formula = "y = x²"
    elif formula == 'x^3':
        y = x ** 3
        true_formula = "y = x³"
    elif formula == 'x^2+x':
        y = x ** 2 + x
        true_formula = "y = x² + x"
    elif formula == 'sin':
        y = torch.sin(x)
        true_formula = "y = sin(x)"
    elif formula == 'sin+x^2':
        y = torch.sin(x) + x ** 2
        true_formula = "y = sin(x) + x²"
    elif formula == 'exp':
        y = torch.exp(-x ** 2)
        true_formula = "y = exp(-x²)"
    else:
        raise ValueError(f"Unknown formula: {formula}")
    
    # Add noise
    y = y + torch.randn_like(y) * noise_std
    
    return x, y, true_formula


def generate_time_series_data(
    n_samples: int = 1000,
    seq_length: int = 20,
    noise_std: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Generate time series prediction data.
    
    Task: Predict next value in sine wave with harmonics.
    """
    t = torch.linspace(0, 20 * math.pi, n_samples + seq_length)
    
    # Sine wave with harmonics
    signal = torch.sin(t) + 0.5 * torch.sin(2 * t) + 0.25 * torch.sin(3 * t)
    signal = signal + torch.randn_like(signal) * noise_std
    
    # Create sequences
    x = []
    y = []
    for i in range(n_samples):
        x.append(signal[i:i+seq_length])
        y.append(signal[i+seq_length])
    
    x = torch.stack(x).unsqueeze(-1)  # (n_samples, seq_length, 1)
    y = torch.stack(y).unsqueeze(-1)  # (n_samples, 1)
    
    return x, y, "Time series: sin + harmonics"


def generate_multivariate_data(
    n_samples: int = 500,
    n_features: int = 3,
    noise_std: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Generate multivariate function data.
    
    y = x1 * x2 + sin(x3) if n_features >= 3
    y = x1 * x2           if n_features == 2
    """
    n_features = max(n_features, 2)  # Minimum 2 features
    x = torch.randn(n_samples, n_features)
    
    if n_features >= 3:
        y = x[:, 0] * x[:, 1] + torch.sin(x[:, 2])
        formula = "y = x₁·x₂ + sin(x₃)"
    else:
        y = x[:, 0] * x[:, 1]
        formula = "y = x₁·x₂"
    
    y = y.unsqueeze(-1)
    y = y + torch.randn_like(y) * noise_std
    
    return x, y, formula


# ============================================================================
# Benchmark Runner
# ============================================================================

class BenchmarkRunner:
    """
    Run standardized benchmarks comparing ONN against baselines.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        epochs: int = 200,
        lr: float = 0.01,
        verbose: bool = True,
    ):
        # Auto-detect device if not specified
        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)
        
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.results: List[Dict] = []
        
        if self.verbose:
            print(f"Using device: {self.device}")
            if self.device.type == 'cuda':
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
                print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def train_model(
        self,
        model: nn.Module,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        model_name: str,
    ) -> Dict:
        """Train a single model and return results."""
        model = model.to(self.device)
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)
        x_val = x_val.to(self.device)
        y_val = y_val.to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        start_time = time.time()
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            # Train
            model.train()
            optimizer.zero_grad()
            
            # Handle different model types
            if isinstance(model, OperationDAG):
                pred, _ = model(x_train, hard=epoch > self.epochs * 0.7)
            else:
                pred = model(x_train)
            
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            # Validate
            model.eval()
            with torch.no_grad():
                if isinstance(model, OperationDAG):
                    val_pred, _ = model(x_val, hard=True)
                else:
                    val_pred = model(x_val)
                val_loss = criterion(val_pred, y_val).item()
                val_losses.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
        
        training_time = time.time() - start_time
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            if isinstance(model, OperationDAG):
                final_pred, _ = model(x_val, hard=True)
            else:
                final_pred = model(x_val)
            
            final_mse = criterion(final_pred, y_val).item()
            
            # Correlation
            pred_flat = final_pred.squeeze().cpu()
            true_flat = y_val.squeeze().cpu()
            if pred_flat.dim() > 0:
                corr = torch.corrcoef(torch.stack([pred_flat, true_flat]))[0, 1].item()
            else:
                corr = 1.0 if pred_flat == true_flat else 0.0
        
        n_params = sum(p.numel() for p in model.parameters())
        
        result = {
            'model_name': model_name,
            'final_mse': final_mse,
            'best_val_loss': best_val_loss,
            'correlation': corr,
            'training_time': training_time,
            'n_params': n_params,
            'train_losses': train_losses,
            'val_losses': val_losses,
        }
        
        if self.verbose:
            print(f"  {model_name}: MSE={final_mse:.4f}, Corr={corr:.4f}, "
                  f"Params={n_params}, Time={training_time:.1f}s")
        
        return result
    
    def run_benchmark(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_name: str,
        train_ratio: float = 0.8,
    ) -> Dict:
        """
        Run benchmark on all models for a given dataset.
        """
        # Split data
        n_train = int(len(x) * train_ratio)
        x_train, x_val = x[:n_train], x[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        # Determine input size
        if x.dim() == 2:
            n_inputs = x.shape[1]
        else:
            n_inputs = x.shape[-1]
        n_outputs = y.shape[-1]
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Benchmark: {task_name}")
            print(f"Data: {len(x)} samples, {n_inputs} inputs → {n_outputs} outputs")
            print(f"{'='*60}")
        
        benchmark_results = {
            'task': task_name,
            'n_samples': len(x),
            'n_inputs': n_inputs,
            'models': {},
        }
        
        # ONN
        onn = OperationDAG(
            n_inputs=n_inputs,
            n_hidden_layers=2,
            nodes_per_layer=4,
            n_outputs=n_outputs,
        )
        result = self.train_model(onn, x_train, y_train, x_val, y_val, "ONN")
        benchmark_results['models']['ONN'] = result
        
        # MLP
        mlp = BaselineMLP(n_inputs=n_inputs, n_hidden=32, n_outputs=n_outputs)
        result = self.train_model(mlp, x_train, y_train, x_val, y_val, "MLP")
        benchmark_results['models']['MLP'] = result
        
        # LSTM (only for sequence data)
        if x.dim() == 3:
            lstm = BaselineLSTM(n_inputs=n_inputs, hidden_size=32, n_outputs=n_outputs)
            result = self.train_model(lstm, x_train, y_train, x_val, y_val, "LSTM")
            benchmark_results['models']['LSTM'] = result
        
        # CNN
        if n_inputs > 1:  # Makes sense for multi-feature
            cnn = BaselineCNN(n_inputs=n_inputs, n_outputs=n_outputs)
            result = self.train_model(cnn, x_train, y_train, x_val, y_val, "CNN")
            benchmark_results['models']['CNN'] = result
        
        self.results.append(benchmark_results)
        return benchmark_results
    
    def print_comparison(self):
        """Print comparison table of all results."""
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        for result in self.results:
            print(f"\nTask: {result['task']}")
            print("-" * 60)
            print(f"{'Model':<10} {'MSE':<12} {'Corr':<10} {'Params':<10} {'Time(s)':<10}")
            print("-" * 60)
            
            for model_name, stats in result['models'].items():
                print(f"{model_name:<10} {stats['final_mse']:<12.4f} "
                      f"{stats['correlation']:<10.4f} {stats['n_params']:<10} "
                      f"{stats['training_time']:<10.1f}")


# ============================================================================
# Quick Benchmark Functions
# ============================================================================

def run_all_benchmarks(epochs: int = 200, verbose: bool = True) -> List[Dict]:
    """
    Run all standard benchmarks.
    
    Returns list of results for each benchmark.
    """
    runner = BenchmarkRunner(epochs=epochs, verbose=verbose)
    
    # Polynomial benchmarks
    for formula in ['x^2', 'sin', 'sin+x^2']:
        x, y, name = generate_polynomial_data(formula=formula)
        runner.run_benchmark(x, y, name)
    
    # Multivariate
    x, y, name = generate_multivariate_data()
    runner.run_benchmark(x, y, name)
    
    # Time series
    x, y, name = generate_time_series_data()
    runner.run_benchmark(x, y, name)
    
    runner.print_comparison()
    
    return runner.results


def quick_comparison(
    x: torch.Tensor,
    y: torch.Tensor,
    task_name: str = "Custom Task",
    epochs: int = 200,
) -> Dict:
    """
    Quick comparison of ONN vs baselines on custom data.
    
    Usage:
        x = torch.randn(500, 2)
        y = (x[:, 0] ** 2 + x[:, 1]).unsqueeze(-1)
        results = quick_comparison(x, y, "Custom: x1² + x2")
    """
    runner = BenchmarkRunner(epochs=epochs)
    result = runner.run_benchmark(x, y, task_name)
    runner.print_comparison()
    return result
