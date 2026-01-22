"""
Improved Training Module for ONN v2.

Key improvements:
1. Curriculum learning - start simple, add complexity
2. Better initialization - favor identity/power operations
3. Data normalization - scale inputs/outputs
4. Warmup + anneal schedule
5. Formula extraction during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Optional, Dict, List, Tuple, Callable
import math
import copy


def normalize_data(
    x: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Normalize input and output data for stable training.
    
    Returns normalized tensors and stats for denormalization.
    """
    # Input normalization
    x_mean = x.mean(dim=0, keepdim=True)
    x_std = x.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_norm = (x - x_mean) / x_std
    
    # Output normalization  
    y_mean = y.mean()
    y_std = y.std().clamp(min=1e-6)
    y_norm = (y - y_mean) / y_std
    
    stats = {
        'x_mean': x_mean,
        'x_std': x_std,
        'y_mean': y_mean,
        'y_std': y_std,
    }
    
    return x_norm, y_norm, stats


def denormalize_output(y_norm: torch.Tensor, stats: Dict) -> torch.Tensor:
    """Denormalize predictions back to original scale."""
    return y_norm * stats['y_std'] + stats['y_mean']


def initialize_for_identity(model: nn.Module):
    """
    Initialize model parameters to favor identity-like operations.
    
    This helps with:
    - Stable initial predictions
    - Faster convergence on simple functions
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'logit' in name:
                # Start with uniform selection
                param.zero_()
            elif 'p' in name and 'output' not in name:
                # Power parameter: start at p=1 (identity)
                param.fill_(1.0)
            elif 'omega' in name:
                # Frequency: start at 1
                param.fill_(1.0)
            elif 'phi' in name:
                # Phase: start at 0
                param.zero_()
            elif 'beta' in name and 'selector' not in name:
                # Arithmetic: start at 1 (addition)
                param.fill_(1.0)
            elif 'output_scale' in name:
                # Output scale: start at 1
                param.fill_(1.0)


def curriculum_schedule(
    epoch: int,
    total_epochs: int,
    phase_ratios: Tuple[float, float, float] = (0.2, 0.5, 0.3),
) -> Dict:
    """
    Curriculum learning schedule.
    
    Phase 1 (20%): Soft selection, low temperature, explore
    Phase 2 (50%): Transition, anneal temperature
    Phase 3 (30%): Hard selection, fine-tune
    
    Returns dict with training parameters for current epoch.
    """
    progress = epoch / max(total_epochs, 1)
    
    phase1_end = phase_ratios[0]
    phase2_end = phase_ratios[0] + phase_ratios[1]
    
    if progress < phase1_end:
        # Phase 1: Exploration
        phase_progress = progress / phase1_end
        return {
            'phase': 1,
            'hard': False,
            'tau': 2.0 - 0.5 * phase_progress,  # 2.0 -> 1.5
            'lr_scale': 1.0,
            'lambda_entropy': 0.1,
            'lambda_l0': 0.0,
        }
    elif progress < phase2_end:
        # Phase 2: Transition
        phase_progress = (progress - phase1_end) / phase_ratios[1]
        return {
            'phase': 2,
            'hard': False,
            'tau': 1.5 - 1.0 * phase_progress,  # 1.5 -> 0.5
            'lr_scale': 0.8,
            'lambda_entropy': 0.05,
            'lambda_l0': 0.01 * phase_progress,
        }
    else:
        # Phase 3: Exploitation
        phase_progress = (progress - phase2_end) / phase_ratios[2]
        return {
            'phase': 3,
            'hard': True,
            'tau': 0.5 - 0.3 * phase_progress,  # 0.5 -> 0.2
            'lr_scale': 0.5,
            'lambda_entropy': 0.01,
            'lambda_l0': 0.05,
        }


class ImprovedONNTrainer:
    """
    Improved trainer with curriculum learning and better convergence.
    """
    
    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.01,
        normalize: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.base_lr = lr
        self.normalize = normalize
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize for stable training
        initialize_for_identity(self.model)
        
        # Optimizer
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Training state
        self.history = []
        self.best_loss = float('inf')
        self.best_model_state = None
        self.norm_stats = None
    
    def train(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 500,
        print_every: int = 50,
        patience: int = 100,
    ) -> Dict:
        """
        Train with curriculum learning.
        
        Args:
            x: Input data
            y: Target data
            epochs: Total training epochs
            print_every: Print frequency
            patience: Early stopping patience
            
        Returns:
            Training history and results
        """
        # Move to device
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Normalize data
        if self.normalize:
            x_norm, y_norm, self.norm_stats = normalize_data(x, y)
        else:
            x_norm, y_norm = x, y
            self.norm_stats = {'y_mean': 0, 'y_std': 1}
        
        no_improve_count = 0
        
        print(f"Training on {self.device}")
        print(f"Data: {x.shape[0]} samples, {x.shape[1]} inputs")
        print("-" * 50)
        
        for epoch in range(epochs):
            # Get curriculum schedule
            schedule = curriculum_schedule(epoch, epochs)
            
            # Adjust learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr * schedule['lr_scale']
            
            # Update model temperature if it has one
            if hasattr(self.model, 'tau'):
                self.model.tau = schedule['tau']
            
            # Training step
            self.model.train()
            self.optimizer.zero_grad()
            
            pred, info = self.model(x_norm, hard=schedule['hard'])
            
            # Compute loss with regularization
            mse_loss = self.loss_fn(pred, y_norm)
            
            # Entropy regularization
            entropy_loss = 0
            if hasattr(self.model, 'entropy_regularization'):
                entropy_loss = self.model.entropy_regularization() * schedule['lambda_entropy']
            
            # L0 regularization (sparsity)
            l0_loss = 0
            if hasattr(self.model, 'l0_regularization'):
                l0_loss = self.model.l0_regularization() * schedule['lambda_l0']
            
            total_loss = mse_loss + entropy_loss + l0_loss
            
            # Check for NaN
            if torch.isnan(total_loss):
                print(f"Warning: NaN loss at epoch {epoch}")
                continue
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track history
            self.history.append({
                'epoch': epoch,
                'phase': schedule['phase'],
                'mse': mse_loss.item(),
                'total': total_loss.item(),
                'tau': schedule['tau'],
            })
            
            # Track best
            if mse_loss.item() < self.best_loss:
                self.best_loss = mse_loss.item()
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Print progress
            if epoch % print_every == 0 or epoch == epochs - 1:
                phase_str = f"Phase {schedule['phase']}"
                print(f"Epoch {epoch:4d} [{phase_str}] | "
                      f"MSE: {mse_loss.item():.4f} | "
                      f"τ={schedule['tau']:.2f} | "
                      f"Best: {self.best_loss:.4f}")
            
            # Early stopping
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            pred, _ = self.model(x_norm, hard=True)
            final_mse = self.loss_fn(pred, y_norm).item()
            
            # Correlation (on original scale)
            pred_denorm = denormalize_output(pred, self.norm_stats)
            corr = torch.corrcoef(torch.stack([
                pred_denorm.squeeze().cpu(),
                y.squeeze().cpu()
            ]))[0, 1].item()
        
        print("-" * 50)
        print(f"Final MSE (normalized): {final_mse:.4f}")
        print(f"Best MSE: {self.best_loss:.4f}")
        print(f"Correlation: {corr:.4f}")
        
        # Get formula
        formula = self.get_formula()
        print(f"Discovered: {formula}")
        
        return {
            'history': self.history,
            'final_mse': final_mse,
            'best_mse': self.best_loss,
            'correlation': corr,
            'formula': formula,
        }
    
    def get_formula(self) -> str:
        """Extract the learned formula from the model."""
        if hasattr(self.model, 'snap_to_discrete'):
            self.model.snap_to_discrete()
        
        if hasattr(self.model, 'get_formula'):
            formula = self.model.get_formula()
        elif hasattr(self.model, 'get_graph_summary'):
            formula = self.model.get_graph_summary()
        else:
            return "Formula extraction not supported"

        # If normalization was used, denormalize the formula
        if self.normalize and self.norm_stats and isinstance(formula, str):
            import re
            x_mean = self.norm_stats['x_mean'].view(-1).tolist()
            x_std = self.norm_stats['x_std'].view(-1).tolist()
            y_mean = float(self.norm_stats['y_mean'])
            y_std = float(self.norm_stats['y_std'])

            def fmt(val: float) -> str:
                return f"{val:.6g}"

            # Replace x{i} with normalized form using token boundaries
            for i, (mean_i, std_i) in enumerate(zip(x_mean, x_std)):
                token = rf"\bx{i}\b"
                repl = f"((x{i} - {fmt(mean_i)})/{fmt(std_i)})"
                formula = re.sub(token, repl, formula)

            # Denormalize output: y = y_std * f(x_norm) + y_mean
            formula = f"({fmt(y_std)})*({formula}) + {fmt(y_mean)}"

        return formula
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions on new data."""
        x = x.to(self.device)
        
        if self.normalize and self.norm_stats:
            x_norm = (x - self.norm_stats['x_mean'].to(self.device)) / self.norm_stats['x_std'].to(self.device)
        else:
            x_norm = x
        
        self.model.eval()
        with torch.no_grad():
            pred, _ = self.model(x_norm, hard=True)
            
            if self.normalize and self.norm_stats:
                pred = denormalize_output(pred, {
                    'y_mean': self.norm_stats['y_mean'].to(self.device),
                    'y_std': self.norm_stats['y_std'].to(self.device),
                })
        
        return pred


def train_onn_improved(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 500,
    lr: float = 0.01,
    print_every: int = 50,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Convenience function for improved ONN training.
    
    Usage:
        from glassbox.sr import OperationDAG
        from glassbox.sr.training import train_onn_improved
        
        model = OperationDAG(n_inputs=1, n_hidden_layers=2, nodes_per_layer=4)
        results = train_onn_improved(model, x, y, epochs=500)
        print(results['formula'])
    """
    trainer = ImprovedONNTrainer(model, lr=lr, device=device)
    return trainer.train(x, y, epochs=epochs, print_every=print_every)
