"""
GPU-Optimized OperationDAG wrapper.

Provides optional torch.compile optimization for the base DAG.
Uses standard PyTorch operations (batched kernels removed for portability).

Usage:
    from glassbox.sr.gpu_optimized_dag import GPUOptimizedDAG, wrap_dag_for_gpu
    
    base_dag = OperationDAG(...)
    gpu_dag = wrap_dag_for_gpu(base_dag)
    
    # Use like normal DAG
    output, info = gpu_dag(x)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


class GPUOptimizedDAG(nn.Module):
    """
    GPU-optimized wrapper for OperationDAG.
    
    Key optimizations:
    1. Optional torch.compile for graph optimization
    2. Memory-contiguous tensor layouts
    3. AMP-compatible forward pass
    
    Attributes:
        base_dag: The underlying OperationDAG
    """
    
    def __init__(
        self,
        base_dag: nn.Module,
        use_compile: bool = False,  # Disabled by default - ONN has dynamic control flow
    ):
        """
        Initialize GPU-optimized DAG wrapper.
        
        Args:
            base_dag: OperationDAG to wrap
            use_compile: Use torch.compile for automatic optimization
        """
        super().__init__()
        self.base_dag = base_dag
        self.use_compile = use_compile
        
        # Apply torch.compile if requested
        if use_compile:
            try:
                self._compiled_dag = torch.compile(base_dag, mode="reduce-overhead")
                self._compile_available = True
            except Exception:
                self._compiled_dag = None
                self._compile_available = False
        else:
            self._compiled_dag = None
            self._compile_available = False
    
    def invalidate_cache(self):
        """No-op for API compatibility."""
        pass
    
    def forward(
        self,
        x: torch.Tensor,
        hard: bool = True,
        return_all_outputs: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through the DAG.
        
        Args:
            x: Input tensor [batch, n_inputs]
            hard: Use hard selections
            return_all_outputs: Return all intermediate outputs
            
        Returns:
            Tuple of (output, info_dict)
        """
        # Ensure contiguous memory layout
        x = x.contiguous()
        
        # Use compiled version if available
        if self._compile_available and self._compiled_dag is not None:
            return self._compiled_dag(x, hard=hard, return_all_outputs=return_all_outputs)
        
        # Fall back to base DAG
        return self.base_dag(x, hard=hard, return_all_outputs=return_all_outputs)
    
    # Delegate attribute access to base_dag for compatibility
    def __getattr__(self, name):
        if name in ['base_dag', '_compiled_dag', '_compile_available', 'use_compile']:
            return super().__getattr__(name)
        return getattr(self.base_dag, name)


def wrap_dag_for_gpu(
    dag: nn.Module,
    use_compile: bool = False,
) -> GPUOptimizedDAG:
    """
    Wrap an OperationDAG for GPU optimization.
    
    Args:
        dag: OperationDAG to wrap
        use_compile: Use torch.compile (experimental)
        
    Returns:
        GPUOptimizedDAG wrapper
    """
    return GPUOptimizedDAG(dag, use_compile=use_compile)
