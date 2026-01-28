"""
Post-Training Pruning Module for ONN

Implements multiple pruning strategies to clean up the network after training:
1. Recursive Graph Pruning - remove nodes that don't contribute to output
2. Mask and Fine-tune - mask weak connections, retrain remaining
3. Symbolic Consolidation - merge mathematically equivalent paths
4. Sensitivity Analysis - measure each node's importance via ablation

The goal is to get the simplest possible formula that still fits the data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS, Adam
from typing import Dict, List, Tuple, Optional, Set
import copy
import math
from collections import defaultdict


class PostTrainingPruner:
    """
    Comprehensive post-training pruning for ONN models.
    
    Usage:
        pruner = PostTrainingPruner(model, x_train, y_train)
        
        # Option 1: Full pipeline
        final_mse, formula = pruner.prune_full_pipeline(verbose=True)
        
        # Option 2: Individual strategies
        pruner.sensitivity_analysis()
        pruner.recursive_graph_prune()
        pruner.mask_and_finetune()
        pruner.symbolic_consolidation()
    """
    
    def __init__(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.x = x.to(self.device)
        self.y = y.to(self.device)
        self.y_sq = self.y.squeeze()
        
        # Cache for analysis results
        self.node_importance: Dict[str, float] = {}
        self.connection_importance: Dict[Tuple[str, str], float] = {}
        self.sensitivity_scores: Dict[str, float] = {}
        
    def get_mse(self) -> float:
        """Compute current MSE."""
        self.model.eval()
        with torch.no_grad():
            pred, _ = self.model(self.x, hard=True)
            return F.mse_loss(pred.squeeze(), self.y_sq).item()
    
    def get_formula(self) -> str:
        """Get current formula string."""
        if hasattr(self.model, 'get_formula'):
            return self.model.get_formula()
        return "?"
    
    # =========================================================================
    # 1. SENSITIVITY ANALYSIS (Ablation Testing)
    # =========================================================================
    
    def sensitivity_analysis(
        self,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Measure each node's importance by ablating it and measuring MSE increase.
        
        For each node:
        1. Zero out its output
        2. Measure new MSE
        3. Importance = (ablated_mse - baseline_mse) / baseline_mse
        
        Higher scores = more important nodes (removing them hurts more).
        
        Returns:
            Dict mapping node names to importance scores
        """
        if verbose:
            print("\n" + "="*60)
            print("SENSITIVITY ANALYSIS (ABLATION TESTING)")
            print("="*60)
        
        baseline_mse = self.get_mse()
        original_state = copy.deepcopy(self.model.state_dict())
        
        if verbose:
            print(f"Baseline MSE: {baseline_mse:.6f}")
            print("\nAblating each node...")
        
        self.sensitivity_scores = {}
        
        # Get all OperationNodes
        if hasattr(self.model, 'layers'):
            for layer_idx, layer in enumerate(self.model.layers):
                for node_idx, node in enumerate(layer.nodes):
                    node_name = f"L{layer_idx}_N{node_idx}"
                    
                    # Save node's output weights
                    if hasattr(node, 'output_proj'):
                        orig_weight = node.output_proj.weight.data.clone()
                        orig_bias = node.output_proj.bias.data.clone() if node.output_proj.bias is not None else None
                        
                        # Zero out the node
                        with torch.no_grad():
                            node.output_proj.weight.data.zero_()
                            if node.output_proj.bias is not None:
                                node.output_proj.bias.data.zero_()
                        
                        # Measure ablated MSE
                        ablated_mse = self.get_mse()
                        
                        # Restore
                        with torch.no_grad():
                            node.output_proj.weight.data.copy_(orig_weight)
                            if orig_bias is not None:
                                node.output_proj.bias.data.copy_(orig_bias)
                        
                        # Compute importance
                        if baseline_mse > 1e-10:
                            importance = (ablated_mse - baseline_mse) / baseline_mse
                        else:
                            importance = ablated_mse
                        
                        self.sensitivity_scores[node_name] = importance
                        
                        if verbose:
                            status = "CRITICAL" if importance > 1.0 else "important" if importance > 0.1 else "minor" if importance > 0.01 else "DEAD"
                            print(f"  {node_name}: Δ={importance:+.4f} ({status})")
        
        # Also test output projection channels
        if hasattr(self.model, 'output_proj'):
            weight = self.model.output_proj.weight.data
            n_channels = weight.shape[1]
            
            if verbose:
                print("\nAblating output channels...")
            
            for ch_idx in range(n_channels):
                ch_name = f"output_ch_{ch_idx}"
                
                orig_ch = weight[0, ch_idx].item()
                
                with torch.no_grad():
                    weight[0, ch_idx] = 0.0
                
                ablated_mse = self.get_mse()
                
                with torch.no_grad():
                    weight[0, ch_idx] = orig_ch
                
                if baseline_mse > 1e-10:
                    importance = (ablated_mse - baseline_mse) / baseline_mse
                else:
                    importance = ablated_mse
                
                self.sensitivity_scores[ch_name] = importance
                
                if verbose and abs(orig_ch) > 0.01:
                    status = "CRITICAL" if importance > 1.0 else "important" if importance > 0.1 else "minor" if importance > 0.01 else "dead"
                    print(f"  {ch_name} (w={orig_ch:.3f}): Δ={importance:+.4f} ({status})")
        
        # Verify state is restored
        self.model.load_state_dict(original_state)
        
        if verbose:
            # Summary
            critical = sum(1 for s in self.sensitivity_scores.values() if s > 1.0)
            important = sum(1 for s in self.sensitivity_scores.values() if 0.1 < s <= 1.0)
            minor = sum(1 for s in self.sensitivity_scores.values() if 0.01 < s <= 0.1)
            dead = sum(1 for s in self.sensitivity_scores.values() if s <= 0.01)
            print(f"\nSummary: {critical} critical, {important} important, {minor} minor, {dead} dead")
        
        
        return self.sensitivity_scores

    def gradient_sensitivity_analysis(self, verbose: bool = True) -> Dict[str, float]:
        """
        Estimate importance using gradients (Taylor expansion approximation).
        Importance ~ |grad * weight|
        Much faster than ablation (O(1) backward pass vs O(N) forward passes).
        """
        if verbose:
            print("\\n" + "="*60)
            print("GRADIENT SENSITIVITY ANALYSIS")
            print("="*60)
            
        self.model.train()
        self.model.zero_grad()
        
        try:
            pred, _ = self.model(self.x, hard=True)
            loss = F.mse_loss(pred.squeeze(), self.y_sq)
            loss.backward()
        except Exception as e:
            if verbose:
                print(f"Gradient computation failed: {e}")
            return {}
            
        self.sensitivity_scores = {}
        
        # Analyze nodes
        if hasattr(self.model, 'layers'):
            for layer_idx, layer in enumerate(self.model.layers):
                for node_idx, node in enumerate(layer.nodes):
                    if hasattr(node, 'output_proj'):
                        w = node.output_proj.weight
                        if w.grad is not None:
                            # Importance = mean(|w * g|)
                            importance = (w * w.grad).abs().mean().item()
                            self.sensitivity_scores[f"L{layer_idx}_N{node_idx}"] = importance
        
        # Analyze output channels
        if hasattr(self.model, 'output_proj'):
            w = self.model.output_proj.weight
            if w.grad is not None:
                for i in range(w.shape[1]):
                    importance = (w[0, i] * w.grad[0, i]).abs().item()
                    self.sensitivity_scores[f"output_ch_{i}"] = importance

        if verbose:
             print(f"Computed sensitivity for {len(self.sensitivity_scores)} components via gradients.")
             
        return self.sensitivity_scores
    
    # =========================================================================
    # 2. RECURSIVE GRAPH PRUNING
    # =========================================================================
    
    def recursive_graph_prune(
        self,
        importance_threshold: float = 0.01,
        prune_dead_nodes: bool = True,
        verbose: bool = True,
    ) -> int:
        """
        Recursively remove nodes that don't contribute to output.
        
        Algorithm:
        1. Start from output layer
        2. Find all nodes with significant contribution to output
        3. Trace back: which input nodes contribute to those nodes?
        4. Remove all nodes not in the "active path"
        
        Args:
            importance_threshold: Nodes with importance below this are pruned
            prune_dead_nodes: Whether to zero out dead nodes
            verbose: Print progress
            
        Returns:
            Number of nodes pruned
        """
        if verbose:
            print("\n" + "="*60)
            print("RECURSIVE GRAPH PRUNING")
            print("="*60)
        
        # Run sensitivity analysis if not done
        if not self.sensitivity_scores:
            self.sensitivity_analysis(verbose=False)
        
        # Find dead nodes (those whose removal doesn't hurt)
        dead_nodes = [name for name, score in self.sensitivity_scores.items() 
                      if score <= importance_threshold and name.startswith('L')]
        
        if verbose:
            print(f"Dead nodes (importance ≤ {importance_threshold}): {len(dead_nodes)}")
            for node in dead_nodes:
                print(f"  - {node}")
        
        nodes_pruned = 0
        
        if prune_dead_nodes and hasattr(self.model, 'layers'):
            for node_name in dead_nodes:
                # Parse layer and node index
                parts = node_name.split('_')
                layer_idx = int(parts[0][1:])  # L0 -> 0
                node_idx = int(parts[1][1:])   # N0 -> 0
                
                if layer_idx < len(self.model.layers):
                    layer = self.model.layers[layer_idx]
                    if node_idx < len(layer.nodes):
                        node = layer.nodes[node_idx]
                        
                        # Zero out the node's contribution
                        if hasattr(node, 'output_proj'):
                            with torch.no_grad():
                                node.output_proj.weight.data.zero_()
                                if node.output_proj.bias is not None:
                                    node.output_proj.bias.data.zero_()
                            nodes_pruned += 1
                            if verbose:
                                print(f"  Pruned: {node_name}")
        
        # Also prune dead output channels
        dead_channels = [name for name, score in self.sensitivity_scores.items()
                        if score <= importance_threshold and name.startswith('output_ch')]
        
        if hasattr(self.model, 'output_proj') and dead_channels:
            weight = self.model.output_proj.weight.data
            for ch_name in dead_channels:
                ch_idx = int(ch_name.split('_')[-1])
                if ch_idx < weight.shape[1]:
                    with torch.no_grad():
                        weight[0, ch_idx] = 0.0
                    if verbose:
                        print(f"  Pruned: {ch_name}")
        
        if verbose:
            new_mse = self.get_mse()
            print(f"\nPruned {nodes_pruned} nodes")
            print(f"MSE after pruning: {new_mse:.6f}")
        
        return nodes_pruned
    
    # =========================================================================
    # 3. MASK AND FINE-TUNE
    # =========================================================================
    
    def mask_and_finetune(
        self,
        weight_threshold: float = 0.1,
        finetune_steps: int = 300,
        verbose: bool = True,
    ) -> float:
        """
        Mask weak connections, then fine-tune remaining weights.
        
        Algorithm:
        1. Identify weights below threshold
        2. Create binary mask (0 for weak, 1 for strong)
        3. Zero out weak weights
        4. Fine-tune remaining weights with L-BFGS
        5. Re-apply mask after each step
        
        Args:
            weight_threshold: Absolute threshold for weak weights
            finetune_steps: Number of L-BFGS steps
            verbose: Print progress
            
        Returns:
            Final MSE after mask and fine-tune
        """
        if verbose:
            print("\n" + "="*60)
            print("MASK AND FINE-TUNE")
            print("="*60)
        
        initial_mse = self.get_mse()
        
        if verbose:
            print(f"Initial MSE: {initial_mse:.6f}")
        
        # Create masks for output_proj
        masks = {}
        total_weights = 0
        masked_weights = 0
        
        if hasattr(self.model, 'output_proj'):
            weight = self.model.output_proj.weight.data
            mask = (weight.abs() >= weight_threshold).float()
            masks['output_proj.weight'] = mask
            total_weights += weight.numel()
            masked_weights += (mask == 0).sum().item()
            
            # Apply mask
            with torch.no_grad():
                weight.mul_(mask)
        
        # Create masks for each node's output_proj
        if hasattr(self.model, 'layers'):
            for layer_idx, layer in enumerate(self.model.layers):
                for node_idx, node in enumerate(layer.nodes):
                    if hasattr(node, 'output_proj'):
                        weight = node.output_proj.weight.data
                        mask = (weight.abs() >= weight_threshold).float()
                        masks[f'layers.{layer_idx}.nodes.{node_idx}.output_proj.weight'] = mask
                        total_weights += weight.numel()
                        masked_weights += (mask == 0).sum().item()
                        
                        with torch.no_grad():
                            weight.mul_(mask)
        
        if verbose:
            sparsity = masked_weights / total_weights * 100 if total_weights > 0 else 0
            print(f"Masked {masked_weights}/{total_weights} weights ({sparsity:.1f}% sparsity)")
            post_mask_mse = self.get_mse()
            print(f"MSE after masking: {post_mask_mse:.6f}")
        
        # Fine-tune with mask
        output_params = [p for n, p in self.model.named_parameters() 
                        if 'output_proj' in n and p.requires_grad]
        
        if output_params and finetune_steps > 0:
            if verbose:
                print(f"Fine-tuning {len(output_params)} parameter tensors...")
            
            def apply_masks():
                with torch.no_grad():
                    if hasattr(self.model, 'output_proj') and 'output_proj.weight' in masks:
                        self.model.output_proj.weight.data.mul_(masks['output_proj.weight'])
                    
                    if hasattr(self.model, 'layers'):
                        for layer_idx, layer in enumerate(self.model.layers):
                            for node_idx, node in enumerate(layer.nodes):
                                key = f'layers.{layer_idx}.nodes.{node_idx}.output_proj.weight'
                                if hasattr(node, 'output_proj') and key in masks:
                                    node.output_proj.weight.data.mul_(masks[key])
            
            try:
                def closure():
                    for p in output_params:
                        if p.grad is not None:
                            p.grad.zero_()
                    pred, _ = self.model(self.x, hard=True)
                    loss = F.mse_loss(pred.squeeze(), self.y_sq)
                    if not torch.isnan(loss):
                        loss.backward()
                    return loss
                
                optimizer = LBFGS(output_params, lr=1.0, max_iter=finetune_steps, 
                                 line_search_fn='strong_wolfe')
                optimizer.step(closure)
                apply_masks()
                
            except Exception as e:
                if verbose:
                    print(f"  L-BFGS failed: {e}, trying Adam...")
                
                # Fallback to Adam
                optimizer = Adam(output_params, lr=0.01)
                for step in range(finetune_steps):
                    optimizer.zero_grad()
                    pred, _ = self.model(self.x, hard=True)
                    loss = F.mse_loss(pred.squeeze(), self.y_sq)
                    loss.backward()
                    optimizer.step()
                    apply_masks()
        
        final_mse = self.get_mse()
        
        if verbose:
            print(f"Final MSE: {final_mse:.6f}")
            print(f"Formula: {self.get_formula()}")
        
        return final_mse
    
    # =========================================================================
    # 4. SYMBOLIC CONSOLIDATION
    # =========================================================================
    
    def symbolic_consolidation(
        self,
        merge_threshold: float = 0.1,
        verbose: bool = True,
    ) -> int:
        """
        Merge mathematically equivalent operations.
        
        Identifies cases like:
        - Multiple x² nodes -> merge into single node with summed coefficient
        - x + x -> 2*x
        - sin(x) * 0.5 + sin(x) * 0.3 -> sin(x) * 0.8
        
        Args:
            merge_threshold: Merge nodes whose formulas match within this tolerance
            verbose: Print progress
            
        Returns:
            Number of merges performed
        """
        if verbose:
            print("\n" + "="*60)
            print("SYMBOLIC CONSOLIDATION")
            print("="*60)
        
        merges = 0
        
        if not hasattr(self.model, 'layers'):
            if verbose:
                print("No layers found, skipping consolidation")
            return 0
        
        # Collect node formulas
        node_formulas: Dict[str, List[Tuple[int, int, float]]] = defaultdict(list)
        
        for layer_idx, layer in enumerate(self.model.layers):
            for node_idx, node in enumerate(layer.nodes):
                if hasattr(node, 'get_formula'):
                    try:
                        formula = node.get_formula()
                        if formula and formula != "0":
                            # Get the node's contribution weight in output_proj
                            if hasattr(self.model, 'output_proj'):
                                # Calculate which output index this node contributes to
                                # This depends on the architecture
                                weight = 1.0  # Default
                                node_formulas[formula].append((layer_idx, node_idx, weight))
                    except Exception as e:
                        if verbose:
                            print(f"Error reading formula: {e}")
        
        if verbose:
            print(f"Found {len(node_formulas)} unique formula patterns:")
            for formula, nodes in node_formulas.items():
                if len(nodes) > 1:
                    print(f"  '{formula}' appears in {len(nodes)} nodes - MERGEABLE")
                else:
                    print(f"  '{formula}' appears in 1 node")
        
        # Merge duplicate formulas by combining their output weights
        for formula, nodes in node_formulas.items():
            if len(nodes) > 1:
                # Keep the first node, zero out the rest, combine weights
                primary_layer, primary_node, _ = nodes[0]
                
                # Sum up all weights from duplicate nodes
                # (In a more sophisticated implementation, we'd properly track
                # which output channels each node contributes to)
                
                for layer_idx, node_idx, _ in nodes[1:]:
                    if layer_idx < len(self.model.layers):
                        layer = self.model.layers[layer_idx]
                        if node_idx < len(layer.nodes):
                            node = layer.nodes[node_idx]
                            if hasattr(node, 'output_proj'):
                                # Add weights to primary node before zeroing out
                                primary_node_obj = self.model.layers[primary_layer].nodes[primary_node]
                                if hasattr(primary_node_obj, 'output_proj'):
                                    with torch.no_grad():
                                        # Add weights
                                        primary_node_obj.output_proj.weight.data.add_(node.output_proj.weight.data)
                                        # Add bias if both exist
                                        if primary_node_obj.output_proj.bias is not None and node.output_proj.bias is not None:
                                            primary_node_obj.output_proj.bias.data.add_(node.output_proj.bias.data)
                                
                                with torch.no_grad():
                                    # Zero out duplicate
                                    node.output_proj.weight.data.zero_()
                                    if node.output_proj.bias is not None:
                                        node.output_proj.bias.data.zero_()
                                merges += 1
                                if verbose:
                                    print(f"  Merged L{layer_idx}_N{node_idx} into L{primary_layer}_N{primary_node}")
        
        if verbose:
            print(f"\nPerformed {merges} merges")
            print(f"MSE after consolidation: {self.get_mse():.6f}")
        
        return merges
    
    # =========================================================================
    # 5. ITERATIVE BACKWARD PRUNING (NEW)
    # =========================================================================
    
    def iterative_backward_prune(
        self,
        mse_tolerance: float = 1.2,
        min_importance: float = 0.05,
        max_iterations: int = 20,
        verbose: bool = True,
    ) -> int:
        """
        Iteratively prune least important nodes until MSE degrades too much.
        
        Algorithm:
        1. Run sensitivity analysis
        2. Remove the least important node (if importance < threshold)
        3. Fine-tune remaining weights
        4. If MSE is still acceptable, repeat
        5. Stop when MSE exceeds tolerance or no more nodes to prune
        
        Args:
            mse_tolerance: Stop if MSE exceeds baseline * mse_tolerance
            min_importance: Only prune nodes with importance below this
            max_iterations: Maximum pruning iterations
            verbose: Print progress
            
        Returns:
            Total number of nodes pruned
        """
        if verbose:
            print("\n" + "="*60)
            print("ITERATIVE BACKWARD PRUNING")
            print("="*60)
        
        baseline_mse = self.get_mse()
        max_acceptable_mse = baseline_mse * mse_tolerance
        
        if verbose:
            print(f"Baseline MSE: {baseline_mse:.6f}")
            print(f"Max acceptable MSE: {max_acceptable_mse:.6f}")
        
        total_pruned = 0
        
        for iteration in range(max_iterations):
            # Gradient-based sensitivity (O(1)) instead of ablation (O(N))
            self.gradient_sensitivity_analysis(verbose=False)
            
            # Find least important node that's still above zero
            candidates = [(name, score) for name, score in self.sensitivity_scores.items()
                         if 0 < score < min_importance and name.startswith('L')]
            
            # Also consider output channels
            candidates += [(name, score) for name, score in self.sensitivity_scores.items()
                          if 0 < score < min_importance and name.startswith('output_ch')]
            
            if not candidates:
                if verbose:
                    print(f"\nIteration {iteration + 1}: No more prunable nodes")
                break
            
            # Sort by importance (lowest first)
            candidates.sort(key=lambda x: x[1])
            target_name, target_importance = candidates[0]
            
            if verbose:
                print(f"\nIteration {iteration + 1}: Pruning {target_name} (importance={target_importance:.4f})")
            
            # Prune the node
            if target_name.startswith('L'):
                parts = target_name.split('_')
                layer_idx = int(parts[0][1:])
                node_idx = int(parts[1][1:])
                
                if hasattr(self.model, 'layers') and layer_idx < len(self.model.layers):
                    layer = self.model.layers[layer_idx]
                    if node_idx < len(layer.nodes):
                        node = layer.nodes[node_idx]
                        if hasattr(node, 'output_proj'):
                            with torch.no_grad():
                                node.output_proj.weight.data.zero_()
                                if node.output_proj.bias is not None:
                                    node.output_proj.bias.data.zero_()
            
            elif target_name.startswith('output_ch'):
                ch_idx = int(target_name.split('_')[-1])
                if hasattr(self.model, 'output_proj'):
                    with torch.no_grad():
                        self.model.output_proj.weight.data[0, ch_idx] = 0.0
            
            total_pruned += 1
            
            # Fine-tune
            self.mask_and_finetune(weight_threshold=0.0, finetune_steps=100, verbose=False)
            
            current_mse = self.get_mse()
            
            if verbose:
                print(f"  MSE after prune+finetune: {current_mse:.6f}")
            
            # Check if MSE is still acceptable
            if current_mse > max_acceptable_mse:
                if verbose:
                    print(f"  MSE exceeded tolerance, stopping")
                break
        
        if verbose:
            print(f"\nTotal pruned: {total_pruned} nodes/channels")
            print(f"Final MSE: {self.get_mse():.6f}")
            print(f"Final formula: {self.get_formula()}")
        
        return total_pruned
    
    # =========================================================================
    # FULL PIPELINE
    # =========================================================================
    
    def prune_full_pipeline(
        self,
        mse_tolerance: float = 1.5,
        verbose: bool = True,
    ) -> Tuple[float, str]:
        """
        Run full pruning pipeline in optimal order.
        
        Pipeline:
        1. Sensitivity Analysis - identify importance of each component
        2. Recursive Graph Pruning - remove definitely dead nodes
        3. Symbolic Consolidation - merge equivalent operations  
        4. Iterative Backward Pruning - gradually remove less important nodes
        5. Mask and Fine-tune - final cleanup and optimization
        
        Args:
            mse_tolerance: Accept simplification if MSE < baseline * tolerance
            verbose: Print detailed progress
            
        Returns:
            (final_mse, final_formula)
        """
        if verbose:
            print("\n" + "="*60)
            print("FULL POST-TRAINING PRUNING PIPELINE")
            print("="*60)
        
        initial_mse = self.get_mse()
        initial_formula = self.get_formula()
        
        if verbose:
            print(f"\nInitial state:")
            print(f"  MSE: {initial_mse:.6f}")
            print(f"  Formula: {initial_formula}")
        
        # Save state in case we need to roll back
        best_state = copy.deepcopy(self.model.state_dict())
        best_mse = initial_mse
        max_acceptable_mse = initial_mse * mse_tolerance
        
        # 1. Sensitivity Analysis
        self.sensitivity_analysis(verbose=verbose)
        
        # 2. Recursive Graph Pruning (remove clearly dead nodes)
        self.recursive_graph_prune(importance_threshold=0.005, verbose=verbose)
        
        current_mse = self.get_mse()
        if current_mse <= max_acceptable_mse and current_mse > 0:
            best_mse = current_mse
            best_state = copy.deepcopy(self.model.state_dict())
        
        # 3. Symbolic Consolidation
        self.symbolic_consolidation(verbose=verbose)
        
        current_mse = self.get_mse()
        if current_mse <= max_acceptable_mse and current_mse > 0:
            best_mse = current_mse
            best_state = copy.deepcopy(self.model.state_dict())
        
        # 4. Iterative Backward Pruning
        self.iterative_backward_prune(
            mse_tolerance=mse_tolerance,
            min_importance=0.05,
            verbose=verbose
        )
        
        current_mse = self.get_mse()
        if current_mse <= max_acceptable_mse and current_mse > 0:
            best_mse = current_mse
            best_state = copy.deepcopy(self.model.state_dict())
        
        # 5. Final Mask and Fine-tune
        self.mask_and_finetune(weight_threshold=0.05, finetune_steps=500, verbose=verbose)
        
        current_mse = self.get_mse()
        if current_mse <= max_acceptable_mse and current_mse > 0:
            best_mse = current_mse
            best_state = copy.deepcopy(self.model.state_dict())
        else:
            # Roll back to best state
            self.model.load_state_dict(best_state)
        
        final_mse = self.get_mse()
        final_formula = self.get_formula()
        
        if verbose:
            print("\n" + "="*60)
            print("PRUNING COMPLETE")
            print("="*60)
            print(f"Initial MSE: {initial_mse:.6f} -> Final MSE: {final_mse:.6f}")
            print(f"Initial: {initial_formula}")
            print(f"Final:   {final_formula}")
            improvement = (initial_mse - final_mse) / initial_mse * 100 if initial_mse > 0 else 0
            print(f"MSE improvement: {improvement:+.1f}%")
        
        return final_mse, final_formula


# ============================================================================
# Convenience functions
# ============================================================================

def prune_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mse_tolerance: float = 1.5,
    verbose: bool = True,
) -> Tuple[float, str]:
    """
    Convenience function to run full pruning pipeline on a model.
    
    Args:
        model: Trained ONN model
        x, y: Training data
        mse_tolerance: Accept if MSE < baseline * tolerance
        verbose: Print progress
        
    Returns:
        (final_mse, final_formula)
    """
    pruner = PostTrainingPruner(model, x, y)
    return pruner.prune_full_pipeline(mse_tolerance=mse_tolerance, verbose=verbose)


def analyze_model_sensitivity(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Run just sensitivity analysis on a model.
    
    Args:
        model: Trained ONN model
        x, y: Training data
        verbose: Print detailed analysis
        
    Returns:
        Dict mapping component names to importance scores
    """
    pruner = PostTrainingPruner(model, x, y)
    return pruner.sensitivity_analysis(verbose=verbose)
