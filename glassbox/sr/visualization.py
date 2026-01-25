"""
Real-time Visualization for Operation-Based Neural Networks.

Provides live visualization of:
1. ONN architecture (operations as nodes, connections as edges)
2. Training progress (fitness/MSE over generations)
3. Top model's structure and coefficients
4. Formula evolution
5. Input/output relationship

Uses matplotlib with animation for real-time updates.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
from typing import Optional, Dict, List, Tuple, Any
import threading
import queue
import time
import math


# Color scheme (inspired by the "Less Dumb Model" green-on-black aesthetic)
COLORS = {
    'background': '#0a0a0a',
    'primary': '#00ff00',      # Bright green - active
    'secondary': '#00cc00',    # Darker green
    'accent': '#00ffaa',       # Cyan-green
    'highlight': '#ffff00',    # Yellow for important
    'dim': '#004400',          # Dim green
    'text': '#00ff00',
    'text_dim': '#008800',
    'error': '#ff4444',        # Red - dead/negative
    'warning': '#ffaa00',
    'grid': '#002200',
    'inactive': '#4444ff',     # Blue - low importance
    'dead': '#ff2222',         # Red - dead neurons/connections
}

# Operation colors for different op types
OP_COLORS = {
    'power': '#00ff00',      # Green
    'periodic': '#00ffff',    # Cyan
    'arithmetic': '#ffff00',  # Yellow
    'exp': '#ff00ff',        # Magenta
    'log': '#ff8800',        # Orange
    'aggregation': '#8888ff', # Light blue
    'identity': '#444444',    # Gray
}


class ONNVisualizer:
    """
    Real-time visualizer for ONN evolutionary training.
    
    Creates a dashboard with:
    - Network architecture view (left)
    - Training curves (top right)
    - Current best formula (middle right)
    - Input/output scatter (bottom right)
    """
    
    def __init__(
        self,
        n_inputs: int = 1,
        n_layers: int = 2,
        nodes_per_layer: int = 4,
        update_interval: int = 100,  # ms between updates
        figsize: Tuple[int, int] = (16, 9),
        dark_mode: bool = True,
        lite_mode: bool = False,  # Skip network diagram for speed
    ):
        """
        Initialize the visualizer.
        
        Args:
            n_inputs: Number of input features
            n_layers: Number of hidden layers
            nodes_per_layer: Nodes per hidden layer
            update_interval: Milliseconds between frame updates
            figsize: Figure size (width, height)
            dark_mode: Use dark theme (green on black)
            lite_mode: If True, skip network diagram (faster updates)
        """
        self.n_inputs = n_inputs
        self.n_layers = n_layers
        self.nodes_per_layer = nodes_per_layer
        self.update_interval = update_interval
        self.dark_mode = dark_mode
        self.lite_mode = lite_mode
        
        # Data storage
        self.history = []
        self.current_gen = 0
        self.current_model = None
        self.current_formula = ""
        self.x_data = None
        self.y_data = None
        self.y_pred = None
        self.best_fitness = float('inf')
        self.correlation = 0.0
        
        # Threading for non-blocking updates
        self.update_queue = queue.Queue()
        self.running = False
        
        # Create figure
        self._setup_figure(figsize)
    
    def _setup_figure(self, figsize):
        """Setup the matplotlib figure and axes."""
        plt.style.use('dark_background' if self.dark_mode else 'default')
        
        self.fig = plt.figure(figsize=figsize, facecolor=COLORS['background'])
        self.fig.suptitle('ONN Evolution Visualizer', 
                         fontsize=20, color=COLORS['primary'], fontweight='bold')
        
        if self.lite_mode:
            # Lite mode: 3 panels in a row (no network diagram)
            gs = self.fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1],
                                       hspace=0.3, wspace=0.25,
                                       left=0.05, right=0.95, top=0.85, bottom=0.1)
            
            self.ax_network = None  # No network in lite mode
            
            # Training curves (left)
            self.ax_fitness = self.fig.add_subplot(gs[0, 0])
            self.ax_fitness.set_facecolor(COLORS['background'])
            self.ax_fitness.set_title('Training Progress', color=COLORS['primary'], fontsize=12)
            
            # Formula display (center)
            self.ax_formula = self.fig.add_subplot(gs[0, 1])
            self.ax_formula.set_facecolor(COLORS['background'])
            self.ax_formula.set_title('Best Formula', color=COLORS['primary'], fontsize=12)
            self.ax_formula.axis('off')
            
            # Fit visualization (right)
            self.ax_fit = self.fig.add_subplot(gs[0, 2])
            self.ax_fit.set_facecolor(COLORS['background'])
            self.ax_fit.set_title('Prediction vs Target', color=COLORS['primary'], fontsize=12)
        else:
            # Full mode: Network on left, 3 panels stacked on right
            gs = self.fig.add_gridspec(3, 2, width_ratios=[1.5, 1], height_ratios=[1, 1, 1],
                                       hspace=0.3, wspace=0.2,
                                       left=0.05, right=0.95, top=0.9, bottom=0.05)
            
            # Network architecture (spans all rows on left)
            self.ax_network = self.fig.add_subplot(gs[:, 0])
            self.ax_network.set_facecolor(COLORS['background'])
            self.ax_network.set_title('Network Architecture', color=COLORS['primary'], fontsize=14)
            
            # Training curves (top right)
            self.ax_fitness = self.fig.add_subplot(gs[0, 1])
            self.ax_fitness.set_facecolor(COLORS['background'])
            self.ax_fitness.set_title('Training Progress', color=COLORS['primary'], fontsize=12)
            
            # Formula display (middle right)
            self.ax_formula = self.fig.add_subplot(gs[1, 1])
            self.ax_formula.set_facecolor(COLORS['background'])
            self.ax_formula.set_title('Best Formula', color=COLORS['primary'], fontsize=12)
            self.ax_formula.axis('off')
            
            # Fit visualization (bottom right)
            self.ax_fit = self.fig.add_subplot(gs[2, 1])
            self.ax_fit.set_facecolor(COLORS['background'])
            self.ax_fit.set_title('Prediction vs Target', color=COLORS['primary'], fontsize=12)
        
        # Style all axes
        axes_to_style = [self.ax_fitness, self.ax_fit]
        if self.ax_network is not None:
            axes_to_style.append(self.ax_network)
        for ax in axes_to_style:
            ax.tick_params(colors=COLORS['text_dim'])
            for spine in ax.spines.values():
                spine.set_color(COLORS['dim'])
    
    def _draw_network(self):
        """Draw the neural network architecture."""
        if self.ax_network is None:
            return  # Lite mode - no network diagram
        
        ax = self.ax_network
        ax.clear()
        ax.set_facecolor(COLORS['background'])
        ax.set_title('Network Architecture', color=COLORS['primary'], fontsize=14)
        ax.axis('off')
        
        # Calculate layout
        total_layers = 1 + self.n_layers + 1  # input + hidden + output
        layer_x = np.linspace(0.1, 0.9, total_layers)
        
        # Get connection weights to determine node importance
        weights = self._get_connection_weights()
        
        # Calculate node importance (how much each node contributes to output)
        node_importance = {}
        for (from_key, to_key), weight in weights.items():
            if to_key == ('output', 0):
                node_importance[from_key] = abs(weight)
        
        max_importance = max(node_importance.values()) if node_importance else 1.0
        
        # Node positions
        node_positions = {}
        node_labels = {}
        node_colors = {}
        node_sizes = {}  # Variable node sizes
        node_border_colors = {}  # Border color separate from fill
        
        # Thresholds for importance classification
        HIGH_IMPORTANCE = 0.3    # Above this = active (green)
        LOW_IMPORTANCE = 0.05    # Below this = dead (red), between = low importance (blue)
        
        # Input layer
        input_y = np.linspace(0.2, 0.8, self.n_inputs) if self.n_inputs > 1 else [0.5]
        for i, y in enumerate(input_y):
            key = ('input', i)
            node_positions[key] = (layer_x[0], y)
            node_labels[key] = f'x{i}'
            node_colors[key] = COLORS['accent']  # Inputs always cyan-green
            node_border_colors[key] = COLORS['accent']
            node_sizes[key] = 0.03
        
        # Hidden layers - color based on importance
        for layer_idx in range(self.n_layers):
            hidden_y = np.linspace(0.1, 0.9, self.nodes_per_layer)
            for node_idx, y in enumerate(hidden_y):
                key = ('hidden', layer_idx, node_idx)
                node_positions[key] = (layer_x[1 + layer_idx], y)
                
                # Get operation info if model available
                op_name = self._get_node_operation(layer_idx, node_idx)
                node_labels[key] = op_name
                
                # Calculate importance ratio
                imp = node_importance.get(key, 0.0) / max_importance if max_importance > 0 else 0.0
                
                # Color based on importance:
                # - High importance (>0.3): GREEN (active, contributes to output)
                # - Low importance (0.05-0.3): BLUE (exists but weak contribution)
                # - Dead (<0.05): RED (no contribution)
                if imp >= HIGH_IMPORTANCE:
                    node_colors[key] = COLORS['primary']  # Green - active
                    node_border_colors[key] = COLORS['primary']
                elif imp >= LOW_IMPORTANCE:
                    node_colors[key] = COLORS['inactive']  # Blue - low importance
                    node_border_colors[key] = COLORS['inactive']
                else:
                    node_colors[key] = COLORS['dead']  # Red - dead
                    node_border_colors[key] = COLORS['dead']
                
                node_sizes[key] = 0.02 + imp * 0.02
        
        # Output layer
        node_positions[('output', 0)] = (layer_x[-1], 0.5)
        node_labels[('output', 0)] = 'y'
        node_colors[('output', 0)] = COLORS['highlight']  # Yellow - output
        node_border_colors[('output', 0)] = COLORS['highlight']
        node_sizes[('output', 0)] = 0.035
        
        # Draw connections (edges) - do this first so nodes are on top
        self._draw_connections(ax, node_positions, layer_x)
        
        # Draw nodes with variable size based on importance
        for key, (x, y) in node_positions.items():
            color = node_colors.get(key, COLORS['primary'])
            size = node_sizes.get(key, 0.03)
            
            # Importance affects border thickness
            imp = node_importance.get(key, 0.1) / max_importance if max_importance > 0 else 0.1
            linewidth = 1.5 + imp * 2.5  # 1.5 to 4.0
            
            # Draw node circle
            circle = Circle((x, y), size, facecolor=COLORS['background'],
                          edgecolor=color, linewidth=linewidth, zorder=3)
            ax.add_patch(circle)
            
            # Add label
            label = node_labels.get(key, '')
            fontsize = 7 if len(label) > 4 else 9
            ax.text(x, y, label, ha='center', va='center',
                   color=color, fontsize=fontsize, fontweight='bold', zorder=4)
        
        # Add layer labels
        ax.text(layer_x[0], 0.02, 'INPUT', ha='center', color=COLORS['text_dim'], fontsize=10)
        for i in range(self.n_layers):
            ax.text(layer_x[1 + i], 0.02, f'LAYER {i+1}', ha='center', 
                   color=COLORS['text_dim'], fontsize=10)
        ax.text(layer_x[-1], 0.02, 'OUTPUT', ha='center', color=COLORS['text_dim'], fontsize=10)
        
        # Add legend for color scheme
        legend_y = 0.98
        legend_x = 0.02
        ax.text(legend_x, legend_y, 'LEGEND:', color=COLORS['text'], fontsize=8, fontweight='bold',
               transform=ax.transAxes, ha='left', va='top')
        ax.text(legend_x, legend_y - 0.04, '● Active (high imp)', color=COLORS['primary'], fontsize=7,
               transform=ax.transAxes, ha='left', va='top')
        ax.text(legend_x, legend_y - 0.07, '● Low importance', color=COLORS['inactive'], fontsize=7,
               transform=ax.transAxes, ha='left', va='top')
        ax.text(legend_x, legend_y - 0.10, '● Dead (pruned)', color=COLORS['dead'], fontsize=7,
               transform=ax.transAxes, ha='left', va='top')
        ax.text(legend_x, legend_y - 0.13, '━ Strong connection', color=COLORS['primary'], fontsize=7,
               transform=ax.transAxes, ha='left', va='top')
        ax.text(legend_x, legend_y - 0.16, '─ Weak connection', color=COLORS['dead'], fontsize=7,
               transform=ax.transAxes, ha='left', va='top')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    def _draw_connections(self, ax, node_positions, layer_x):
        """
        Draw connections with clear visual hierarchy:
        
        - THICK GREEN: Best formula path (high weight connections)
        - THIN GREEN: Active but less important connections  
        - THIN RED: Dead connections (near-zero weight)
        
        Also shows edge constants for important connections.
        """
        from matplotlib.collections import LineCollection
        
        # Get connection weights if model available
        weights = self._get_connection_weights()
        
        # Collect connections with their weights
        connections = []  # [(line, weight, is_to_output), ...]
        
        # Build all potential connections
        for layer_idx in range(self.n_layers + 1):
            if layer_idx == 0:
                # Input to first hidden layer
                from_keys = [('input', i) for i in range(self.n_inputs)]
                to_keys = [('hidden', 0, i) for i in range(self.nodes_per_layer)]
                is_to_output = False
            elif layer_idx < self.n_layers:
                # Hidden to hidden
                from_keys = [('hidden', layer_idx - 1, i) for i in range(self.nodes_per_layer)]
                to_keys = [('hidden', layer_idx, i) for i in range(self.nodes_per_layer)]
                is_to_output = False
            else:
                # Last hidden to output
                from_keys = [('hidden', self.n_layers - 1, i) for i in range(self.nodes_per_layer)]
                # Also add input directly to output
                from_keys += [('input', i) for i in range(self.n_inputs)]
                to_keys = [('output', 0)]
                is_to_output = True
            
            for from_key in from_keys:
                for to_key in to_keys:
                    if from_key in node_positions and to_key in node_positions:
                        x1, y1 = node_positions[from_key]
                        x2, y2 = node_positions[to_key]
                        weight = weights.get((from_key, to_key), 0.0)
                        connections.append(((x1, y1, x2, y2), weight, is_to_output, from_key, to_key))
        
        # Find max weight for normalization
        all_weights = [abs(w) for _, w, _, _, _ in connections if w != 0]
        max_weight = max(all_weights) if all_weights else 1.0
        
        # Thresholds for connection classification
        HIGH_THRESHOLD = max_weight * 0.3   # Top connections = thick green
        LOW_THRESHOLD = 0.05                 # Below this = dead (thin red)
        
        # Store labels to draw last (on top)
        edge_labels = []
        
        for (x1, y1, x2, y2), weight, is_to_output, from_key, to_key in connections:
            abs_weight = abs(weight)
            normalized = abs_weight / max_weight if max_weight > 0 else 0
            
            # Determine connection style based on importance
            if abs_weight < LOW_THRESHOLD:
                # DEAD CONNECTION: thin red line
                color = COLORS['dead']
                linewidth = 0.5
                alpha = 0.3
                show_label = False
            elif abs_weight >= HIGH_THRESHOLD:
                # BEST FORMULA PATH: thick green line
                color = COLORS['primary']
                linewidth = 2.0 + normalized * 3.0  # 2.0 to 5.0
                alpha = 0.9
                show_label = is_to_output  # Show weight label for output connections
            else:
                # LESS IMPORTANT: thin green line
                color = COLORS['secondary']
                linewidth = 0.8 + normalized * 1.2  # 0.8 to 2.0
                alpha = 0.5
                show_label = False
            
            # Draw the connection
            ax.plot([x1, x2], [y1, y2], color=color, 
                   alpha=alpha, linewidth=linewidth, zorder=1,
                   solid_capstyle='round')
            
            # Store label for important connections to output
            if show_label and abs_weight >= HIGH_THRESHOLD:
                # Position label at midpoint of connection
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                # Offset slightly to avoid overlapping the line
                offset_y = 0.02 if y1 < y2 else -0.02
                label_text = f'{weight:.2f}'
                edge_labels.append((mid_x, mid_y + offset_y, label_text, color))
        
        # Draw edge labels on top
        for x, y, text, color in edge_labels:
            ax.text(x, y, text, ha='center', va='center',
                   color=color, fontsize=7, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.1', facecolor=COLORS['background'], 
                            edgecolor=color, alpha=0.8),
                   zorder=5)
    
    def _get_node_operation(self, layer_idx: int, node_idx: int) -> str:
        """Get the operation name for a specific node."""
        if self.current_model is None:
            return '?'
        
        try:
            if hasattr(self.current_model, 'layers'):
                layer = self.current_model.layers[layer_idx]
                if hasattr(layer, 'nodes'):
                    node = layer.nodes[node_idx]
                    if hasattr(node, 'get_selected_operation'):
                        op_str = node.get_selected_operation()
                        # Simplify the operation string
                        if 'power:' in op_str.lower():
                            if 'square' in op_str.lower():
                                return 'x²'
                            elif 'identity' in op_str.lower():
                                return 'x'
                            else:
                                return 'x^p'
                        elif 'periodic:' in op_str.lower():
                            if 'sin' in op_str.lower():
                                return 'sin'
                            elif 'cos' in op_str.lower():
                                return 'cos'
                        elif 'arithmetic:' in op_str.lower():
                            if 'add' in op_str.lower():
                                return '+'
                            elif 'mul' in op_str.lower():
                                return '×'
                        elif 'exp' in op_str.lower():
                            return 'exp'
                        elif 'log' in op_str.lower():
                            return 'log'
                        return op_str.split(':')[0][:4]
            return '?'
        except Exception:
            return '?'
    
    def _get_op_color(self, op_name: str) -> str:
        """Get color for an operation."""
        op_lower = op_name.lower()
        if 'x²' in op_name or 'x^' in op_name or 'pow' in op_lower:
            return OP_COLORS['power']
        elif 'sin' in op_lower or 'cos' in op_lower:
            return OP_COLORS['periodic']
        elif '+' in op_name or '×' in op_name or 'add' in op_lower or 'mul' in op_lower:
            return OP_COLORS['arithmetic']
        elif 'exp' in op_lower:
            return OP_COLORS['exp']
        elif 'log' in op_lower:
            return OP_COLORS['log']
        elif 'x' == op_name.lower() or 'identity' in op_lower:
            return OP_COLORS['identity']
        return COLORS['primary']
    
    def _get_connection_weights(self) -> Dict:
        """
        Get ALL connection weights from the model including internal routing.
        
        Returns weights for:
        - Input -> Hidden layer connections (routing weights)
        - Hidden -> Hidden layer connections (routing weights)  
        - Hidden -> Output connections (output_proj weights)
        """
        weights = {}
        if self.current_model is None:
            return weights
        
        try:
            # Get output projection weights (these are the final layer weights)
            if hasattr(self.current_model, 'output_proj'):
                output_weights = self.current_model.output_proj.weight.data[0].cpu().numpy()
                
                # Normalize for visualization
                max_weight = max(abs(output_weights.max()), abs(output_weights.min()), 1e-6)
                
                idx = 0
                # Input -> Output connections
                for i in range(self.n_inputs):
                    w = float(output_weights[idx]) / max_weight if idx < len(output_weights) else 0
                    weights[(('input', i), ('output', 0))] = w
                    idx += 1
                    
                # Hidden -> Output connections
                for layer_idx in range(self.n_layers):
                    for node_idx in range(self.nodes_per_layer):
                        key = (('hidden', layer_idx, node_idx), ('output', 0))
                        w = float(output_weights[idx]) / max_weight if idx < len(output_weights) else 0
                        weights[key] = w
                        idx += 1
            
            # Get internal routing weights from each layer
            if hasattr(self.current_model, 'layers'):
                for layer_idx, layer in enumerate(self.current_model.layers):
                    if hasattr(layer, 'nodes'):
                        for node_idx, node in enumerate(layer.nodes):
                            # Get routing info (which inputs this node uses)
                            if hasattr(node, 'get_routing_info'):
                                routing_info = node.get_routing_info()
                                primary_sources = routing_info.get('primary_sources', [])
                                routing_weights = routing_info.get('weights', [1.0, 1.0])
                                
                                # Map source indices to node keys
                                for src_slot, src_idx in enumerate(primary_sources[:2]):
                                    if src_idx < self.n_inputs:
                                        from_key = ('input', src_idx)
                                    else:
                                        # Calculate which hidden layer/node
                                        adj_idx = src_idx - self.n_inputs
                                        src_layer = adj_idx // self.nodes_per_layer
                                        src_node = adj_idx % self.nodes_per_layer
                                        if src_layer < layer_idx:  # Only from previous layers
                                            from_key = ('hidden', src_layer, src_node)
                                        else:
                                            continue
                                    
                                    to_key = ('hidden', layer_idx, node_idx)
                                    
                                    # Get weight for this routing connection
                                    w = routing_weights[src_slot] if src_slot < len(routing_weights) else 0.5
                                    weights[(from_key, to_key)] = w
                            
                            # Also get edge/scale weights if available
                            if hasattr(node, 'router') and hasattr(node.router, 'scales'):
                                scales = node.router.scales.data.cpu().numpy()
                                # These scale the input signals
        except Exception as e:
            pass  # Silently handle errors
        
        return weights
    
    def _draw_fitness_curve(self):
        """Draw the fitness/MSE over generations."""
        ax = self.ax_fitness
        ax.clear()
        ax.set_facecolor(COLORS['background'])
        ax.set_title('Training Progress', color=COLORS['primary'], fontsize=12)
        
        if not self.history:
            ax.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center',
                   color=COLORS['text_dim'], fontsize=12, transform=ax.transAxes)
            return
        
        generations = [h['generation'] for h in self.history]
        best_fits = [h.get('best_fitness', h.get('fitness', 0)) for h in self.history]
        
        # Plot best fitness
        ax.semilogy(generations, best_fits, color=COLORS['primary'], 
                   linewidth=2, label='Best MSE')
        
        # Plot mean fitness if available
        if 'mean_fitness' in self.history[0]:
            mean_fits = [h['mean_fitness'] for h in self.history]
            ax.semilogy(generations, mean_fits, color=COLORS['secondary'], 
                       linewidth=1, alpha=0.7, label='Mean MSE')
        
        # Mark best ever
        if self.best_fitness < float('inf'):
            ax.axhline(y=self.best_fitness, color=COLORS['highlight'], 
                      linestyle='--', alpha=0.5, label=f'Best: {self.best_fitness:.4f}')
        
        ax.set_xlabel('Generation', color=COLORS['text_dim'])
        ax.set_ylabel('MSE (log scale)', color=COLORS['text_dim'])
        ax.legend(loc='upper right', fontsize=8, facecolor=COLORS['background'],
                 edgecolor=COLORS['dim'], labelcolor=COLORS['text'])
        ax.grid(True, alpha=0.3, color=COLORS['grid'])
        ax.tick_params(colors=COLORS['text_dim'])
        for spine in ax.spines.values():
            spine.set_color(COLORS['dim'])
    
    def _draw_formula(self):
        """Draw the current best formula."""
        ax = self.ax_formula
        ax.clear()
        ax.set_facecolor(COLORS['background'])
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Best Formula', ha='center', va='top',
               color=COLORS['primary'], fontsize=14, fontweight='bold',
               transform=ax.transAxes)
        
        # Generation info
        ax.text(0.5, 0.8, f'Generation: {self.current_gen}', ha='center', va='top',
               color=COLORS['text_dim'], fontsize=10, transform=ax.transAxes)
        
        # Formula
        formula = self.current_formula or "Searching..."
        # Wrap long formulas
        if len(formula) > 40:
            # Try to break at + or -
            mid = len(formula) // 2
            for i in range(mid, len(formula)):
                if formula[i] in ['+', '-']:
                    formula = formula[:i] + '\n' + formula[i:]
                    break
        
        ax.text(0.5, 0.55, formula, ha='center', va='center',
               color=COLORS['highlight'], fontsize=12, fontweight='bold',
               transform=ax.transAxes, family='monospace',
               bbox=dict(boxstyle='round', facecolor=COLORS['background'],
                        edgecolor=COLORS['primary'], alpha=0.8))
        
        # Metrics
        metrics_text = f'MSE: {self.best_fitness:.6f}  |  Corr: {self.correlation:.4f}'
        ax.text(0.5, 0.2, metrics_text, ha='center', va='center',
               color=COLORS['accent'], fontsize=11, transform=ax.transAxes)
        
        # Status indicator
        if self.correlation > 0.99:
            status = "✓ STRUCTURE FOUND"
            status_color = COLORS['highlight']
        elif self.correlation > 0.9:
            status = "◐ Converging..."
            status_color = COLORS['accent']
        else:
            status = "○ Exploring..."
            status_color = COLORS['text_dim']
        
        ax.text(0.5, 0.05, status, ha='center', va='bottom',
               color=status_color, fontsize=10, fontweight='bold',
               transform=ax.transAxes)
    
    def _draw_fit_plot(self):
        """Draw prediction vs target scatter plot."""
        ax = self.ax_fit
        ax.clear()
        ax.set_facecolor(COLORS['background'])
        ax.set_title('Prediction vs Target', color=COLORS['primary'], fontsize=12)
        
        if self.x_data is None or self.y_data is None:
            ax.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center',
                   color=COLORS['text_dim'], fontsize=12, transform=ax.transAxes)
            return
        
        x_np = self.x_data.cpu().numpy().flatten() if torch.is_tensor(self.x_data) else self.x_data.flatten()
        y_np = self.y_data.cpu().numpy().flatten() if torch.is_tensor(self.y_data) else self.y_data.flatten()
        
        # Sort for line plot
        sort_idx = np.argsort(x_np)
        x_sorted = x_np[sort_idx]
        y_sorted = y_np[sort_idx]
        
        # Plot target
        ax.scatter(x_sorted, y_sorted, color=COLORS['primary'], s=20, alpha=0.6, label='Target')
        
        # Plot prediction if available
        if self.y_pred is not None:
            y_pred_np = self.y_pred.cpu().numpy().flatten() if torch.is_tensor(self.y_pred) else self.y_pred.flatten()
            y_pred_sorted = y_pred_np[sort_idx]
            ax.plot(x_sorted, y_pred_sorted, color=COLORS['highlight'], 
                   linewidth=2, label='Prediction')
        
        ax.set_xlabel('x', color=COLORS['text_dim'])
        ax.set_ylabel('y', color=COLORS['text_dim'])
        ax.legend(loc='upper right', fontsize=8, facecolor=COLORS['background'],
                 edgecolor=COLORS['dim'], labelcolor=COLORS['text'])
        ax.grid(True, alpha=0.3, color=COLORS['grid'])
        ax.tick_params(colors=COLORS['text_dim'])
        for spine in ax.spines.values():
            spine.set_color(COLORS['dim'])
    
    def update(self, frame=None):
        """Update all visualizations."""
        # Use blitting-friendly approach - only redraw what changed
        try:
            if not self.lite_mode:
                self._draw_network()
            self._draw_fitness_curve()
            self._draw_formula()
            self._draw_fit_plot()
            self.fig.canvas.draw_idle()  # More efficient than draw()
            self.fig.canvas.flush_events()  # Process events without blocking
        except Exception as e:
            pass  # Silently handle drawing errors to avoid crashes
        return []
    
    def update_from_trainer(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        generation: int,
        history: List[Dict],
        formula: str = "",
        best_fitness: float = float('inf'),
        correlation: float = 0.0,
    ):
        """
        Update visualization with data from the trainer.
        
        All tensor operations are done on CPU to avoid GPU contention.
        Call this after each generation in the evolutionary loop.
        """
        self.current_model = model
        self.current_gen = generation
        self.history = history
        self.current_formula = formula
        self.best_fitness = best_fitness
        self.correlation = correlation
        
        # Move data to CPU for visualization (GPU stays free for training)
        x_cpu = x.detach().cpu() if torch.is_tensor(x) else torch.tensor(x)
        y_cpu = y.detach().cpu() if torch.is_tensor(y) else torch.tensor(y)
        
        # Subsample for visualization (faster rendering)
        max_points = 150  # Reduced for better performance
        if len(x_cpu) > max_points:
            indices = torch.linspace(0, len(x_cpu)-1, max_points).long()
            self.x_data = x_cpu[indices]
            self.y_data = y_cpu[indices]
        else:
            self.x_data = x_cpu
            self.y_data = y_cpu
        
        # Get prediction on CPU (don't use GPU for visualization)
        if model is not None:
            try:
                model.eval()
                with torch.no_grad():
                    # Run inference on CPU for visualization
                    x_viz = self.x_data.to(next(model.parameters()).device)
                    pred, _ = model(x_viz, hard=True)
                    self.y_pred = pred.squeeze().cpu()  # Always move back to CPU
            except Exception:
                self.y_pred = None
        
        # Redraw with minimal blocking
        self.update()
        plt.pause(0.001)  # Minimal pause - just enough to process events
    
    def show(self, block: bool = False):
        """Show the visualization window."""
        plt.ion()  # Interactive mode
        self.update()
        plt.show(block=block)
    
    def close(self):
        """Close the visualization window."""
        plt.close(self.fig)
        self.running = False


class LiveTrainingVisualizer:
    """
    Wrapper for easy integration with EvolutionaryTrainer.
    
    Usage:
        viz = LiveTrainingVisualizer()
        trainer = EvolutionaryTrainer(..., visualizer=viz)
        trainer.train(x, y, generations=30)
    """
    
    def __init__(
        self,
        update_every: int = 1,  # Update every N generations
        figsize: Tuple[int, int] = (14, 8),
        lite_mode: bool = False,  # Skip network diagram for speed
    ):
        """
        Args:
            update_every: Update visualization every N generations
            figsize: Figure size
            lite_mode: If True, skip network diagram (faster)
        """
        self.update_every = update_every
        self.figsize = figsize
        self.lite_mode = lite_mode
        self.visualizer = None
        self._initialized = False
    
    def initialize(self, n_inputs: int, n_layers: int, nodes_per_layer: int):
        """Initialize the visualizer with network dimensions."""
        self.visualizer = ONNVisualizer(
            n_inputs=n_inputs,
            n_layers=n_layers,
            nodes_per_layer=nodes_per_layer,
            figsize=self.figsize,
            lite_mode=self.lite_mode,
        )
        self.visualizer.show(block=False)
        self._initialized = True
    
    def on_generation(
        self,
        generation: int,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        history: List[Dict],
        formula: str = "",
        best_fitness: float = float('inf'),
        correlation: float = 0.0,
    ):
        """
        Callback for each generation.
        
        Call this in your training loop after each generation.
        """
        if not self._initialized:
            return
        
        if generation % self.update_every == 0:
            self.visualizer.update_from_trainer(
                model=model,
                x=x,
                y=y,
                generation=generation,
                history=history,
                formula=formula,
                best_fitness=best_fitness,
                correlation=correlation,
            )
    
    def close(self):
        """Close the visualizer."""
        if self.visualizer:
            self.visualizer.close()


def create_network_diagram(
    model: nn.Module,
    save_path: Optional[str] = None,
    title: str = "ONN Architecture",
) -> plt.Figure:
    """
    Create a static network diagram from a trained model.
    
    Args:
        model: The ONN model
        save_path: Path to save the figure (optional)
        title: Title for the diagram
        
    Returns:
        matplotlib Figure
    """
    # Get model dimensions
    n_inputs = model.n_inputs if hasattr(model, 'n_inputs') else 1
    n_layers = model.n_hidden_layers if hasattr(model, 'n_hidden_layers') else 2
    nodes_per_layer = model.nodes_per_layer if hasattr(model, 'nodes_per_layer') else 4
    
    viz = ONNVisualizer(
        n_inputs=n_inputs,
        n_layers=n_layers,
        nodes_per_layer=nodes_per_layer,
        figsize=(12, 8),
    )
    viz.current_model = model
    
    # Get formula if available
    if hasattr(model, 'get_formula'):
        viz.current_formula = model.get_formula()
    
    viz.update()
    
    if save_path:
        viz.fig.savefig(save_path, facecolor=COLORS['background'], 
                       edgecolor='none', bbox_inches='tight', dpi=150)
        print(f"Saved network diagram to {save_path}")
    
    return viz.fig


# Convenience function for quick visualization
def visualize_evolution(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    formula: str = "",
    title: str = "ONN Result",
):
    """
    Quick visualization of a trained ONN.
    
    Args:
        model: Trained ONN model
        x: Input data
        y: Target data
        formula: Extracted formula string
        title: Plot title
    """
    n_inputs = model.n_inputs if hasattr(model, 'n_inputs') else 1
    n_layers = model.n_hidden_layers if hasattr(model, 'n_hidden_layers') else 2
    nodes_per_layer = model.nodes_per_layer if hasattr(model, 'nodes_per_layer') else 4
    
    viz = ONNVisualizer(
        n_inputs=n_inputs,
        n_layers=n_layers,
        nodes_per_layer=nodes_per_layer,
    )
    
    # Compute correlation
    model.eval()
    with torch.no_grad():
        pred, _ = model(x, hard=True)
        pred_np = pred.squeeze().cpu().numpy()
        y_np = y.squeeze().cpu().numpy()
        correlation = np.corrcoef(pred_np, y_np)[0, 1] if len(pred_np) > 1 else 0
    
    viz.update_from_trainer(
        model=model,
        x=x,
        y=y,
        generation=0,
        history=[],
        formula=formula,
        best_fitness=torch.nn.functional.mse_loss(pred.squeeze(), y.squeeze()).item(),
        correlation=correlation,
    )
    
    plt.show(block=True)


if __name__ == "__main__":
    # Demo visualization
    print("ONN Visualizer Demo")
    print("="*50)
    
    # Create dummy visualization
    viz = ONNVisualizer(n_inputs=1, n_layers=2, nodes_per_layer=4)
    
    # Simulate training data
    x = torch.linspace(-3, 3, 100).reshape(-1, 1)
    y = x.pow(2)
    
    # Simulate history
    for gen in range(20):
        viz.history.append({
            'generation': gen,
            'best_fitness': 10 * np.exp(-gen/5),
            'mean_fitness': 20 * np.exp(-gen/5),
        })
    
    viz.x_data = x
    viz.y_data = y
    viz.y_pred = x.pow(2) + torch.randn_like(x) * 0.5
    viz.current_formula = "x₀²"
    viz.best_fitness = 0.05
    viz.correlation = 0.998
    viz.current_gen = 20
    
    viz.show(block=True)
