import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import numpy as np
from glassbox.kan.diff_network import GlassboxDiffKAN

# Detect device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

def test_darts_full_kinetic():
    print("[Test] DARTS for FULL Kinetic Energy (0.5 * m * v^2)")
    
    # Generate Data
    m = np.random.uniform(1, 10, 1000)
    v = np.random.uniform(0, 10, 1000)
    y = 0.5 * m * (v**2) # The real deal
    
    X = torch.tensor(np.column_stack([m, v]), dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32).to(DEVICE)
    
    model = GlassboxDiffKAN(['m', 'v'], hidden_sizes=[4], output_size=1).to(DEVICE)
    
    # 2 Optimizers
    arch_params = []
    model_params = []
    for name, param in model.named_parameters():
        if 'weight' in name: # weights (ops) or weight (products)
            arch_params.append(param)
        else:
            model_params.append(param)
            
    optimizer_model = torch.optim.Adam(model_params, lr=0.05)
    optimizer_arch = torch.optim.Adam(arch_params, lr=0.02)
    
    criterion = nn.MSELoss()
    
    print("Training with Entropy Regularization...")
    for step in range(2000):
        optimizer_model.zero_grad()
        optimizer_arch.zero_grad()
        
        pred = model(X)
        mse = criterion(pred, y_t)
        ent = model.entropy_loss()
        
        # Loss = MSE + lambda * Entropy
        # Start lambda small, increase over time (annealing)
        lambda_ent = 0.01 * (step / 2000) 
        loss = mse + lambda_ent * ent
        
        loss.backward()
        
        optimizer_model.step()
        optimizer_arch.step()
        
        if step % 200 == 0:
            print(f"Step {step}: MSE = {mse.item():.4f}, Ent = {ent.item():.4f}")
            
    print(f"Final MSE: {mse.item():.4f}")
    
    # Analyze Results
    print("\n--- Discovered Structure ---")
    
    # Check Product Edges
    prod_idx = 0
    inputs = ['m', 'v']
    for i in range(2):
        for j in range(i, 2):
            edge = model.product_edges[prod_idx]
            gate = torch.sigmoid(edge.weight).item()
            if gate > 0.1:
                print(f"Product({inputs[i]}*{inputs[j]}): Gate={gate:.2f}, Scale={edge.scale1.item():.2f}*{edge.scale2.item():.2f}")
            prod_idx += 1
            
    model.prune_and_print()

if __name__ == "__main__":
    test_darts_full_kinetic()
