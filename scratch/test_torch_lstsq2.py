import torch
import time
import numpy as np

def fast_triples(basis, y, tolerance=1e-6):
    n_basis = basis.shape[1]
    N = basis.shape[0]
    
    # Pre-build tensor
    basis_t = torch.from_numpy(basis).float()
    y_t = torch.from_numpy(y).float().unsqueeze(1) # N x 1
    
    t0 = time.time()
    
    # We want to chunk the triples so we don't blow up memory
    # Total triples: n_basis * (n_basis - 1) * (n_basis - 2) // 6
    # For n=150, that's ~550k
    
    # Let's generate all triples efficiently
    idx = torch.combinations(torch.arange(n_basis), r=3)
    n_triples = len(idx)
    
    chunk_size = 50000
    
    for start in range(0, n_triples, chunk_size):
        end = min(start + chunk_size, n_triples)
        chunk_idx = idx[start:end] # size: C x 3
        
        X = basis_t[:, chunk_idx] # N x C x 3
        X = X.permute(1, 0, 2) # C x N x 3
        
        y_batch = y_t.expand(end - start, N, 1) # C x N x 1
        
        sol = torch.linalg.lstsq(X, y_batch).solution # C x 3 x 1
        pred = torch.bmm(X, sol) # C x N x 1
        mse = torch.mean((y_batch - pred)**2, dim=1).squeeze(-1) # C
        
        best_mse, best_idx = torch.min(mse, dim=0)
        
        if best_mse < tolerance:
            idx_in_chunk = best_idx.item()
            real_idx = start + idx_in_chunk
            return idx[real_idx].tolist(), sol[idx_in_chunk].flatten().tolist(), best_mse.item()
            
    return None

import time

np.random.seed(0)
basis = np.random.randn(1000, 150)
y = np.random.randn(1000)

t0=time.time()
res = fast_triples(basis, y)
print("Time:", time.time() - t0, res)

