import torch
import time
import numpy as np

N = 1000
n_basis = 50

basis = torch.randn(N, n_basis)
y = torch.randn(N, 1)

t0 = time.time()

# Let's count pairs: 50*49/2 = 1225
pairs = [(i,j) for i in range(n_basis) for j in range(i+1, n_basis)]
batch_X = basis[:, pairs] # wait, basis is Nx50. basis[:, pairs] would be N x 1225 x 2.
# PyTorch allows selecting:
X = basis[:, pairs] # Shape: N x 1225 x 2.
X = X.permute(1, 0, 2) # 1225 x N x 2

y_batch = y.expand(1225, N, 1)

print(X.shape, y_batch.shape)
coeffs = torch.linalg.lstsq(X, y_batch).solution # 1225 x 2 x 1
pred = torch.bmm(X, coeffs) # 1225 x N x 1
mse = torch.mean((y_batch - pred)**2, dim=1).squeeze(-1) # 1225

print(mse.min())
print("Time:", time.time() - t0)
