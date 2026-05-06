import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def chunk_triples(args):
    start, end, basis, y, tol = args
    n_basis = basis.shape[1]
    
    # We only process a slice of 'i'
    for i in range(start, end):
        for j in range(i + 1, n_basis):
            for k in range(j + 1, n_basis):
                X = basis[:, [i, j, k]]
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    y_pred = X @ coeffs
                    mse = np.mean((y - y_pred) ** 2)
                    if mse < tol:
                        return ([i,j,k], coeffs.tolist(), mse)
                except (np.linalg.LinAlgError, ValueError):
                    pass
    return None

if __name__ == "__main__":
    np.random.seed(0)
    basis = np.random.randn(1000, 150)
    y = np.random.randn(1000)
    
    n_basis = 150
    chunks = []
    # chunk by 'i'
    # There are 150 values of i.
    for i in range(n_basis):
        chunks.append((i, i+1, basis, y, 1e-6))
    
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as ex:
        for res in ex.map(chunk_triples, chunks):
            if res is not None:
                print("Found:", res)
                break
    print("Time ProcPool:", time.time() - t0)

