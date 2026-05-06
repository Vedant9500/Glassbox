import time
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

def chunk_triples_joblib(start, end, basis, y, tol):
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
    t0 = time.time()
    
    results = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(chunk_triples_joblib)(i, i+1, basis, y, 1e-6) for i in range(n_basis)
    )
    for res in results:
        if res is not None:
            print("Found:", res)
            break
            
    print("Time joblib:", time.time() - t0)

