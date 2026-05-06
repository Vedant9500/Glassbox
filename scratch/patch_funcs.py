import re

def _search_pairs_chunk(chunk_start_i: int, chunk_end_i: int, basis: np.ndarray, y: np.ndarray, tolerance: float):
    import numpy as np
    n_basis = basis.shape[1]
    for i in range(chunk_start_i, chunk_end_i):
        for j in range(i + 1, n_basis):
            X = basis[:, [i, j]]
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                y_pred = X @ coeffs
                mse = np.mean((y - y_pred) ** 2)
                if mse < tolerance:
                    return [i, j], coeffs.tolist(), float(mse)
            except (np.linalg.LinAlgError, ValueError):
                pass
    return None

def _search_triples_chunk(chunk_start_i: int, chunk_end_i: int, basis: np.ndarray, y: np.ndarray, tolerance: float):
    import numpy as np
    n_basis = basis.shape[1]
    for i in range(chunk_start_i, chunk_end_i):
        for j in range(i + 1, n_basis):
            for k in range(j + 1, n_basis):
                X = basis[:, [i, j, k]]
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    y_pred = X @ coeffs
                    mse = np.mean((y - y_pred) ** 2)
                    if mse < tolerance:
                        return [i, j, k], coeffs.tolist(), float(mse)
                except (np.linalg.LinAlgError, ValueError):
                    pass
    return None

