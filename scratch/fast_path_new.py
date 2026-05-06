# These go at module level
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

def find_exact_symbolic_match(
    basis: np.ndarray,
    names: List[str],
    y: np.ndarray,
    max_terms: int = 3,
    tolerance: float = 1e-6,
    num_threads: int = 1,
) -> Optional[Tuple[str, float, np.ndarray]]:
    """
    Search for exact symbolic matches before falling back to LASSO.
    
    Tries single terms, pairs, and triples of basis functions to find
    exact symbolic solutions (MSE < tolerance).
    
    Args:
        basis: (N, n_basis) matrix
        names: List of basis function names
        y: Target values (N,)
        max_terms: Maximum number of terms to try in combination
        tolerance: MSE threshold for "exact" match
        
    Returns:
        (formula, mse, coefficients) if exact match found, else None
    """
    import math
    try:
        from joblib import Parallel, delayed
    except ImportError:
        Parallel = None

    n_basis = basis.shape[1]
    y = y.flatten()
    
    # Try single basis functions with coefficient fitting
    for i in range(n_basis):
        if names[i] == "1":  # Skip constant-only
            continue
        
        # Try with and without constant
        for include_const in [False, True]:
            if include_const:
                X = np.column_stack([np.ones(len(y)), basis[:, i]])
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    y_pred = X @ coeffs
                    mse = np.mean((y - y_pred) ** 2)
                    if mse < tolerance:
                        formula = _format_affine_formula(names[i], float(coeffs[1]), float(coeffs[0]))
                        
                        full_coeffs = np.zeros(n_basis)
                        const_idx = names.index("1") if "1" in names else 0
                        full_coeffs[const_idx] = coeffs[0] if include_const else 0
                        full_coeffs[i] = coeffs[1] if include_const else coeffs[0]
                        return formula, mse, full_coeffs
                except (np.linalg.LinAlgError, ValueError):
                    pass
            else:
                X = basis[:, i:i+1]
                try:
                    coeff, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    y_pred = X @ coeff
                    mse = np.mean((y - y_pred) ** 2)
                    if mse < tolerance:
                        formula = _format_affine_formula(names[i], float(coeff[0]), 0.0)
                        
                        full_coeffs = np.zeros(n_basis)
                        full_coeffs[i] = coeff[0]
                        return formula, mse, full_coeffs
                except (np.linalg.LinAlgError, ValueError):
                    pass
    
    def chunk_ranges(n: int, chunks: int) -> List[Tuple[int, int]]:
        if chunks <= 1 or n <= 1:
            return [(0, n)]
        size = max(1, math.ceil(n / chunks))
        return [(i, min(i + size, n)) for i in range(0, n, size)]

    def build_formula(indices: List[int], coeffs: np.ndarray) -> Tuple[str, np.ndarray]:
        terms = []
        full_coeffs = np.zeros(n_basis)
        for idx, c in zip(indices, coeffs):
            if abs(c) < 1e-6:
                continue
            name = names[idx]
            if name == "1":
                terms.append(get_constant_symbol(c, 0.05))
            elif abs(c - 1.0) < 0.01:
                terms.append(name)
            elif abs(c + 1.0) < 0.01:
                terms.append(f"-{name}")
            elif abs(c - round(c)) < 0.01 and abs(c) < 100:
                terms.append(f"{int(round(c))}*{name}")
            else:
                coef_sym = get_constant_symbol(c, 0.05)
                terms.append(f"{coef_sym}*{name}")
            full_coeffs[idx] = c

        formula = _join_formula_terms(terms)
        return formula, full_coeffs

    # Try pairs of basis functions (including constant)
    if max_terms >= 2:
        if num_threads > 1 and Parallel is not None:
            ranges = chunk_ranges(n_basis, num_threads * 2)
            results = Parallel(n_jobs=num_threads, return_as="generator_unordered")(
                delayed(_search_pairs_chunk)(start, end, basis, y, tolerance) for start, end in ranges
            )
            for result in results:
                if result is not None:
                    indices, coeffs_list, mse = result
                    formula, full_coeffs = build_formula(indices, np.array(coeffs_list))
                    return formula, mse, full_coeffs
        else:
            result = _search_pairs_chunk(0, n_basis, basis, y, tolerance)
            if result is not None:
                indices, coeffs_list, mse = result
                formula, full_coeffs = build_formula(indices, np.array(coeffs_list))
                return formula, mse, full_coeffs
    
    # Try triples of basis functions (including constant)
    if max_terms >= 3:
        if num_threads > 1 and Parallel is not None:
            # More chunks for better load balancing
            ranges = chunk_ranges(n_basis, num_threads * 4)
            results = Parallel(n_jobs=num_threads, return_as="generator_unordered")(
                delayed(_search_triples_chunk)(start, end, basis, y, tolerance) for start, end in ranges
            )
            for result in results:
                if result is not None:
                    indices, coeffs_list, mse = result
                    formula, full_coeffs = build_formula(indices, np.array(coeffs_list))
                    return formula, mse, full_coeffs
        else:
            result = _search_triples_chunk(0, n_basis, basis, y, tolerance)
            if result is not None:
                indices, coeffs_list, mse = result
                formula, full_coeffs = build_formula(indices, np.array(coeffs_list))
                return formula, mse, full_coeffs
    
    return None
