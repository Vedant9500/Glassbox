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

    def search_pairs_range(start_i: int, end_i: int, stop_event: threading.Event):
        for i in range(start_i, end_i):
            if stop_event.is_set():
                return None
            for j in range(i + 1, n_basis):
                X = basis[:, [i, j]]
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    y_pred = X @ coeffs
                    mse = np.mean((y - y_pred) ** 2)
                    if mse < tolerance:
                        formula, full_coeffs = build_formula([i, j], coeffs)
                        stop_event.set()
                        return formula, mse, full_coeffs
                except (np.linalg.LinAlgError, ValueError):
                    pass
        return None

    def search_triples_range(start_i: int, end_i: int, stop_event: threading.Event):
        for i in range(start_i, end_i):
            if stop_event.is_set():
                return None
            for j in range(i + 1, n_basis):
                for k in range(j + 1, n_basis):
                    X = basis[:, [i, j, k]]
                    try:
                        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                        y_pred = X @ coeffs
                        mse = np.mean((y - y_pred) ** 2)
                        if mse < tolerance:
                            formula, full_coeffs = build_formula([i, j, k], coeffs)
                            stop_event.set()
                            return formula, mse, full_coeffs
                    except (np.linalg.LinAlgError, ValueError):
                        pass
        return None

    # Try pairs of basis functions (including constant)
    if max_terms >= 2:
        if num_threads and num_threads > 1:
            stop_event = threading.Event()
            ranges = chunk_ranges(n_basis, num_threads)
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(search_pairs_range, start, end, stop_event) for start, end in ranges]
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        return result
        else:
            result = search_pairs_range(0, n_basis, threading.Event())
            if result is not None:
                return result
    
    # Try triples of basis functions (including constant)
    if max_terms >= 3:
        if num_threads and num_threads > 1:
            stop_event = threading.Event()
            ranges = chunk_ranges(n_basis, num_threads)
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(search_triples_range, start, end, stop_event) for start, end in ranges]
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        return result
        else:
            result = search_triples_range(0, n_basis, threading.Event())
            if result is not None:
                return result
    
    return None

