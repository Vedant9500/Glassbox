# 🔬 Curve Classifier Architectural Audit

> **Component:** `glassbox/curve_classifier/`  
> **Files analyzed:** `generate_curve_data.py` (1858 lines), `curve_classifier_integration.py` (785 lines), `train_curve_classifier.py` (980 lines), `classifier_fast_path.py` (3194 lines)  
> **Goal:** Make the classifier smarter, more accurate, more robust, and basis-independent.

---

## The Current System in 30 Seconds

The curve classifier takes a 1D curve `y = f(x)`, extracts a **370-dimensional feature vector** (128 raw resampled points + 32 FFT magnitudes + 32 FFT phases + 128 derivative samples + 9 statistical moments + 37 curvature features + 4 differential invariants), and feeds it through a neural net (MLP/CNN/GLU) to produce **multi-hot operator probabilities** (sin, cos, power, exp, log, addition, multiplication, rational, identity — plus 5 constant classes).

These predictions then warm-start the fast-path LASSO regression and bias the evolutionary ONN search.

---

## 🔴 Critical Weakness #1: The Feature Vector is NOT Basis-Independent

### The Paradigm Shift
*The entire classifier is secretly an interpolation machine over a fixed grid, not a mathematical structure recognizer. It memorizes what `sin(x)` looks like on `[-5, 5]` with 256 points — change the domain to `[0, 100]` or resample to 50 points, and every single feature changes unpredictably.*

### The Elegant Solution — Differential Signature Embedding

The fundamental mathematical truth: **the operators in `y = f(x)` are properties of the function, not the coordinate system**. A sine wave is a sine wave whether sampled at 50 points or 50,000, whether on `[-1, 1]` or `[-100, 100]`.

The key insight is that certain **differential relationships** are invariant to affine reparameterization `x → αx + β` and rescaling `y → γy + δ`:

| Invariant | Formula | What it identifies |
|---|---|---|
| **Exponential ratio** | `y'' / y'` = const | Pure exponentials `e^(ax)` |
| **Sinusoidal ratio** | `y''' / y'` = const | Pure sinusoids `sin(ωx)` |
| **Power-law ratio** | `(y' · y''') / (y'')²` = const | Power laws `x^p` |
| **Schwarzian derivative** | `y'''/y' − 3/2·(y''/y')²` | Rational / Möbius functions |

You already have these 4 invariants in `extract_differential_invariants()`, but they are **buried as 4 features out of 370** (only 1.08% of the feature vector). The classifier drowns them in 360 features that are **all basis-dependent**.

**Proposed architecture: Invariant-First Feature Hierarchy**

```
Level 0 (32 features) — Pure Invariants (basis-free):
  - 4 differential invariants (current)
  - 8 ratio-stability features (running variance of each ratio over windows)
  - 4 autocorrelation decay rates of each invariant
  - 4 invariant cross-correlations (exp×sin, power×rational, etc.)
  - 8 zero-crossing patterns of invariant channels  
  - 4 invariant spectral entropy values

Level 1 (64 features) — Scale-Normalized Signatures:
  - FFT magnitudes normalized by L2-norm (not max) → removes amplitude basis
  - Derivative ratios (dy/y, d²y/y) instead of raw derivatives → removes y-scale basis
  - Curvature profile normalized by arc-length → removes x-scale basis

Level 2 (128 features) — Conditional Raw Features:
  - Current raw resampled curve (but now GATED by Level 0 confidence)
  - Only used when invariants are ambiguous
```

**Why this works:** Level 0 features are mathematically invariant to `(x, y) → (αx + β, γy + δ)`. The classifier can identify operators purely from structural relationships. Level 1 adds discriminative power with normalization tricks. Level 2 provides fallback detail.

### C++ Execution

This is Python-native, but the core computation reduces to:

```python
def extract_invariant_features(y: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """32-dimensional basis-invariant feature vector."""
    n = len(y)
    if n < 15:
        return np.zeros(32, dtype=np.float64)
    
    # Lanczos-smoothed derivatives (7-point stencils — already in codebase)
    k1 = np.array([3, 2, 1, 0, -1, -2, -3]) / (28.0 * dx)
    k2 = np.array([5, 0, -3, -4, -3, 0, 5]) / (42.0 * dx**2)
    k3 = np.array([1, -1, -1, 0, 1, 1, -1]) / (6.0 * dx**3)
    
    dy = np.convolve(y, k1, mode='valid')
    ddy = np.convolve(y, k2, mode='valid')
    dddy = np.convolve(y, k3, mode='valid')
    
    eps = 1e-6
    safe_dy = np.where(np.abs(dy) > eps, dy, eps * np.sign(dy + 1e-12))
    safe_ddy = np.where(np.abs(ddy) > eps, ddy, eps * np.sign(ddy + 1e-12))
    
    # 4 raw invariant channels
    inv_exp = ddy / safe_dy                          # y''/y' → constant for exp
    inv_sin = dddy / safe_dy                         # y'''/y' → constant for sin
    inv_pow = (dy * dddy) / (safe_ddy ** 2)          # (y'·y''')/(y'')² → constant for power
    inv_rat = inv_sin - 1.5 * (inv_exp ** 2)         # Schwarzian → 0 for Möbius

    channels = [inv_exp, inv_sin, inv_pow, inv_rat]
    
    features = []
    for ch in channels:
        ch_clean = np.nan_to_num(np.clip(ch, -100, 100), nan=0.0)
        
        # Core statistics (4 per channel = 16 total)
        features.append(np.var(ch_clean))              # Variance (low = pure operator)
        features.append(np.median(np.abs(ch_clean)))   # Robust center
        
        # Windowed stability: split into 4 windows, measure inter-window variance
        windows = np.array_split(ch_clean, 4)
        window_means = [np.mean(w) for w in windows if len(w) > 0]
        features.append(np.var(window_means) if len(window_means) > 1 else 0.0)
        
        # Zero-crossing rate (captures periodicity of the invariant itself)
        zc = np.sum(np.diff(np.sign(ch_clean)) != 0) / max(len(ch_clean) - 1, 1)
        features.append(zc)
    
    # Cross-correlations between invariant channels (6 pairs)
    for i in range(4):
        for j in range(i + 1, 4):
            ci = np.nan_to_num(np.clip(channels[i], -100, 100))
            cj = np.nan_to_num(np.clip(channels[j], -100, 100))
            m = min(len(ci), len(cj))
            if m > 2:
                corr = np.corrcoef(ci[:m], cj[:m])[0, 1]
                features.append(0.0 if np.isnan(corr) else corr)
            else:
                features.append(0.0)
    
    # Spectral entropy of each invariant channel (4 values)
    for ch in channels:
        ch_clean = np.nan_to_num(np.clip(ch, -100, 100))
        if len(ch_clean) >= 8:
            fft_mag = np.abs(np.fft.rfft(ch_clean))
            fft_mag = fft_mag / (np.sum(fft_mag) + 1e-12)
            entropy = -np.sum(fft_mag * np.log(fft_mag + 1e-12))
            features.append(entropy)
        else:
            features.append(0.0)
    
    # Autocorrelation decay rate of each channel (4 values)
    for ch in channels:
        ch_clean = np.nan_to_num(np.clip(ch, -100, 100))
        if len(ch_clean) >= 8:
            autocorr = np.correlate(ch_clean - np.mean(ch_clean), 
                                     ch_clean - np.mean(ch_clean), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-12)
            # Decay rate: where autocorrelation drops below 1/e
            below_threshold = np.where(autocorr < 1/np.e)[0]
            decay = below_threshold[0] / len(ch_clean) if len(below_threshold) > 0 else 1.0
            features.append(decay)
        else:
            features.append(1.0)
    
    result = np.array(features[:32], dtype=np.float64)
    if len(result) < 32:
        result = np.pad(result, (0, 32 - len(result)))
    return result
```

> **Impact estimate:** Current differential invariants are 4 numbers. This expands to 32 rich, basis-invariant features. When used as the **primary decision layer** in a hierarchical model, this should dramatically improve generalization to unseen domains `[a, b]`.

---

## 🔴 Critical Weakness #2: The Multi-Input Strategy is Interpolation Theater

### The Paradigm Shift
*For multivariate inputs, the classifier slices through each variable independently at the median of all others, then takes `max()` over predictions. This is fundamentally wrong — it cannot detect **interaction** operators like `x₁ · sin(x₂)` or `x₁² / x₂` because those vanish or flatten when you fix one variable at its median.*

### The Elegant Solution — Mutual Information Interaction Detection

Instead of 1D slicing, detect **which variables interact** before classifying:

```
Step 1: Compute pairwise Friedman H-statistic (partial dependence interaction)
Step 2: For strongly-interacting variable pairs (H > 0.1):
          - Extract 2D partial dependence surface
          - Compute gradient field ∇f on the surface
          - The CURL of the gradient field is zero iff the variables are additively separable
          - Non-zero curl → multiplicative interaction detected
Step 3: For each variable group (independent or interacting):
          - Run the invariant-first classifier on the appropriate slice/surface
Step 4: Merge predictions with interaction-aware aggregation (not just max)
```

This solves the specific blind spot: `y = x₁ · sin(x₂)` has a non-separable partial dependence surface, so the `multiplication` operator gets detected from the curl analysis, and `sin` gets detected from the x₂-axis invariants.

### C++ Execution

```python
def detect_variable_interactions(
    x: np.ndarray, 
    y: np.ndarray,
    n_grid: int = 20,
    interaction_threshold: float = 0.1,
) -> List[Tuple[int, int, float]]:
    """Detect pairwise variable interactions using partial dependence.
    
    Returns list of (var_i, var_j, H_statistic) for interacting pairs.
    """
    n_vars = x.shape[1]
    interactions = []
    
    # Total variance of the response
    var_y = np.var(y)
    if var_y < 1e-12:
        return interactions
    
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            # Compute partial dependence for f(xi, xj)
            xi_grid = np.linspace(x[:, i].min(), x[:, i].max(), n_grid)
            xj_grid = np.linspace(x[:, j].min(), x[:, j].max(), n_grid)
            
            # PD values: average response over all other variables
            pd_ij = np.zeros((n_grid, n_grid))
            pd_i = np.zeros(n_grid)
            pd_j = np.zeros(n_grid)
            
            from scipy.interpolate import NearestNDInterpolator
            interp = NearestNDInterpolator(x, y)
            
            x_template = np.median(x, axis=0)
            
            for gi in range(n_grid):
                x_query = x_template.copy()
                x_query[i] = xi_grid[gi]
                # Marginal PD for variable i
                pd_i[gi] = interp(x_query.reshape(1, -1))[0]
                
                for gj in range(n_grid):
                    x_query_ij = x_template.copy()
                    x_query_ij[i] = xi_grid[gi]
                    x_query_ij[j] = xj_grid[gj]
                    pd_ij[gi, gj] = interp(x_query_ij.reshape(1, -1))[0]
            
            for gj in range(n_grid):
                x_query = x_template.copy()
                x_query[j] = xj_grid[gj]
                pd_j[gj] = interp(x_query.reshape(1, -1))[0]
            
            # H-statistic: variance of interaction residual / variance of joint PD
            interaction_surface = pd_ij - pd_i[:, None] - pd_j[None, :] + np.mean(y)
            H = np.var(interaction_surface) / max(np.var(pd_ij), 1e-12)
            
            if H > interaction_threshold:
                interactions.append((i, j, float(H)))
    
    return sorted(interactions, key=lambda t: -t[2])
```

> **Impact estimate:** This addresses the "multivariate blindness" identified in the [Tier 4 audit](file:///d:/Glassbox/scripts/benchmark_suite.py). Formulas like `G·m₁·m₂/r²` require detecting the 3-way multiplicative interaction, which 1D slicing structurally cannot do.

---

## 🟡 Weakness #3: Feature Extraction Has a Noise Amplification Problem

### The Paradigm Shift
*Computing `np.gradient()` twice on noisy data amplifies noise quadratically. The `_smooth_signal()` mitigation is a band-aid — it blurs real structure along with noise. The real fix isn't better smoothing, it's computing derivatives in a space where noise doesn't amplify.*

### The Elegant Solution — Spectral Differentiation

Instead of finite-difference derivatives (which amplify high-frequency noise), compute derivatives **in the Fourier domain** where differentiation is multiplication by `iω`:

```python
def spectral_derivatives(y: np.ndarray, n_points: int = 64) -> np.ndarray:
    """Compute derivatives via spectral differentiation.
    
    In Fourier space: d/dx → iω, d²/dx² → -ω²
    This naturally suppresses high-frequency noise because the FFT
    already separates signal from noise, and we only need to multiply
    — not amplify via finite differences.
    """
    n = len(y)
    # Detrend to enforce periodicity assumption
    y_detrended = y - np.linspace(y[0], y[-1], n)
    
    # FFT
    Y = np.fft.rfft(y_detrended)
    freqs = np.fft.rfftfreq(n, d=1.0/n)
    omega = 2 * np.pi * freqs
    
    # Automatic noise floor: zero out coefficients below 1% of max
    magnitudes = np.abs(Y)
    noise_floor = 0.01 * np.max(magnitudes)
    Y_clean = np.where(magnitudes > noise_floor, Y, 0.0)
    
    # Spectral derivative: multiply by iω
    dY = 1j * omega * Y_clean
    ddY = -(omega ** 2) * Y_clean
    
    # Inverse FFT back to spatial domain
    dy = np.fft.irfft(dY, n=n)
    ddy = np.fft.irfft(ddY, n=n)
    
    # Add back trend derivative
    trend_slope = (y[-1] - y[0]) / max(n - 1, 1)
    dy += trend_slope
    
    # Resample to fixed size
    dy_resampled = np.interp(np.linspace(0, 1, n_points), np.linspace(0, 1, n), dy)
    ddy_resampled = np.interp(np.linspace(0, 1, n_points), np.linspace(0, 1, n), ddy)
    
    # Normalize
    dy_max = np.abs(dy_resampled).max()
    ddy_max = np.abs(ddy_resampled).max()
    if dy_max > 1e-10:
        dy_resampled /= dy_max
    if ddy_max > 1e-10:
        ddy_resampled /= ddy_max
    
    return np.concatenate([dy_resampled, ddy_resampled])
```

> **Impact estimate:** This eliminates the Savitzky-Golay dependency and makes derivative features robust to both uniform and non-uniform noise profiles. Particularly impactful for the `noise_profile='multi'` training regime.

---

## 🟡 Weakness #4: The PCFG Grammar Has a Hidden Depth Bias

### The Paradigm Shift
*The PCFG generator creates formulas with Lample & Charton-style depth splitting, but the production weights (`unary=0.30, binary=0.25, term=0.45`) heavily favor shallow formulas. At depth 4, the probability of reaching a depth-4 composition like `sin(exp(x²))` is `0.30 × 0.30 × 0.30 × 0.45 ≈ 1.2%`. The classifier is starved of deep compositions at training time.*

### The Elegant Solution — Curriculum-Scheduled Depth Annealing

Instead of fixed production weights, **anneal the depth bias during data generation**:

```python
class DepthAnnealedPCFG(PCFGFormulaGenerator):
    """PCFG with scheduled depth annealing.
    
    Early in generation: shallow formulas (builds basic operator recognition)
    Late in generation: deep compositions (builds compositional reasoning)
    """
    
    def __init__(self, max_depth: int = 6):  # Increased from 4
        super().__init__(max_depth=max_depth)
        self.progress = 0.0  # 0.0 = start, 1.0 = end of generation
    
    def set_progress(self, progress: float):
        """Set generation progress for weight annealing."""
        self.progress = np.clip(progress, 0.0, 1.0)
        # Anneal from shallow-heavy to deep-heavy
        t = self.progress
        self.weights = {
            'unary': 0.20 + 0.25 * t,    # 0.20 → 0.45
            'binary': 0.15 + 0.20 * t,   # 0.15 → 0.35
            'term': 0.65 - 0.45 * t,     # 0.65 → 0.20
        }
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
```

This creates a natural curriculum: the classifier first learns to recognize individual operators, then gradually learns to decompose nested compositions — exactly how a human mathematician would learn.

---

## 🟡 Weakness #5: The Constant Detection Classes Are Useless Noise

### The Paradigm Shift

*Classes 9–13 (`const_pi`, `const_e`, `const_1`, `const_2`, `const_half`) cannot be detected from a curve's shape. The value of a multiplicative constant is absorbed into the amplitude normalization during `extract_raw_features()`. These classes pollute the label space and waste model capacity.*

### The Elegant Solution — Post-Classification Constant Snapping

Remove constants from the classifier entirely. Instead, detect them **after** the fast-path LASSO regression finds coefficients:

```python
KNOWN_CONSTANTS = {
    'pi': (np.pi, 'π'),
    'e': (np.e, 'e'),
    'sqrt2': (np.sqrt(2), '√2'),
    'phi': ((1 + np.sqrt(5)) / 2, 'φ'),  # Golden ratio
}

def snap_coefficient_to_constant(value: float, tolerance: float = 0.02) -> Optional[str]:
    """Check if a LASSO coefficient is close to a known mathematical constant."""
    abs_val = abs(value)
    for name, (const, symbol) in KNOWN_CONSTANTS.items():
        for mult in [1.0, 2.0, 0.5, -1.0, -2.0, -0.5]:
            if abs(abs_val - abs(const * mult)) / max(abs(const * mult), 1e-6) < tolerance:
                sign = '-' if value * mult < 0 else ''
                mult_str = '' if abs(mult) == 1.0 else f'{abs(mult):.0f}·'
                return f"{sign}{mult_str}{symbol}"
    return None
```

This is **architecturally correct** because constants are properties of the coefficient values, not the curve shape.

> **Impact estimate:** Removing 5 classes shrinks the output space from 14 to 9, reducing per-class confusion and freeing model capacity for the operators that actually matter. The constant detection moves to the post-regression phase where it's well-posed.

---

## 🟢 Opportunity #1: Ensemble the Invariant Classifier with the Shape Classifier

### The Paradigm Shift
*Don't choose between basis-invariant features and shape features. Use both as complementary experts.*

### The Elegant Solution — Gated Expert Mixture

```python
class InvariantGatedExpert(nn.Module):
    """Two-expert classifier: invariant expert + shape expert, with learned gating."""
    
    def __init__(self, n_invariant: int = 32, n_shape: int = 338, 
                 n_classes: int = 9, hidden: int = 128):
        super().__init__()
        
        # Expert 1: Invariant features → operator prediction
        self.invariant_expert = nn.Sequential(
            nn.Linear(n_invariant, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_classes),
        )
        
        # Expert 2: Shape features → operator prediction (existing GLU/MLP)
        self.shape_expert = nn.Sequential(
            nn.Linear(n_shape, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
        )
        self.shape_glu = nn.GLU(dim=1)
        self.shape_head = nn.Linear(hidden, n_classes)
        
        # Gating network: decides trust in each expert
        self.gate = nn.Sequential(
            nn.Linear(n_invariant + n_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x_invariant, x_shape):
        # Expert predictions
        logits_inv = self.invariant_expert(x_invariant)
        
        shape_h = self.shape_expert(x_shape)
        shape_h = self.shape_glu(shape_h)
        logits_shape = self.shape_head(shape_h)
        
        # Gating
        gate_input = torch.cat([x_invariant, x_shape], dim=1)
        gate_weights = self.gate(gate_input)  # (batch, 2)
        
        # Weighted combination
        logits = (gate_weights[:, 0:1] * logits_inv + 
                  gate_weights[:, 1:2] * logits_shape)
        
        return logits
```

**Why this is elegant:** When the invariant expert is confident (low variance in differential ratios), the gate trusts it. When the data is noisy or too short for reliable derivatives, the gate falls back to shape features. The classifier automatically learns when each expert is reliable.

---

## 🟢 Opportunity #2: Train on Scale-Augmented Data (Cheap Robustness)

### The Paradigm Shift
*The training data is generated on `x ∈ [-5, 5]` with mild augmentation (`x_scale ∈ [0.8, 1.2]`). This is far too narrow. Real-world data arrives on arbitrary domains.*

### The Elegant Solution

The existing augmentation infrastructure already supports this — just widen the parameters drastically during generation:

```bash
python generate_curve_data.py \
  --x-ranges "-0.5:0.5,-1:1,-2:2,-5:5,-10:10,-50:50,-100:100" \
  --x-scale-min 0.1 \
  --x-scale-max 10.0 \
  --x-shift-std 0.3 \
  --y-scale-min 0.01 \
  --y-scale-max 100.0 \
  --y-offset-std 2.0 \
  --noise-profile multi \
  --pcfg-ratio 0.4 \
  --pcfg-max-depth 6 \
  --balance-classes \
  --n-samples 500000
```

**This is the single highest-ROI change.** It requires zero code changes and directly attacks basis dependence at the training distribution level.

---

## 📊 Summary: Priority-Ranked Action Plan

| # | Change | Effort | Impact | Basis Independence |
|---|--------|--------|--------|-------------------|
| 1 | **Widen training augmentation** (x-scale, y-scale, x-ranges) | ⚡ None (config only) | 🟢 High | ✅ Direct |
| 2 | **Expand invariant features** from 4 → 32 dimensions | 🔨 Medium | 🟢 High | ✅ Direct |
| 3 | **Spectral differentiation** replacing finite-difference | 🔨 Medium | 🟡 Medium | ✅ Indirect |
| 4 | **Remove constant classes** (9–13), detect post-regression | 🔨 Low | 🟡 Medium | ⬜ Declutter |
| 5 | **Gated expert mixture** (invariant + shape) | 🔨 High | 🟢 High | ✅ Direct |
| 6 | **Interaction detection** for multivariate | 🔨 High | 🟢 High for Tier 3+ | ✅ Direct |
| 7 | **PCFG depth annealing** curriculum | 🔨 Low | 🟡 Medium | ⬜ Coverage |

> [!IMPORTANT]
> **Start with #1 (free) and #2 (medium effort, huge payoff).** The invariant feature expansion alone should measurably improve out-of-distribution generalization because the classifier will have a strong basis-independent signal to rely on, rather than memorizing curve shapes on `[-5, 5]`.

> [!TIP]
> **Quick validation test:** After implementing #2, train two models — one with invariant features, one without. Evaluate both on curves generated with `x ∈ [-100, 100]` (never seen during training). The invariant-heavy model should dramatically outperform on this out-of-distribution test.
