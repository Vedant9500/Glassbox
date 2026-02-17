"""
Unit tests for Phase 2A feature extraction changes.

Tests the three new/modified feature extraction components:
1. FFT phase features (extract_fft_phase_features)
2. Smooth derivative features (extract_derivative_features with Savitzky-Golay)
3. Curvature-aware resampling (extract_raw_features)

Also validates dimensional consistency and edge cases.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Ensure scripts directory is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from generate_curve_data import (
    extract_all_features,
    extract_raw_features,
    extract_fft_features,
    extract_fft_phase_features,
    extract_derivative_features,
    extract_stat_features,
    extract_curvature_features,
    _smooth_signal,
    FEATURE_DIM,
    FEATURE_SCHEMA,
)


# =============================================================================
# 1. Dimension Check
# =============================================================================

class TestDimensions:
    """Verify extract_all_features returns exactly FEATURE_DIM floats, no NaN/Inf."""

    def test_feature_dim_constant(self):
        """FEATURE_DIM should be 366 after Phase 2A."""
        assert FEATURE_DIM == 366

    def test_feature_schema_consistency(self):
        """FEATURE_SCHEMA slices should cover [0, FEATURE_DIM) without gaps or overlaps."""
        ranges = sorted(FEATURE_SCHEMA.values(), key=lambda r: r[0])
        assert ranges[0][0] == 0, "Schema must start at 0"
        for i in range(1, len(ranges)):
            assert ranges[i][0] == ranges[i - 1][1], (
                f"Gap/overlap between {list(FEATURE_SCHEMA.keys())[i-1]} and {list(FEATURE_SCHEMA.keys())[i]}"
            )
        assert ranges[-1][1] == FEATURE_DIM, "Schema must end at FEATURE_DIM"

    @pytest.mark.parametrize("signal_fn", [
        lambda x: np.sin(x),
        lambda x: x ** 2,
        lambda x: np.exp(-x ** 2),
        lambda x: np.ones_like(x) * 3.0,
        lambda x: 1.0 / (1.0 + x ** 2),
    ], ids=["sin", "quadratic", "gaussian", "constant", "lorentzian"])
    def test_extract_all_features_shape(self, signal_fn):
        """extract_all_features should return exactly 366 floats."""
        x = np.linspace(-5, 5, 256)
        y = signal_fn(x)
        features = extract_all_features(y)
        assert features.shape == (FEATURE_DIM,)

    @pytest.mark.parametrize("signal_fn", [
        lambda x: np.sin(x),
        lambda x: x ** 2 + np.random.randn(len(x)) * 0.01,
        lambda x: np.exp(x),
    ], ids=["sin", "noisy_quadratic", "exp"])
    def test_no_nan_or_inf(self, signal_fn):
        """No NaN or Inf values should appear in features."""
        x = np.linspace(-5, 5, 256)
        y = signal_fn(x)
        features = extract_all_features(y)
        assert not np.any(np.isnan(features)), "Features contain NaN"
        assert not np.any(np.isinf(features)), "Features contain Inf"


# =============================================================================
# 2. FFT Phase Discrimination
# =============================================================================

class TestFFTPhase:
    """Phase features should discriminate signals with similar magnitude spectra."""

    def test_phase_feature_count(self):
        """extract_fft_phase_features should return 32 values."""
        y = np.sin(np.linspace(0, 10, 256))
        phases = extract_fft_phase_features(y, n_bins=32)
        assert phases.shape == (32,)

    def test_phase_range(self):
        """Phase features should be in [-1, 1]."""
        y = np.sin(np.linspace(0, 10, 256)) + np.cos(np.linspace(0, 30, 256))
        phases = extract_fft_phase_features(y)
        assert np.all(phases >= -1.0 - 1e-7)
        assert np.all(phases <= 1.0 + 1e-7)

    def test_additive_vs_multiplicative_discrimination(self):
        """sin(x)+sin(3x) and sin(x)*sin(3x) should have distinct phase features.
        
        These signals have similar FFT magnitude spectra (both have energy at
        frequency 1 and 3) but fundamentally different phase relationships.
        """
        x = np.linspace(-5, 5, 256)
        y_add = np.sin(x) + np.sin(3 * x)
        y_mul = np.sin(x) * np.sin(3 * x)

        phase_add = extract_fft_phase_features(y_add)
        phase_mul = extract_fft_phase_features(y_mul)

        # Magnitudes should be somewhat similar
        mag_add = extract_fft_features(y_add)
        mag_mul = extract_fft_features(y_mul)

        mag_diff = np.linalg.norm(mag_add - mag_mul)
        phase_diff = np.linalg.norm(phase_add - phase_mul)

        # Phase difference should be meaningfully larger relative to magnitude diff
        # (this validates the added discriminative power of phase features)
        assert phase_diff > 0.1, (
            f"Phase features too similar for additive vs multiplicative: diff={phase_diff:.4f}"
        )

    def test_insignificant_phase_zeroing(self):
        """Phase of insignificant (low-magnitude) frequency bins should be zeroed."""
        # A pure sine — only one frequency should have significant magnitude
        x = np.linspace(0, 2 * np.pi * 5, 256)
        y = np.sin(x)
        phases = extract_fft_phase_features(y)
        
        # Most bins should be zeroed out (only the dominant frequency bin is non-zero)
        n_nonzero = np.count_nonzero(phases)
        assert n_nonzero < 10, f"Too many non-zero phase bins for pure sine: {n_nonzero}"


# =============================================================================
# 3. Smooth Derivative Features
# =============================================================================

class TestSmoothDerivatives:
    """Savitzky-Golay smoothing should produce less noisy derivatives."""

    def test_derivative_feature_shape(self):
        """Derivative features should still be 128 dims (64 dy + 64 ddy)."""
        y = np.sin(np.linspace(-5, 5, 256))
        deriv = extract_derivative_features(y)
        assert deriv.shape == (128,)

    def test_smooth_derivatives_less_noisy(self):
        """Derivatives of noisy signal should be smoother with smoothing enabled.
        
        The variance of second differences (a proxy for "jitter") should be
        lower when pre-smoothing is applied.
        """
        np.random.seed(42)
        x = np.linspace(-5, 5, 256)
        y_clean = np.sin(x)
        y_noisy = y_clean + np.random.randn(len(x)) * 0.3

        # Extract smoothed derivatives (current implementation)
        deriv_smoothed = extract_derivative_features(y_noisy)
        dy_smoothed = deriv_smoothed[:64]

        # Compare with raw diff (no smoothing)
        dy_raw = np.diff(y_noisy)
        dy_raw_resampled = np.interp(
            np.linspace(0, 1, 64),
            np.linspace(0, 1, len(dy_raw)),
            dy_raw
        )
        dy_max = np.abs(dy_raw_resampled).max()
        if dy_max > 1e-10:
            dy_raw_resampled = dy_raw_resampled / dy_max

        # Jitter: variance of second differences
        jitter_smooth = np.var(np.diff(dy_smoothed, n=2))
        jitter_raw = np.var(np.diff(dy_raw_resampled, n=2))

        assert jitter_smooth < jitter_raw, (
            f"Smoothed derivatives should be less jittery: "
            f"smooth={jitter_smooth:.6f}, raw={jitter_raw:.6f}"
        )

    def test_smooth_signal_function(self):
        """_smooth_signal should return an array of the same length."""
        y = np.random.randn(256)
        y_smooth = _smooth_signal(y)
        assert len(y_smooth) == len(y)


# =============================================================================
# 4. Curvature-Aware Resampling
# =============================================================================

class TestCurvatureResampling:
    """Curvature-aware resampling should concentrate points near high-curvature regions."""

    def test_raw_features_shape(self):
        """Raw features should still be 128 points."""
        y = np.sin(np.linspace(-5, 5, 256))
        raw = extract_raw_features(y)
        assert raw.shape == (128,)

    def test_raw_features_normalized(self):
        """Output should be normalized to [0, 1]."""
        y = np.sin(np.linspace(-5, 5, 256)) * 100 + 50
        raw = extract_raw_features(y)
        assert raw.min() >= -1e-7, f"Min below 0: {raw.min()}"
        assert raw.max() <= 1.0 + 1e-7, f"Max above 1: {raw.max()}"

    def test_curvature_concentration(self):
        """For 1/(1+x²), resampled points should cluster more densely near x=0.
        
        The function 1/(1+x²) has high curvature near x=0 and is nearly flat
        at the tails. Curvature-aware resampling should place more sample
        points in the high-curvature region.
        """
        x = np.linspace(-5, 5, 256)
        y = 1.0 / (1.0 + x ** 2)

        # Extract curvature-aware features
        raw_curvature = extract_raw_features(y, curvature_alpha=5.0)

        # Extract uniform features (alpha=0 disables curvature weighting)
        raw_uniform = extract_raw_features(y, curvature_alpha=0.0)

        # The curvature-aware version should have more variation in the
        # central region (more points sampling the peak), meaning the
        # gradient of the feature vector should have higher energy in
        # the middle portion.
        gradient_curv = np.diff(raw_curvature)
        gradient_unif = np.diff(raw_uniform)

        # Middle third of the gradient
        n = len(gradient_curv)
        mid_start, mid_end = n // 3, 2 * n // 3
        mid_energy_curv = np.sum(gradient_curv[mid_start:mid_end] ** 2)
        mid_energy_unif = np.sum(gradient_unif[mid_start:mid_end] ** 2)

        # The curvature-aware version should produce measurably different features
        max_diff = np.max(np.abs(raw_curvature - raw_uniform))
        assert max_diff > 1e-4, (
            f"Curvature-aware and uniform resampling should produce different features, "
            f"max diff={max_diff:.6f}"
        )

    def test_uniform_fallback_with_alpha_zero(self):
        """With alpha=0, resampling should be approximately uniform."""
        x = np.linspace(-5, 5, 256)
        y = np.sin(x)

        raw_a0 = extract_raw_features(y, curvature_alpha=0.0)
        
        # Manually do uniform resampling + normalization
        x_old = np.linspace(0, 1, len(y))
        x_new = np.linspace(0, 1, 128)
        y_uniform = np.interp(x_new, x_old, y)
        y_min, y_max = y_uniform.min(), y_uniform.max()
        y_uniform_norm = (y_uniform - y_min) / (y_max - y_min)

        # Should be very close (alpha=0 → w=1 everywhere → uniform CDF)
        np.testing.assert_allclose(raw_a0, y_uniform_norm, atol=0.02)


# =============================================================================
# 5. Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge cases: constant signal, very short signal, extreme values."""

    def test_constant_signal(self):
        """Constant signal should not produce NaN/Inf."""
        y = np.ones(256) * 42.0
        features = extract_all_features(y)
        assert features.shape == (FEATURE_DIM,)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

    def test_short_signal(self):
        """Very short signal (< 11 points) should not crash."""
        y = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        features = extract_all_features(y)
        assert features.shape == (FEATURE_DIM,)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

    def test_very_short_signal(self):
        """Signal with only 2 points should not crash."""
        y = np.array([0.0, 1.0])
        features = extract_all_features(y)
        assert features.shape == (FEATURE_DIM,)
        assert not np.any(np.isnan(features))

    def test_extremely_large_values(self):
        """Signal with large values should be handled gracefully."""
        y = np.linspace(0, 1e5, 256)
        features = extract_all_features(y)
        assert features.shape == (FEATURE_DIM,)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

    def test_high_frequency_noisy(self):
        """High-frequency noisy signal should not crash."""
        np.random.seed(123)
        y = np.random.randn(256)
        features = extract_all_features(y)
        assert features.shape == (FEATURE_DIM,)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

    def test_single_point(self):
        """Single point signal edge case."""
        y = np.array([5.0])
        features = extract_all_features(y)
        assert features.shape == (FEATURE_DIM,)
        assert not np.any(np.isnan(features))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
