"""Tests for the ρ_A metric and rays evaluation pipeline."""

import numpy as np
import pytest

from euler_eval.metrics.rho_a import (
    classify_fov_domain,
    compute_angular_errors,
    compute_rho_a,
    aggregate_rho_a,
    aggregate_angular_errors,
    get_threshold_for_domain,
    FOV_THRESHOLDS,
)
from euler_eval.data import to_numpy_directions


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_intrinsics(fx, fy, cx, cy):
    """Build a (3,3) intrinsics matrix."""
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def _make_unit_directions(height, width, K):
    """Generate a GT ray direction map from intrinsics."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(width, dtype=np.float32),
                       np.arange(height, dtype=np.float32))
    dirs = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
    norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs / norms


# ---------------------------------------------------------------------------
# classify_fov_domain
# ---------------------------------------------------------------------------


class TestClassifyFovDomain:
    def test_narrow_fov(self):
        # 640x480 with large focal length → small FoV
        K = _make_intrinsics(1000.0, 1000.0, 320.0, 240.0)
        assert classify_fov_domain(K, 480, 640) == "sfov"

    def test_wide_fov(self):
        # 640x480 with moderate focal → large FoV
        K = _make_intrinsics(200.0, 200.0, 320.0, 240.0)
        assert classify_fov_domain(K, 480, 640) == "lfov"

    def test_very_wide_fov(self):
        # Tiny focal length → panoramic
        K = _make_intrinsics(50.0, 50.0, 320.0, 240.0)
        assert classify_fov_domain(K, 480, 640) == "pano"


# ---------------------------------------------------------------------------
# get_threshold_for_domain
# ---------------------------------------------------------------------------


class TestGetThreshold:
    def test_known_domains(self):
        assert get_threshold_for_domain("sfov") == 15.0
        assert get_threshold_for_domain("lfov") == 20.0
        assert get_threshold_for_domain("pano") == 30.0

    def test_case_insensitive(self):
        assert get_threshold_for_domain("SFOV") == 15.0
        assert get_threshold_for_domain("Pano") == 30.0

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown FoV domain"):
            get_threshold_for_domain("ultra_wide")


# ---------------------------------------------------------------------------
# compute_angular_errors
# ---------------------------------------------------------------------------


class TestComputeAngularErrors:
    def test_identical_directions(self):
        """Identical GT and pred → near-zero angular error."""
        K = _make_intrinsics(500.0, 500.0, 160.0, 120.0)
        dirs = _make_unit_directions(240, 320, K)
        errors = compute_angular_errors(dirs, dirs)
        assert len(errors) > 0
        # Float32 dot-product precision can cause small nonzero arccos values
        np.testing.assert_allclose(errors, 0.0, atol=0.05)

    def test_orthogonal_directions(self):
        """90° offset should give ~90° error."""
        gt = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)  # (1,1,3)
        pred = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)
        errors = compute_angular_errors(pred, gt)
        np.testing.assert_allclose(errors, 90.0, atol=1e-4)

    def test_opposite_directions(self):
        """180° offset."""
        gt = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
        pred = np.array([[[0.0, 0.0, -1.0]]], dtype=np.float32)
        errors = compute_angular_errors(pred, gt)
        np.testing.assert_allclose(errors, 180.0, atol=1e-4)

    def test_with_valid_mask(self):
        """Only masked pixels should be in output."""
        dirs = np.ones((4, 4, 3), dtype=np.float32)
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = True
        mask[1, 1] = True
        errors = compute_angular_errors(dirs, dirs, valid_mask=mask)
        assert len(errors) == 2

    def test_return_metadata(self):
        gt = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
        pred = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
        errors, meta = compute_angular_errors(pred, gt, return_metadata=True)
        assert meta["valid_pixel_count"] == 1
        assert meta["mean_angular_error"] is not None
        np.testing.assert_allclose(meta["mean_angular_error"], 0.0, atol=1e-5)

    def test_empty_valid_mask(self):
        dirs = np.ones((2, 2, 3), dtype=np.float32)
        mask = np.zeros((2, 2), dtype=bool)
        errors, meta = compute_angular_errors(dirs, dirs, valid_mask=mask,
                                              return_metadata=True)
        assert len(errors) == 0
        assert meta["valid_pixel_count"] == 0
        assert meta["mean_angular_error"] is None

    def test_non_unit_vectors_are_normalized(self):
        """Vectors that aren't unit should be normalized internally."""
        gt = np.array([[[2.0, 0.0, 0.0]]], dtype=np.float32)
        pred = np.array([[[0.0, 3.0, 0.0]]], dtype=np.float32)
        errors = compute_angular_errors(pred, gt)
        np.testing.assert_allclose(errors, 90.0, atol=1e-4)


# ---------------------------------------------------------------------------
# compute_rho_a
# ---------------------------------------------------------------------------


class TestComputeRhoA:
    def test_perfect_prediction(self):
        """Zero angular errors → ρ_A = 1.0."""
        errors = np.zeros(100)
        rho_a = compute_rho_a(errors, threshold_deg=15.0)
        np.testing.assert_allclose(rho_a, 1.0, atol=1e-6)

    def test_all_errors_above_threshold(self):
        """All errors above threshold → ρ_A = 0.0."""
        errors = np.full(100, 20.0)
        rho_a = compute_rho_a(errors, threshold_deg=15.0)
        np.testing.assert_allclose(rho_a, 0.0, atol=1e-6)

    def test_uniform_distribution(self):
        """Uniform errors [0, threshold] → AUC should be ~0.5."""
        errors = np.linspace(0, 15.0, 10000)
        rho_a = compute_rho_a(errors, threshold_deg=15.0)
        # For uniform [0, T], accuracy(θ) = θ/T, AUC = T/2/T = 0.5
        np.testing.assert_allclose(rho_a, 0.5, atol=0.02)

    def test_empty_errors(self):
        rho_a = compute_rho_a(np.array([]), threshold_deg=15.0)
        assert np.isnan(rho_a)

    def test_different_thresholds(self):
        """Higher threshold → higher ρ_A for same errors."""
        errors = np.random.uniform(0, 10, 1000)
        rho_sfov = compute_rho_a(errors, threshold_deg=15.0)
        rho_pano = compute_rho_a(errors, threshold_deg=30.0)
        # With errors in [0, 10], higher threshold includes more area
        assert rho_pano >= rho_sfov

    def test_value_range(self):
        """ρ_A should be in [0, 1]."""
        errors = np.random.uniform(0, 25, 500)
        for thresh in [15.0, 20.0, 30.0]:
            rho_a = compute_rho_a(errors, thresh)
            assert 0.0 <= rho_a <= 1.0


# ---------------------------------------------------------------------------
# aggregate_rho_a
# ---------------------------------------------------------------------------


class TestAggregateRhoA:
    def test_basic(self):
        result = aggregate_rho_a([0.8, 0.9, 0.85])
        np.testing.assert_allclose(result["mean"], 0.85, atol=1e-6)
        assert "median" in result
        assert "std" in result

    def test_with_nan(self):
        result = aggregate_rho_a([0.8, float("nan"), 0.9])
        np.testing.assert_allclose(result["mean"], 0.85, atol=1e-6)

    def test_all_nan(self):
        result = aggregate_rho_a([float("nan"), float("nan")])
        assert np.isnan(result["mean"])

    def test_empty(self):
        result = aggregate_rho_a([])
        assert np.isnan(result["mean"])


# ---------------------------------------------------------------------------
# aggregate_angular_errors
# ---------------------------------------------------------------------------


class TestAggregateAngularErrors:
    def test_basic(self):
        a1 = np.array([5.0, 10.0, 15.0])
        a2 = np.array([1.0, 2.0])
        result = aggregate_angular_errors([a1, a2])
        assert "mean_angle" in result
        assert "median_angle" in result
        assert "percent_below_5" in result
        assert "percent_below_15" in result
        assert "percent_below_30" in result

    def test_empty_arrays(self):
        result = aggregate_angular_errors([np.array([]), np.array([])])
        assert np.isnan(result["mean_angle"])


# ---------------------------------------------------------------------------
# to_numpy_directions
# ---------------------------------------------------------------------------


class TestToNumpyDirections:
    def test_hwc_numpy(self):
        data = np.random.randn(10, 20, 3).astype(np.float32)
        result = to_numpy_directions(data)
        assert result.shape == (10, 20, 3)
        assert result.dtype == np.float32

    def test_chw_torch(self):
        import torch
        data = torch.randn(3, 10, 20)
        result = to_numpy_directions(data)
        assert result.shape == (10, 20, 3)

    def test_batch_dim_removed(self):
        import torch
        data = torch.randn(1, 3, 10, 20)
        result = to_numpy_directions(data)
        assert result.shape == (10, 20, 3)

    def test_invalid_shape(self):
        data = np.random.randn(10, 20).astype(np.float32)
        with pytest.raises(ValueError):
            to_numpy_directions(data)


# ---------------------------------------------------------------------------
# End-to-end: pinhole GT → rho_a
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_perfect_pinhole_prediction(self):
        """Model perfectly predicts rays from a pinhole camera → ρ_A ≈ 1."""
        K = _make_intrinsics(500.0, 500.0, 160.0, 120.0)
        gt = _make_unit_directions(240, 320, K)
        pred = gt.copy()

        errors = compute_angular_errors(pred, gt)
        rho_a = compute_rho_a(errors, threshold_deg=15.0)
        np.testing.assert_allclose(rho_a, 1.0, atol=1e-3)

    def test_noisy_prediction(self):
        """Small noise → high but not perfect ρ_A."""
        K = _make_intrinsics(500.0, 500.0, 160.0, 120.0)
        gt = _make_unit_directions(60, 80, K)
        noise = np.random.randn(*gt.shape).astype(np.float32) * 0.01
        pred = gt + noise
        # Renormalize
        pred /= np.linalg.norm(pred, axis=-1, keepdims=True)

        errors = compute_angular_errors(pred, gt)
        rho_a = compute_rho_a(errors, threshold_deg=15.0)
        assert rho_a > 0.9  # should be very close to 1 with small noise
        assert rho_a < 1.0

    def test_large_error_prediction(self):
        """Random directions → low ρ_A."""
        gt = np.array([[[0, 0, 1]]] * 100, dtype=np.float32).reshape(10, 10, 3)
        pred = np.random.randn(10, 10, 3).astype(np.float32)
        pred /= np.linalg.norm(pred, axis=-1, keepdims=True)

        errors = compute_angular_errors(pred, gt)
        rho_a = compute_rho_a(errors, threshold_deg=15.0)
        assert rho_a < 0.5  # random directions shouldn't score well
