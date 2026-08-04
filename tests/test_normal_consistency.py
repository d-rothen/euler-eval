"""Geometric correctness tests for the depth normal-consistency metric.

The normals are derived from an unprojected point map, so a depth map rendered
from a known plane must yield that plane's analytic normal — under any camera,
for planar and radial depth alike.  An image-space gradient estimator (which
ignores the perspective divide) fails these.
"""

import numpy as np
import pytest

from euler_eval.data import align_intrinsics_to_prediction
from euler_eval.metrics.normal_consistency import (
    aggregate_normal_consistency,
    compute_normal_angles,
    depth_to_normals,
    depth_to_points,
    resolve_intrinsics,
)
from euler_eval.metrics.utils import convert_planar_to_radial

SHAPE = (48, 64)


def _camera(shape=SHAPE, focal=None):
    """Pinhole camera matrix with the principal point at the image centre."""
    height, width = shape
    focal = float(width) if focal is None else float(focal)
    return np.array(
        [
            [focal, 0.0, (width - 1) / 2.0],
            [0.0, focal, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _plane_depth(K, normal, distance, shape=SHAPE, radial=False):
    """Render the depth map of the plane ``n · P = distance`` seen by *K*."""
    height, width = shape
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    directions = np.stack([x, y, np.ones_like(x)], axis=-1)

    n = _unit(normal)
    z = distance / (directions @ n)
    if radial:
        return z * np.linalg.norm(directions, axis=-1)
    return z


def _interior(shape=SHAPE):
    mask = np.zeros(shape, dtype=bool)
    mask[2:-2, 2:-2] = True
    return mask


def _dots_with(normals, normal):
    return np.einsum("ijk,k->ij", normals, _unit(normal))


class TestUnprojection:
    def test_fronto_parallel_plane_has_axis_aligned_normals(self):
        K = _camera()
        depth = np.full(SHAPE, 7.5, dtype=np.float64)

        normals = depth_to_normals(depth, intrinsics=K)

        dots = _dots_with(normals, [0.0, 0.0, 1.0])[_interior()]
        assert dots.min() > 0.9999

    def test_slanted_plane_recovers_analytic_normal(self):
        K = _camera()
        normal = [0.35, -0.2, 1.0]
        depth = _plane_depth(K, normal, distance=12.0)

        normals = depth_to_normals(depth, intrinsics=K)

        dots = _dots_with(normals, normal)[_interior()]
        assert dots.min() > 0.999

    def test_radial_depth_matches_planar_depth(self):
        K = _camera()
        normal = [0.25, 0.3, 1.0]
        planar = _plane_depth(K, normal, distance=9.0)
        radial = _plane_depth(K, normal, distance=9.0, radial=True)

        # The rendered radial map must match the shared planar→radial helper.
        intrinsics = {
            "fx": K[0, 0],
            "fy": K[1, 1],
            "cx": K[0, 2],
            "cy": K[1, 2],
        }
        np.testing.assert_allclose(
            radial, convert_planar_to_radial(planar, intrinsics), rtol=1e-6
        )

        planar_normals = depth_to_normals(planar, intrinsics=K)
        radial_normals = depth_to_normals(radial, intrinsics=K, is_radial=True)

        np.testing.assert_allclose(planar_normals, radial_normals, atol=1e-5)

    def test_radial_depth_ignored_flag_distorts_geometry(self):
        """Mislabelling radial depth as planar must not silently agree."""
        K = _camera()
        normal = [0.4, 0.0, 1.0]
        radial = _plane_depth(K, normal, distance=9.0, radial=True)

        correct = depth_to_normals(radial, intrinsics=K, is_radial=True)
        mislabelled = depth_to_normals(radial, intrinsics=K, is_radial=False)

        interior = _interior()
        assert _dots_with(correct, normal)[interior].min() > 0.999
        assert _dots_with(mislabelled, normal)[interior].min() < 0.999

    def test_intrinsics_change_the_reconstructed_surface(self):
        K = _camera()
        normal = [0.35, -0.2, 1.0]
        depth = _plane_depth(K, normal, distance=12.0)

        wrong_K = _camera(focal=2 * SHAPE[1])
        normals = depth_to_normals(depth, intrinsics=wrong_K)

        dots = np.clip(_dots_with(normals, normal)[_interior()], -1.0, 1.0)
        angles = np.degrees(np.arccos(dots))
        assert angles.mean() > 1.0

    def test_points_round_trip_through_intrinsics(self):
        K = _camera()
        depth = _plane_depth(K, [0.1, 0.2, 1.0], distance=6.0)

        points = depth_to_points(depth, intrinsics=K, dtype=np.float64)

        # Re-projecting the point map must return the original pixel grid.
        u = points[..., 0] / points[..., 2] * K[0, 0] + K[0, 2]
        v = points[..., 1] / points[..., 2] * K[1, 1] + K[1, 2]
        expected_u, expected_v = np.meshgrid(
            np.arange(SHAPE[1], dtype=np.float64),
            np.arange(SHAPE[0], dtype=np.float64),
        )
        np.testing.assert_allclose(u, expected_u, atol=1e-6)
        np.testing.assert_allclose(v, expected_v, atol=1e-6)
        np.testing.assert_allclose(points[..., 2], depth, rtol=1e-9)

    def test_default_precision_matches_float64(self):
        """float32 is the default for speed; it must not move the result."""
        K = _camera()
        depth_gt = _plane_depth(K, [0.2, 0.1, 1.0], distance=8.0)
        depth_pred = _plane_depth(K, [0.24, 0.05, 1.0], distance=8.2)

        angles32 = compute_normal_angles(depth_pred, depth_gt, intrinsics=K)
        angles64 = compute_normal_angles(
            depth_pred, depth_gt, intrinsics=K, dtype=np.float64
        )

        assert angles32.mean() == pytest.approx(angles64.mean(), abs=1e-3)

    def test_rejects_non_2d_depth(self):
        with pytest.raises(ValueError, match="H, W"):
            depth_to_points(np.zeros((4, 4, 3)))


class TestResolveIntrinsics:
    def test_camera_matrix_is_used_as_given(self):
        K = _camera(focal=123.0)
        values, source = resolve_intrinsics(SHAPE, K)

        assert source == "sample"
        assert values["fx"] == pytest.approx(123.0)
        assert values["cx"] == pytest.approx((SHAPE[1] - 1) / 2.0)

    def test_mapping_and_scalar_are_accepted(self):
        mapping, source = resolve_intrinsics(SHAPE, {"fx": 80.0, "fy": 80.0})
        assert source == "sample"
        assert mapping["fy"] == pytest.approx(80.0)
        # A mapping without a principal point centres it.
        assert mapping["cy"] == pytest.approx((SHAPE[0] - 1) / 2.0)

        scalar, source = resolve_intrinsics(SHAPE, 90.0)
        assert source == "sample"
        assert (scalar["fx"], scalar["fy"]) == (90.0, 90.0)

    def test_missing_intrinsics_fall_back_to_assumed_camera(self):
        values, source = resolve_intrinsics(SHAPE, None)

        assert source == "assumed"
        assert values["fx"] == pytest.approx(float(SHAPE[1]))

    @pytest.mark.parametrize(
        "bad",
        [
            np.diag([0.0, 0.0, 1.0]),  # zero focal length
            np.full((3, 3), np.nan),  # non-finite
            np.zeros((2, 2)),  # wrong shape
            {"fx": None, "fy": None},  # unusable mapping
        ],
    )
    def test_unusable_intrinsics_fall_back(self, bad):
        _, source = resolve_intrinsics(SHAPE, bad)
        assert source == "assumed"


class TestNormalAngles:
    def test_identical_depth_maps_have_zero_angular_error(self):
        K = _camera()
        depth = _plane_depth(K, [0.2, 0.1, 1.0], distance=8.0)

        angles = compute_normal_angles(depth, depth, intrinsics=K)

        assert angles.size > 0
        assert angles.max() < 1e-3

    def test_two_planes_report_their_true_angle(self):
        K = _camera()
        gt_normal = [0.0, 0.0, 1.0]
        pred_normal = [0.3, 0.0, 1.0]
        depth_gt = _plane_depth(K, gt_normal, distance=10.0)
        depth_pred = _plane_depth(K, pred_normal, distance=10.0)

        expected = np.degrees(
            np.arccos(np.dot(_unit(gt_normal), _unit(pred_normal)))
        )

        angles, metadata = compute_normal_angles(
            depth_pred, depth_gt, _interior(), intrinsics=K, return_metadata=True
        )

        assert angles.mean() == pytest.approx(expected, abs=0.1)
        assert metadata["mean_angle"] == pytest.approx(angles.mean())
        assert metadata["intrinsics_source"] == "sample"
        assert metadata["focal_length_used"] == pytest.approx(K[0, 0])
        assert metadata["depth_is_radial"] is False

    def test_metadata_flags_an_assumed_camera(self):
        depth = np.full(SHAPE, 5.0, dtype=np.float64)

        _, metadata = compute_normal_angles(depth, depth, return_metadata=True)

        assert metadata["intrinsics_source"] == "assumed"
        assert metadata["valid_pixels_after_erosion"] < metadata[
            "valid_pixels_before_erosion"
        ]

    def test_empty_valid_mask_returns_no_angles(self):
        depth = np.full(SHAPE, 5.0, dtype=np.float64)
        empty = np.zeros(SHAPE, dtype=bool)

        angles, metadata = compute_normal_angles(
            depth, depth, empty, return_metadata=True
        )

        assert angles.size == 0
        assert metadata["mean_angle"] is None

    def test_aggregate_matches_pooled_angles(self):
        first = np.array([0.0, 10.0], dtype=np.float32)
        second = np.array([20.0, 40.0], dtype=np.float32)

        summary = aggregate_normal_consistency([first, np.array([]), second])

        assert summary["mean_angle"] == pytest.approx(17.5)
        assert summary["median_angle"] == pytest.approx(15.0)
        assert summary["percent_below_11_25"] == pytest.approx(50.0)
        assert summary["percent_below_30"] == pytest.approx(75.0)


class TestAlignIntrinsicsToPrediction:
    def test_matching_shapes_are_unchanged(self):
        K = _camera()
        np.testing.assert_array_equal(
            align_intrinsics_to_prediction(K, SHAPE, SHAPE), K
        )

    def test_vae_crop_keeps_the_principal_point(self):
        K = _camera(shape=(69, 101))
        # A top-left crop to the next multiple of 8 keeps the pixel origin.
        aligned = align_intrinsics_to_prediction(K, (69, 101), (64, 96))
        np.testing.assert_allclose(aligned, K)

    def test_resize_scales_focal_length_and_principal_point(self):
        K = _camera(shape=(48, 64), focal=64.0)

        aligned = align_intrinsics_to_prediction(K, (48, 64), (24, 32))

        assert aligned[0, 0] == pytest.approx(32.0)
        assert aligned[1, 1] == pytest.approx(32.0)
        # cx = 31.5 in a 64-wide image -> 15.5 in a 32-wide one.
        assert aligned[0, 2] == pytest.approx(15.5)
        assert aligned[1, 2] == pytest.approx(11.5)

    def test_rescaled_intrinsics_recover_normals_after_resize(self):
        """A resized depth map must still reconstruct the same surface."""
        import cv2

        K = _camera(shape=(96, 128), focal=128.0)
        normal = [0.3, -0.15, 1.0]
        depth = _plane_depth(K, normal, distance=11.0, shape=(96, 128))

        resized = cv2.resize(depth, (64, 48), interpolation=cv2.INTER_AREA)
        resized_K = align_intrinsics_to_prediction(K, (96, 128), (48, 64))

        normals = depth_to_normals(resized, intrinsics=resized_K)
        dots = _dots_with(normals, normal)[_interior((48, 64))]
        assert dots.min() > 0.999

        # The unscaled matrix reconstructs a different (wrong) surface.
        wrong = depth_to_normals(resized, intrinsics=K)
        assert _dots_with(wrong, normal)[_interior((48, 64))].min() < 0.999
