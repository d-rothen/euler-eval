"""Tests for the points_3d (per-pixel 3D point map) modality.

Covers the data helpers (converters, gauge alignment, error decomposition,
unprojection), the metric modules (distance / decomposition / geometry /
cloud), the ``evaluate_points_3d_samples`` pipeline, and the CLI namespace
wiring.
"""

import numpy as np
import pytest

from euler_eval.data import (
    apply_point_transform,
    build_points_3d_eval_dataset,
    decompose_point_errors,
    fit_scale_transform,
    fit_similarity_transform,
    to_numpy_points_3d,
    unproject_depth_to_points,
)
from euler_eval.evaluate import evaluate_points_3d_samples
from euler_eval.metrics import (
    compute_cloud_distance_metrics,
    compute_decomposition_metrics,
    compute_fscore_auc,
    compute_point_edge_f1,
    compute_point_error_metrics,
    compute_point_normal_angles,
)

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_intrinsics(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def _gt_pointmap(h=40, w=60, seed=0):
    rng = np.random.RandomState(seed)
    K = _make_intrinsics(400.0, 400.0, w / 2.0, h / 2.0)
    depth = rng.rand(h, w).astype(np.float32) * 8.0 + 2.0
    return unproject_depth_to_points(depth, K, depth_is_radial=False), K


class _OnePointMapDataset:
    """Minimal MultiModalDataset stand-in yielding one points_3d pair."""

    def __init__(self, pred, gt, K=None, full_id="/scene/000000"):
        self.pred = pred
        self.gt = gt
        self.K = K
        self.full_id = full_id

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        sample = {
            "id": self.full_id.strip("/").split("/")[-1],
            "full_id": self.full_id,
            "gt": self.gt,
            "pred": self.pred,
        }
        if self.K is not None:
            sample["calibration"] = {"K": self.K}
        return sample


# ---------------------------------------------------------------------------
# Converters / unprojection
# ---------------------------------------------------------------------------


class TestConverters:
    def test_hwc_numpy(self):
        data = np.random.randn(10, 20, 3).astype(np.float32)
        out = to_numpy_points_3d(data)
        assert out.shape == (10, 20, 3)

    def test_chw_torch(self):
        import torch
        out = to_numpy_points_3d(torch.randn(3, 8, 9))
        assert out.shape == (8, 9, 3)

    def test_batch_dim_removed(self):
        import torch
        out = to_numpy_points_3d(torch.randn(1, 3, 8, 9))
        assert out.shape == (8, 9, 3)

    def test_unproject_planar(self):
        K = _make_intrinsics(500, 500, 160, 120)
        depth = np.full((240, 320), 4.0, np.float32)
        P = unproject_depth_to_points(depth, K, depth_is_radial=False)
        # planar Z is exactly the depth value
        np.testing.assert_allclose(P[..., 2], 4.0, atol=1e-5)
        # principal-point pixel (v=cy, u=cx) maps to (0, 0, Z)
        np.testing.assert_allclose(P[120, 160], [0, 0, 4.0], atol=1e-4)

    def test_unproject_radial_shorter_than_planar(self):
        K = _make_intrinsics(500, 500, 160, 120)
        depth = np.full((240, 320), 4.0, np.float32)
        P = unproject_depth_to_points(depth, K, depth_is_radial=True)
        # off-axis radial depth yields planar Z < radial value
        assert P[0, 0, 2] < 4.0


# ---------------------------------------------------------------------------
# Gauge alignment
# ---------------------------------------------------------------------------


class TestAlignment:
    def test_fit_scale_recovers_scale(self):
        gt, _ = _gt_pointmap()
        pts = gt.reshape(-1, 3)
        s = fit_scale_transform(pts * 3.0, pts)
        np.testing.assert_allclose(s, 1 / 3.0, atol=1e-4)

    def test_similarity_recovers_transform(self):
        gt, _ = _gt_pointmap()
        pts = gt.reshape(-1, 3)
        R = _rot_z(0.3)
        pred = apply_point_transform(pts, 2.0, R, np.array([1.0, -2.0, 0.5], np.float32))
        s, R_fit, t_fit = fit_similarity_transform(pred, pts)
        recovered = apply_point_transform(pred, s, R_fit, t_fit)
        rel = np.linalg.norm(recovered - pts) / np.linalg.norm(pts)
        assert rel < 1e-4

    def test_similarity_rigid_no_scale(self):
        gt, _ = _gt_pointmap()
        pts = gt.reshape(-1, 3)
        pred = apply_point_transform(pts, 1.0, _rot_z(0.2), np.zeros(3, np.float32))
        s, _, _ = fit_similarity_transform(pred, pts, with_scale=False)
        assert abs(s - 1.0) < 1e-6

    def test_similarity_degenerate_returns_identity(self):
        pts = np.zeros((2, 3), np.float32)
        s, R, t = fit_similarity_transform(pts, pts)
        assert s == 1.0
        np.testing.assert_array_equal(R, np.eye(3))


# ---------------------------------------------------------------------------
# Error decomposition
# ---------------------------------------------------------------------------


class TestDecomposition:
    def test_pure_radial_error(self):
        # A point straight ahead; push it further along the ray (z).
        gt = np.array([[0.0, 0.0, 5.0]], np.float32)
        pred = np.array([[0.0, 0.0, 5.5]], np.float32)
        d = decompose_point_errors(pred, gt)
        np.testing.assert_allclose(abs(d["radial"][0]), 0.5, atol=1e-5)
        np.testing.assert_allclose(d["lateral"][0], 0.0, atol=1e-5)

    def test_pure_lateral_error(self):
        gt = np.array([[0.0, 0.0, 5.0]], np.float32)
        pred = np.array([[0.3, 0.0, 5.0]], np.float32)  # shift perpendicular to ray
        d = decompose_point_errors(pred, gt)
        np.testing.assert_allclose(d["radial"][0], 0.0, atol=1e-5)
        np.testing.assert_allclose(d["lateral"][0], 0.3, atol=1e-5)

    def test_decomposition_fraction_extremes(self):
        radial = np.array([0.0, 0.0], np.float32)
        lateral = np.array([0.5, 0.5], np.float32)
        m = compute_decomposition_metrics(radial, lateral)
        assert m["lateral_fraction"] == 1.0
        m2 = compute_decomposition_metrics(lateral, radial)
        assert m2["lateral_fraction"] == 0.0


# ---------------------------------------------------------------------------
# Point-error metrics
# ---------------------------------------------------------------------------


class TestPointError:
    def test_perfect(self):
        e = np.zeros(100, np.float32)
        m = compute_point_error_metrics(e, e)
        assert m["mae3d"] == 0.0
        assert m["acc_0_05"] == 100.0
        assert m["acc_1"] == 100.0

    def test_threshold_counts(self):
        e = np.array([0.04, 0.2, 0.4], np.float32)
        rel = e.copy()
        m = compute_point_error_metrics(e, rel)
        # only the 0.04 sample is < 0.05
        np.testing.assert_allclose(m["acc_0_05"], 100.0 / 3.0, atol=1e-4)
        # 0.04 and 0.2 are < 0.25
        np.testing.assert_allclose(m["acc_0_25"], 200.0 / 3.0, atol=1e-4)

    def test_empty(self):
        assert compute_point_error_metrics(np.array([]), np.array([])) is None


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class TestGeometry:
    def test_normals_identical_zero_angle(self):
        gt, _ = _gt_pointmap(h=30, w=40)
        mask = np.ones(gt.shape[:2], bool)
        angles, meta = compute_point_normal_angles(gt, gt, mask, return_metadata=True)
        assert angles.size > 0
        assert meta["mean_angle"] < 0.5

    def test_edge_f1_matching_discontinuity(self):
        # Frontoparallel plane with a Z step halfway across -> a vertical edge.
        h, w = 20, 30
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = np.where(u < w // 2, 3.0, 6.0).astype(np.float32)
        P = np.stack([u.astype(np.float32) * 0.01, v.astype(np.float32) * 0.01, z], -1)
        mask = np.ones((h, w), bool)
        res = compute_point_edge_f1(P, P, mask)
        assert res["f1"] == 1.0
        assert res["num_gt_edges"] > 0

    def test_edge_f1_missing_prediction(self):
        h, w = 20, 30
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = np.where(u < w // 2, 3.0, 6.0).astype(np.float32)
        gt = np.stack([u * 0.01, v * 0.01, z], -1).astype(np.float32)
        flat = np.stack([u * 0.01, v * 0.01, np.full_like(z, 4.0)], -1).astype(np.float32)
        mask = np.ones((h, w), bool)
        res = compute_point_edge_f1(flat, gt, mask)
        assert res["recall"] == 0.0


# ---------------------------------------------------------------------------
# Cloud distance
# ---------------------------------------------------------------------------


class TestCloud:
    def test_fscore_auc_extremes(self):
        perfect = np.zeros(4, dtype=np.float64)
        missed = np.full(4, 2.0, dtype=np.float64)
        assert compute_fscore_auc(perfect, perfect, 1.0) == 100.0
        assert compute_fscore_auc(missed, missed, 1.0) == 0.0

    def test_perfect_clouds(self):
        gt, _ = _gt_pointmap()
        pts = gt.reshape(-1, 3)
        m = compute_cloud_distance_metrics(pts, pts)
        assert m["chamfer"]["distance"] < 1e-6
        assert m["fscore"]["tau_0_1"]["f1"] == 1.0

    def test_cloud_f_a_uses_requested_depth_range(self):
        gt, _ = _gt_pointmap()
        pts = gt.reshape(-1, 3)
        m = compute_cloud_distance_metrics(
            pts,
            pts,
            fscore_auc_max_threshold=0.5,
        )
        assert m["f_a"] == 100.0

    def test_shifted_cloud(self):
        gt, _ = _gt_pointmap()
        pts = gt.reshape(-1, 3)
        shifted = pts + np.array([1.0, 0, 0], np.float32)
        m = compute_cloud_distance_metrics(pts, shifted)
        assert m["chamfer"]["distance"] > 0.1

    def test_subsample_cap(self):
        gt, _ = _gt_pointmap(h=60, w=80)
        pts = gt.reshape(-1, 3)
        # cap below the cloud size still returns finite metrics
        m = compute_cloud_distance_metrics(pts, pts, max_points=200)
        assert m is not None and np.isfinite(m["chamfer"]["distance"])

    def test_empty(self):
        assert compute_cloud_distance_metrics(np.zeros((0, 3)), np.ones((3, 3))) is None


# ---------------------------------------------------------------------------
# End-to-end evaluation
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_perfect_metric_prediction_native_only(self):
        gt, K = _gt_pointmap()
        result = evaluate_points_3d_samples(
            _OnePointMapDataset(gt.copy(), gt, K), num_workers=0, alignment_mode="none"
        )
        assert result["points_3d_metric"] is None
        assert result["space_info"]["emitted_spaces"] == ["native"]
        pe = result["points_3d"]["point_error"]["image_mean"]
        assert pe["mae3d"] < 1e-4
        assert pe["acc_0_05"] == 100.0
        assert result["points_3d"]["error_decomposition"]["rho_a"]["mean"] > 0.95
        assert result["dataset_info"]["fov_domain"] == "sfov"
        assert result["dataset_info"]["f_a_max_threshold"] == pytest.approx(
            result["dataset_info"]["max_depth"] / 20.0
        )
        assert result["points_3d"]["cloud_distance"]["f_a"] == 100.0

    def test_similarity_recovers_relative_prediction(self):
        gt, K = _gt_pointmap()
        pred = apply_point_transform(
            gt, 0.5, _rot_z(0.25), np.array([1.0, 2.0, -0.5], np.float32)
        )
        result = evaluate_points_3d_samples(
            _OnePointMapDataset(pred, gt, K),
            num_workers=0,
            alignment_mode="similarity",
        )
        assert result["space_info"]["emitted_spaces"] == ["native", "metric"]
        native_mae = result["points_3d_native"]["point_error"]["image_mean"]["mae3d"]
        metric_mae = result["points_3d_metric"]["point_error"]["image_mean"]["mae3d"]
        assert metric_mae < 1e-3
        assert native_mae > metric_mae
        assert "f_a" in result["points_3d_native"]["cloud_distance"]
        assert result["points_3d_metric"]["cloud_distance"]["f_a"] == pytest.approx(
            100.0, abs=1e-3
        )
        assert result["points_3d"] is result["points_3d_metric"]

    def test_auto_with_relative_hint_aligns(self):
        gt, K = _gt_pointmap()
        pred = apply_point_transform(gt, 0.5, _rot_z(0.1), np.zeros(3, np.float32))
        result = evaluate_points_3d_samples(
            _OnePointMapDataset(pred, gt, K),
            num_workers=0,
            alignment_mode="auto",
            input_space_hint="relative",
        )
        assert result["space_info"]["calibration_applied"] is True
        assert result["space_info"]["metric_space_source"] == "similarity"

    def test_auto_metric_default_no_align(self):
        gt, K = _gt_pointmap()
        result = evaluate_points_3d_samples(
            _OnePointMapDataset(gt.copy(), gt, K), num_workers=0, alignment_mode="auto"
        )
        assert result["space_info"]["calibration_applied"] is False
        assert result["points_3d_metric"] is None

    def test_scale_alignment(self):
        gt, K = _gt_pointmap()
        result = evaluate_points_3d_samples(
            _OnePointMapDataset(gt * 4.0, gt, K), num_workers=0, alignment_mode="scale"
        )
        metric_mae = result["points_3d_metric"]["point_error"]["image_mean"]["mae3d"]
        assert metric_mae < 1e-3

    def test_lateral_shift_is_camera_dominated(self):
        gt, K = _gt_pointmap()
        pred = gt.copy()
        pred[..., 0] += 0.3  # lateral shift in camera X
        result = evaluate_points_3d_samples(
            _OnePointMapDataset(pred, gt, K), num_workers=0, alignment_mode="none"
        )
        dec = result["points_3d"]["error_decomposition"]
        assert dec["lateral_fraction"] > 0.6

    def test_per_file_metrics_structure(self):
        gt, K = _gt_pointmap()
        result = evaluate_points_3d_samples(
            _OnePointMapDataset(gt.copy(), gt, K, full_id="/scene_01/cam0/000000"),
            num_workers=0,
            alignment_mode="none",
        )
        pfm = result["per_file_metrics"]
        node = pfm["children"]["scene_01"]["children"]["cam0"]
        entry = node["files"][0]
        assert entry["id"] == "000000"
        assert "points_3d" in entry["metrics"]
        assert "point_error" in entry["metrics"]["points_3d"]

    def test_empty_dataset_raises(self):
        class _Empty:
            def __len__(self):
                return 0

            def __getitem__(self, i):
                raise IndexError

        with pytest.raises(ValueError):
            evaluate_points_3d_samples(_Empty(), num_workers=0)

    def test_invalid_alignment_mode(self):
        gt, K = _gt_pointmap()
        with pytest.raises(ValueError):
            evaluate_points_3d_samples(
                _OnePointMapDataset(gt, gt, K), num_workers=0, alignment_mode="bogus"
            )


# ---------------------------------------------------------------------------
# CLI namespace wiring
# ---------------------------------------------------------------------------

_METRIC_NAMESPACE_RE = __import__("re").compile(r"^[a-z0-9]+(?:\.[a-z0-9_]+)+$")


def _flatten_numeric(obj, prefix=""):
    out = []
    for key, value in obj.items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append(full)
        elif isinstance(value, dict):
            out.extend(_flatten_numeric(value, full))
    return out


class TestCliNamespace:
    def test_envelope_namespace_conforms(self):
        from euler_eval.cli import (
            _POINTS_3D_EVAL_DESCRIPTIONS,
            _POINTS_3D_METRIC_NAMESPACE,
            _EvalNamespace,
            _points_3d_eval_axes,
            _points_3d_metric_set_envelope,
        )

        ns = _EvalNamespace(
            producer="euler-eval",
            producer_version="1.0.0",
            modalities=("points_3d",),
            axes=_points_3d_eval_axes(),
            descriptions=_POINTS_3D_EVAL_DESCRIPTIONS,
        )
        envelope = _points_3d_metric_set_envelope(ns, metadata={})
        assert envelope["metricNamespace"] == _POINTS_3D_METRIC_NAMESPACE
        assert _METRIC_NAMESPACE_RE.fullmatch(envelope["metricNamespace"])
        assert envelope["axes"]["space"]["position"] == 0
        assert envelope["axes"]["reduction"]["position"] == 2
        assert "mae3d" in envelope["metricDescriptions"]
        assert "f_a" in envelope["metricDescriptions"]

    def test_real_metric_tree_paths_conform(self):
        """Flattened paths of an actual eval result live under points3d.eval."""
        from euler_eval.cli import _POINTS_3D_METRIC_NAMESPACE, _clean_metric_tree

        gt, K = _gt_pointmap()
        pred = apply_point_transform(gt, 0.5, _rot_z(0.2), np.zeros(3, np.float32))
        result = evaluate_points_3d_samples(
            _OnePointMapDataset(pred, gt, K),
            num_workers=0,
            alignment_mode="similarity",
        )
        eval_tree = {"points3d": {"eval": {}}}
        for space, key in (("native", "points_3d_native"), ("metric", "points_3d_metric")):
            branch = result.get(key)
            if branch is not None:
                eval_tree["points3d"]["eval"][space] = _clean_metric_tree(branch)

        names = _flatten_numeric(eval_tree)
        assert names, "expected at least one metric path"
        for name in names:
            assert name.startswith(f"{_POINTS_3D_METRIC_NAMESPACE}.")
            assert _METRIC_NAMESPACE_RE.fullmatch(name), name

    def test_validators_accept_points_3d(self, tmp_path):
        from euler_eval.cli import validate_dataset_entry, validate_gt_config

        d = tmp_path / "p3d"
        d.mkdir()
        # gt + dataset entry with only points_3d should validate cleanly
        validate_gt_config({"points_3d": {"path": str(d)}})
        validate_dataset_entry({"name": "m", "points_3d": {"path": str(d)}}, 0)
        # an entry with no recognized modality still raises
        with pytest.raises(ValueError):
            validate_dataset_entry({"name": "m"}, 0)


# ---------------------------------------------------------------------------
# Sanity checker
# ---------------------------------------------------------------------------


class TestSanity:
    def _checker(self):
        from euler_eval.sanity_checker import SanityChecker

        return SanityChecker()

    def test_input_no_valid_points_warns(self):
        sc = self._checker()
        zeros = np.zeros((4, 4, 3), np.float32)
        sc.validate_points_3d_input(zeros, zeros, "f0")
        assert sc.get_points_3d_report()["total_warnings"] >= 1

    def test_metrics_warnings(self):
        sc = self._checker()
        sc.validate_points_3d_metrics(
            mae3d=999.0, lateral_fraction=0.95, rho_a=0.01, mean_angle=80.0,
            file_id="f0",
        )
        report = sc.get_points_3d_report()
        types = set()
        for info in report["warnings_by_type"].values():
            types.add(info["type"])
        assert "camera_dominated_error" in types
        assert "rho_a_too_low" in types

    def test_report_handles_metric_warning_without_counted_input(self, capsys):
        sc = self._checker()
        sc.validate_points_3d_metrics(lateral_fraction=0.95, file_id="f0")

        sc.print_report()

        output = capsys.readouterr().out
        assert "Occurrences: 1/0 (0.0%)" in output

    def test_full_report_includes_points_3d(self):
        sc = self._checker()
        assert "points_3d" in sc.get_full_report()


# ---------------------------------------------------------------------------
# GT synthesis from depth + intrinsics
# ---------------------------------------------------------------------------


class _DepthGtDataset:
    """GT supplied as a depth map; the evaluator unprojects it on the fly."""

    def __init__(self, pred, depth, K, full_id="/scene/000000"):
        self.pred = pred
        self.depth = depth
        self.K = K
        self.full_id = full_id

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return {
            "id": self.full_id.strip("/").split("/")[-1],
            "full_id": self.full_id,
            "gt": self.depth,
            "pred": self.pred,
            "intrinsics": {"K": self.K},
        }


class TestGtSynthesis:
    def test_synthesized_gt_matches_unprojected_prediction(self):
        h, w = 40, 60
        K = _make_intrinsics(400.0, 400.0, w / 2.0, h / 2.0)
        depth = np.random.RandomState(3).rand(h, w).astype(np.float32) * 8.0 + 2.0
        gt_pts = unproject_depth_to_points(depth, K, depth_is_radial=False)
        result = evaluate_points_3d_samples(
            _DepthGtDataset(gt_pts.copy(), depth, K),
            num_workers=0,
            alignment_mode="none",
            gt_is_depth=True,
            gt_depth_is_radial=False,  # this GT depth is planar Z
        )
        pe = result["points_3d"]["point_error"]["image_mean"]
        assert pe["mae3d"] < 1e-4
        assert result["dataset_info"]["gt_synthesized_from_depth"] is True
        assert result["dataset_info"]["fov_domain"] == "sfov"

    def test_radial_gt_depth_synthesis(self):
        h, w = 30, 40
        K = _make_intrinsics(400.0, 400.0, w / 2.0, h / 2.0)
        planar = np.random.RandomState(5).rand(h, w).astype(np.float32) * 6.0 + 2.0
        gt_pts = unproject_depth_to_points(planar, K, depth_is_radial=False)
        radial = np.linalg.norm(gt_pts, axis=-1).astype(np.float32)
        result = evaluate_points_3d_samples(
            _DepthGtDataset(gt_pts.copy(), radial, K),
            num_workers=0,
            alignment_mode="none",
            gt_is_depth=True,
            gt_depth_is_radial=True,
        )
        assert result["points_3d"]["point_error"]["image_mean"]["mae3d"] < 1e-3

    def test_synthesis_requires_intrinsics(self):
        h, w = 20, 30
        depth = np.full((h, w), 5.0, np.float32)
        pred = np.zeros((h, w, 3), np.float32)

        class _NoIntrinsics:
            def __len__(self):
                return 1

            def __getitem__(self, i):
                return {"id": "0", "full_id": "/0", "gt": depth, "pred": pred}

        with pytest.raises(ValueError):
            evaluate_points_3d_samples(
                _NoIntrinsics(), num_workers=0, gt_is_depth=True
            )

    def test_builder_requires_a_gt_source(self):
        with pytest.raises(ValueError):
            build_points_3d_eval_dataset(pred_points_3d_path="/pred")
