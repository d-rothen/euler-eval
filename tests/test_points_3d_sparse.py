"""Tests for points_3d evaluation against a sparse pointcloud GT.

Covers the directed (completeness/recall) cloud-distance metric, the
``project_point_cloud_to_point_map`` helper, and the
``evaluate_points_3d_sparse_samples`` pipeline that unprojects a dense depth
prediction and scores it as a 3D point map against a sparse LiDAR cloud
(``points_3d_metrics_proposal.md`` §4-D).
"""

import json
import zipfile

import numpy as np
import pytest

from euler_eval.data import (
    apply_point_transform,
    project_point_cloud_to_point_map,
    unproject_depth_to_points,
)
from euler_eval.evaluate import evaluate_points_3d_sparse_samples
from euler_eval.metrics import compute_sparse_cloud_distance_metrics
from euler_eval.sanity_checker import SanityChecker


def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_intrinsics(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def _dense_depth(h=40, w=60, seed=0):
    rng = np.random.RandomState(seed)
    return (rng.rand(h, w).astype(np.float32) * 8.0 + 2.0)


def _sparse_cloud_from_depth(depth, K, n=300, seed=1):
    """Sample a sparse camera-frame cloud from a dense planar-depth surface."""
    h, w = depth.shape
    gt_dense = unproject_depth_to_points(depth, K, depth_is_radial=False)
    rng = np.random.RandomState(seed)
    vs = rng.randint(2, h - 2, n)
    us = rng.randint(2, w - 2, n)
    return gt_dense[vs, us].astype(np.float32)


class _SparsePointDataset:
    """Minimal MultiModalDataset stand-in: sparse GT cloud + dense depth pred."""

    def __init__(self, cloud, pred_depth, K, extrinsics=None,
                 segmentation=None, full_id="/scene/000000"):
        self.cloud = cloud
        self.pred = pred_depth
        self.K = K
        self.extrinsics = (
            np.eye(4, dtype=np.float32) if extrinsics is None else extrinsics
        )
        self.segmentation = segmentation
        self.full_id = full_id

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        sample = {
            "id": self.full_id.strip("/").split("/")[-1],
            "full_id": self.full_id,
            "gt": self.cloud,
            "pred": self.pred,
            "intrinsics": {"K": self.K},
            "camera_extrinsics": {"lidar2rgb": self.extrinsics},
        }
        if self.segmentation is not None:
            sample["segmentation"] = {"seg": self.segmentation}
        return sample


def _perfect_dataset(seed=0, **kwargs):
    depth = _dense_depth(seed=seed)
    K = _make_intrinsics(400.0, 400.0, depth.shape[1] / 2.0, depth.shape[0] / 2.0)
    cloud = _sparse_cloud_from_depth(depth, K, seed=seed + 1)
    return _SparsePointDataset(cloud, depth, K, **kwargs), depth, K, cloud


def test_sparse_evaluation_counts_sanity_check_samples():
    dataset, _, _, _ = _perfect_dataset()
    sanity_checker = SanityChecker()

    evaluate_points_3d_sparse_samples(
        dataset,
        pred_is_radial=False,
        num_workers=0,
        sanity_checker=sanity_checker,
        alignment_mode="none",
    )

    assert sanity_checker.get_points_3d_report()["total_samples"] == 1


# ---------------------------------------------------------------------------
# Directed (sparse) cloud distance
# ---------------------------------------------------------------------------


class TestSparseCloud:
    def test_perfect_coverage(self):
        depth = _dense_depth()
        K = _make_intrinsics(400.0, 400.0, 30.0, 20.0)
        pred = unproject_depth_to_points(depth, K, depth_is_radial=False).reshape(-1, 3)
        gt = pred[::7]  # sparse subset of the dense surface
        m = compute_sparse_cloud_distance_metrics(pred, gt)
        assert m["chamfer"]["completeness"] < 1e-6
        assert m["chamfer"]["median"] < 1e-6
        assert m["fscore"]["tau_0_1"]["recall"] == 1.0

    def test_only_directed_keys(self):
        pred = np.random.RandomState(0).randn(500, 3).astype(np.float32)
        gt = np.random.RandomState(1).randn(50, 3).astype(np.float32)
        m = compute_sparse_cloud_distance_metrics(pred, gt)
        # The misleading pred->gt (accuracy / precision / f1) side is omitted.
        assert set(m["chamfer"]) == {"completeness", "median"}
        for tau_block in m["fscore"].values():
            assert set(tau_block) == {"recall"}

    def test_f_a_is_available_without_publishing_symmetric_diagnostics(self):
        points = np.random.RandomState(2).randn(100, 3).astype(np.float32)
        m = compute_sparse_cloud_distance_metrics(
            points,
            points,
            fscore_auc_max_threshold=1.0,
        )
        assert m["f_a"] == 100.0
        assert set(m["chamfer"]) == {"completeness", "median"}
        for tau_block in m["fscore"].values():
            assert set(tau_block) == {"recall"}

    def test_offset_cloud_increases_completeness(self):
        depth = _dense_depth()
        K = _make_intrinsics(400.0, 400.0, 30.0, 20.0)
        pred = unproject_depth_to_points(depth, K, depth_is_radial=False).reshape(-1, 3)
        gt = pred[::7] + np.array([0.0, 0.0, 3.0], np.float32)  # push off the surface
        m = compute_sparse_cloud_distance_metrics(pred, gt)
        assert m["chamfer"]["completeness"] > 0.1
        assert m["fscore"]["tau_0_05"]["recall"] < 1.0

    def test_recall_is_monotone_in_threshold(self):
        depth = _dense_depth()
        K = _make_intrinsics(400.0, 400.0, 30.0, 20.0)
        pred = unproject_depth_to_points(depth, K, depth_is_radial=False).reshape(-1, 3)
        gt = pred[::5] + np.array([0.15, 0.0, 0.0], np.float32)
        m = compute_sparse_cloud_distance_metrics(pred, gt)
        recalls = [m["fscore"][f"tau_{t}"]["recall"]
                   for t in ("0_05", "0_1", "0_25", "0_5")]
        assert recalls == sorted(recalls)

    def test_empty(self):
        assert compute_sparse_cloud_distance_metrics(
            np.zeros((0, 3)), np.ones((3, 3))
        ) is None
        assert compute_sparse_cloud_distance_metrics(
            np.ones((3, 3)), np.zeros((0, 3))
        ) is None


# ---------------------------------------------------------------------------
# Projection to a camera-frame point map
# ---------------------------------------------------------------------------


class TestProjection:
    def test_nearest_point_per_pixel(self):
        intrinsics = np.eye(3, dtype=np.float32)
        extrinsics = np.eye(4, dtype=np.float32)
        point_cloud = np.array(
            [
                [4.0, 4.0, 4.0, 0.0],  # same pixel as next, farther away
                [2.0, 2.0, 2.0, 0.0],
                [4.0, 2.0, 2.0, 0.0],
                [1.0, 1.0, -1.0, 0.0],  # behind camera
                [20.0, 20.0, 1.0, 0.0],  # outside image
            ],
            dtype=np.float32,
        )
        pmap, mask, meta = project_point_cloud_to_point_map(
            point_cloud, intrinsics, extrinsics, image_shape=(3, 3)
        )
        assert int(mask.sum()) == 2
        assert meta["input_points"] == 5
        assert meta["projected_pixels"] == 2
        # Nearest point at pixel (1,1) is (2,2,2), not the farther (4,4,4).
        np.testing.assert_allclose(pmap[1, 1], [2.0, 2.0, 2.0], rtol=1e-6)
        np.testing.assert_allclose(pmap[1, 2], [4.0, 2.0, 2.0], rtol=1e-6)
        np.testing.assert_array_equal(pmap[~mask], 0.0)

    def test_extrinsics_applied(self):
        intrinsics = np.eye(3, dtype=np.float32)
        extrinsics = np.eye(4, dtype=np.float32)
        extrinsics[0, 3] = 1.0  # +1 in camera X
        cloud = np.array([[0.0, 0.0, 2.0]], dtype=np.float32)
        pmap, mask, _ = project_point_cloud_to_point_map(
            cloud, intrinsics, extrinsics, image_shape=(5, 5)
        )
        assert int(mask.sum()) == 1
        # camera point = extrinsics @ [0,0,2,1] = [1,0,2]
        cam_pt = pmap[mask][0]
        np.testing.assert_allclose(cam_pt, [1.0, 0.0, 2.0], atol=1e-5)


# ---------------------------------------------------------------------------
# End-to-end evaluation
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_perfect_metric_prediction_native_only(self):
        ds, depth, K, cloud = _perfect_dataset()
        res = evaluate_points_3d_sparse_samples(
            ds, pred_is_radial=False, num_workers=0, alignment_mode="none"
        )
        assert res["points_3d_metric"] is None
        assert res["space_info"]["emitted_spaces"] == ["native"]
        assert res["dataset_info"]["gt_representation"] == "point_cloud"
        assert res["dataset_info"]["fov_domain"] == "sfov"
        p = res["points_3d"]
        assert p["point_error"]["image_mean"]["mae3d"] < 1e-4
        assert p["cloud_distance"]["chamfer"]["completeness"] < 1e-4
        assert p["cloud_distance"]["fscore"]["tau_0_1"]["recall"] == 1.0
        assert "f_a" in p["cloud_distance"]
        assert res["dataset_info"]["f_a_max_threshold"] == pytest.approx(
            res["dataset_info"]["max_depth"] / 20.0
        )
        assert p["error_decomposition"]["rho_a"]["mean"] > 0.9
        # No dense-neighbourhood geometric metrics on sparse GT.
        assert "geometric" not in p

    def test_affine_alignment_recovers(self):
        depth = _dense_depth(seed=3)
        K = _make_intrinsics(400.0, 400.0, 30.0, 20.0)
        cloud = _sparse_cloud_from_depth(depth, K, seed=4)
        pred = (depth * 0.3 + 5.0).astype(np.float32)  # up-to-affine corruption
        ds = _SparsePointDataset(cloud, pred, K)
        res = evaluate_points_3d_sparse_samples(
            ds, pred_is_radial=False, num_workers=0, alignment_mode="affine"
        )
        assert res["space_info"]["emitted_spaces"] == ["native", "metric"]
        assert res["space_info"]["metric_space_source"] == "scale_shift"
        native_mae = res["points_3d_native"]["point_error"]["image_mean"]["mae3d"]
        metric_mae = res["points_3d_metric"]["point_error"]["image_mean"]["mae3d"]
        assert metric_mae < native_mae
        assert metric_mae < 0.1
        assert res["points_3d"] is res["points_3d_metric"]

    def test_auto_affine_detects_normalized(self):
        depth = _dense_depth(seed=5)
        K = _make_intrinsics(400.0, 400.0, 30.0, 20.0)
        cloud = _sparse_cloud_from_depth(depth, K, seed=6)
        pred = ((depth - depth.min()) / (depth.max() - depth.min())).astype(np.float32)
        ds = _SparsePointDataset(cloud, pred, K)
        res = evaluate_points_3d_sparse_samples(
            ds, pred_is_radial=False, num_workers=0, alignment_mode="auto_affine"
        )
        assert res["space_info"]["input_space_detected"] == "normalized"
        assert res["space_info"]["calibration_applied"] is True
        assert res["points_3d_metric"]["point_error"]["image_mean"]["mae3d"] < 0.1

    def test_auto_affine_metric_prediction_no_alignment(self):
        ds, *_ = _perfect_dataset(seed=7)
        res = evaluate_points_3d_sparse_samples(
            ds, pred_is_radial=False, num_workers=0, alignment_mode="auto_affine"
        )
        assert res["space_info"]["input_space_detected"] == "metric"
        assert res["space_info"]["calibration_applied"] is False
        assert res["points_3d_metric"] is None

    def test_affine_hint_forces_alignment(self):
        depth = _dense_depth(seed=8)
        K = _make_intrinsics(400.0, 400.0, 30.0, 20.0)
        cloud = _sparse_cloud_from_depth(depth, K, seed=9)
        pred = (depth * 10.0 + 3.0).astype(np.float32)  # non-normalized range
        ds = _SparsePointDataset(cloud, pred, K)
        res = evaluate_points_3d_sparse_samples(
            ds, pred_is_radial=False, num_workers=0,
            alignment_mode="auto_affine", input_space_hint="affine",
        )
        assert res["space_info"]["input_space_detected"] == "affine"
        assert res["space_info"]["calibration_applied"] is True

    def test_per_file_metrics_structure(self):
        ds, *_ = _perfect_dataset()
        ds.full_id = "/scene_01/cam0/000000"
        res = evaluate_points_3d_sparse_samples(
            ds, pred_is_radial=False, num_workers=0, alignment_mode="none"
        )
        pfm = res["per_file_metrics"]
        node = pfm["children"]["scene_01"]["children"]["cam0"]
        entry = node["files"][0]
        assert entry["id"] == "000000"
        assert "points_3d" in entry["metrics"]
        assert "cloud_distance" in entry["metrics"]["points_3d"]
        assert "points_3d_native" in entry["metrics"]

    def test_sky_mask_excludes_pixels(self):
        depth = _dense_depth(seed=2)
        K = _make_intrinsics(400.0, 400.0, 30.0, 20.0)
        cloud = _sparse_cloud_from_depth(depth, K, seed=3)
        sky = np.zeros(depth.shape, dtype=bool)
        sky[:, :30] = True  # left half is "sky" -> excluded
        ds = _SparsePointDataset(cloud, depth, K, segmentation=sky)
        masked = evaluate_points_3d_sparse_samples(
            ds, pred_is_radial=False, num_workers=0,
            alignment_mode="none", sky_mask_enabled=True,
        )
        full = evaluate_points_3d_sparse_samples(
            _SparsePointDataset(cloud, depth, K), pred_is_radial=False,
            num_workers=0, alignment_mode="none",
        )
        assert (
            masked["dataset_info"]["evaluated_points"]
            < full["dataset_info"]["evaluated_points"]
        )

    def test_no_valid_projection_does_not_crash(self):
        K = _make_intrinsics(400.0, 400.0, 30.0, 20.0)
        cloud_behind = np.array([[0.0, 0.0, -5.0], [1.0, 1.0, -3.0]], np.float32)
        ds = _SparsePointDataset(cloud_behind, np.full((40, 60), 5.0, np.float32), K)
        res = evaluate_points_3d_sparse_samples(
            ds, pred_is_radial=False, num_workers=0, alignment_mode="affine"
        )
        assert res["dataset_info"]["projected_pixels"] == 0
        assert res["dataset_info"]["evaluated_points"] == 0
        # No point_error / cloud_distance when nothing projects, but it is safe.
        assert "cloud_distance" not in res["points_3d"]

    def test_empty_dataset_raises(self):
        class _Empty:
            def __len__(self):
                return 0

            def __getitem__(self, i):
                raise IndexError

        with pytest.raises(ValueError):
            evaluate_points_3d_sparse_samples(_Empty(), pred_is_radial=False)

    def test_invalid_alignment_mode_raises(self):
        ds, *_ = _perfect_dataset()
        with pytest.raises(ValueError):
            evaluate_points_3d_sparse_samples(
                ds, pred_is_radial=False, num_workers=0, alignment_mode="bogus"
            )

    def test_missing_extrinsics_raises(self):
        depth = _dense_depth()
        K = _make_intrinsics(400.0, 400.0, 30.0, 20.0)
        cloud = _sparse_cloud_from_depth(depth, K)

        class _NoExtrinsics:
            def __len__(self):
                return 1

            def __getitem__(self, i):
                return {
                    "id": "0", "full_id": "/0", "gt": cloud, "pred": depth,
                    "intrinsics": {"K": K},
                }

        with pytest.raises(ValueError):
            evaluate_points_3d_sparse_samples(
                _NoExtrinsics(), pred_is_radial=False, num_workers=0
            )


# ---------------------------------------------------------------------------
# Native points_3d prediction vs sparse GT (scored directly, no unprojection)
# ---------------------------------------------------------------------------


class TestEvaluateNativePoints:
    def _pointmap_dataset(self, pred_points, seed=0):
        depth = _dense_depth(seed=seed)
        K = _make_intrinsics(400.0, 400.0, depth.shape[1] / 2.0, depth.shape[0] / 2.0)
        gt_dense = unproject_depth_to_points(depth, K, depth_is_radial=False)
        cloud = _sparse_cloud_from_depth(depth, K, seed=seed + 1)
        return _SparsePointDataset(cloud, pred_points, K), gt_dense, K

    def test_perfect_metric_pointmap_native_only(self):
        depth = _dense_depth()
        K = _make_intrinsics(400.0, 400.0, 30.0, 20.0)
        gt_dense = unproject_depth_to_points(depth, K, depth_is_radial=False)
        cloud = _sparse_cloud_from_depth(depth, K)
        ds = _SparsePointDataset(cloud, gt_dense.copy(), K)
        res = evaluate_points_3d_sparse_samples(
            ds, num_workers=0, alignment_mode="none", pred_is_depth=False
        )
        assert res["points_3d_metric"] is None
        assert res["space_info"]["emitted_spaces"] == ["native"]
        assert res["dataset_info"]["pred_representation"] == "points_3d"
        p = res["points_3d"]
        assert p["point_error"]["image_mean"]["mae3d"] < 1e-4
        assert p["cloud_distance"]["chamfer"]["completeness"] < 1e-4
        assert p["cloud_distance"]["fscore"]["tau_0_1"]["recall"] == 1.0
        assert 0.0 <= p["cloud_distance"]["f_a"] <= 100.0

    def test_similarity_recovers_relative_pointmap(self):
        ds, gt_dense, K = self._pointmap_dataset(None)
        pred = apply_point_transform(
            gt_dense, 0.5, _rot_z(0.25), np.array([1.0, 2.0, -0.5], np.float32)
        )
        ds = _SparsePointDataset(
            _sparse_cloud_from_depth(_dense_depth(), K), pred, K
        )
        res = evaluate_points_3d_sparse_samples(
            ds, num_workers=0, alignment_mode="similarity", pred_is_depth=False
        )
        assert res["space_info"]["emitted_spaces"] == ["native", "metric"]
        assert res["space_info"]["metric_space_source"] == "similarity"
        native_mae = res["points_3d_native"]["point_error"]["image_mean"]["mae3d"]
        metric_mae = res["points_3d_metric"]["point_error"]["image_mean"]["mae3d"]
        assert metric_mae < 1e-3
        assert native_mae > metric_mae

    def test_scale_recovers_scaled_pointmap(self):
        depth = _dense_depth()
        K = _make_intrinsics(400.0, 400.0, 30.0, 20.0)
        gt_dense = unproject_depth_to_points(depth, K, depth_is_radial=False)
        cloud = _sparse_cloud_from_depth(depth, K)
        ds = _SparsePointDataset(cloud, gt_dense * 4.0, K)
        res = evaluate_points_3d_sparse_samples(
            ds, num_workers=0, alignment_mode="scale", pred_is_depth=False
        )
        assert res["points_3d_metric"]["point_error"]["image_mean"]["mae3d"] < 1e-3

    def test_auto_with_relative_hint_aligns(self):
        depth = _dense_depth()
        K = _make_intrinsics(400.0, 400.0, 30.0, 20.0)
        gt_dense = unproject_depth_to_points(depth, K, depth_is_radial=False)
        pred = apply_point_transform(gt_dense, 0.5, _rot_z(0.1), np.zeros(3, np.float32))
        ds = _SparsePointDataset(_sparse_cloud_from_depth(depth, K), pred, K)
        res = evaluate_points_3d_sparse_samples(
            ds, num_workers=0, alignment_mode="auto",
            input_space_hint="relative", pred_is_depth=False,
        )
        assert res["space_info"]["calibration_applied"] is True
        assert res["space_info"]["metric_space_source"] == "similarity"

    def test_auto_metric_default_no_align(self):
        depth = _dense_depth()
        K = _make_intrinsics(400.0, 400.0, 30.0, 20.0)
        gt_dense = unproject_depth_to_points(depth, K, depth_is_radial=False)
        ds = _SparsePointDataset(_sparse_cloud_from_depth(depth, K), gt_dense.copy(), K)
        res = evaluate_points_3d_sparse_samples(
            ds, num_workers=0, alignment_mode="auto", pred_is_depth=False
        )
        assert res["space_info"]["calibration_applied"] is False
        assert res["points_3d_metric"] is None

    def test_depth_alignment_mode_rejected_for_pointmap(self):
        depth = _dense_depth()
        K = _make_intrinsics(400.0, 400.0, 30.0, 20.0)
        gt_dense = unproject_depth_to_points(depth, K, depth_is_radial=False)
        ds = _SparsePointDataset(_sparse_cloud_from_depth(depth, K), gt_dense, K)
        with pytest.raises(ValueError):
            evaluate_points_3d_sparse_samples(
                ds, num_workers=0, alignment_mode="auto_affine", pred_is_depth=False
            )


# ---------------------------------------------------------------------------
# Save wiring
# ---------------------------------------------------------------------------


class TestSaveWiring:
    def test_points_3d_uses_eval_json_in_its_modality_path(self, tmp_path):
        from euler_eval.cli import save_results

        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        cfg = {"name": "m", "points_3d": {"path": str(pred_dir)}}

        p3d_out = save_results(
            {"points3d": {"eval": {}}}, cfg, modality="points_3d"
        )

        assert p3d_out == pred_dir / "eval.json"
        assert p3d_out.exists()
        assert not (pred_dir / "points3d_eval.json").exists()
        assert not (pred_dir / "points_3d_eval.json").exists()

    def test_points_3d_writes_eval_json_into_zip(self, tmp_path):
        from euler_eval.cli import save_results

        archive = tmp_path / "predictions.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("points/frame_000.npy", b"points")

        results = {"points3d": {"eval": {"native": {}}}}
        cfg = {"name": "m", "points_3d": {"path": str(archive)}}
        p3d_out = save_results(results, cfg, modality="points_3d")

        assert p3d_out == archive / "eval.json"
        with zipfile.ZipFile(archive) as zf:
            assert "eval.json" in zf.namelist()
            assert "points3d_eval.json" not in zf.namelist()
            assert "points_3d_eval.json" not in zf.namelist()
            assert json.loads(zf.read("eval.json")) == results


# ---------------------------------------------------------------------------
# CLI namespace conformance
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
    def test_sparse_metric_paths_conform(self):
        pytest.importorskip("euler_metric_naming")
        from euler_eval.cli import _POINTS_3D_METRIC_NAMESPACE, _clean_metric_tree

        depth = _dense_depth(seed=11)
        K = _make_intrinsics(400.0, 400.0, 30.0, 20.0)
        cloud = _sparse_cloud_from_depth(depth, K, seed=12)
        pred = (depth * 0.5 + 2.0).astype(np.float32)
        res = evaluate_points_3d_sparse_samples(
            _SparsePointDataset(cloud, pred, K),
            pred_is_radial=False, num_workers=0, alignment_mode="affine",
        )
        tree = {"points3d": {"eval": {}}}
        for space, key in (("native", "points_3d_native"),
                           ("metric", "points_3d_metric")):
            branch = res.get(key)
            if branch is not None:
                tree["points3d"]["eval"][space] = _clean_metric_tree(branch)

        names = _flatten_numeric(tree)
        assert names
        for name in names:
            assert name.startswith(f"{_POINTS_3D_METRIC_NAMESPACE}.")
            assert _METRIC_NAMESPACE_RE.fullmatch(name), name

    def test_envelope_descriptions_cover_sparse_leaves(self):
        pytest.importorskip("euler_metric_naming")
        from euler_eval.cli import (
            _POINTS_3D_EVAL_DESCRIPTIONS,
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
        descs = envelope["metricDescriptions"]
        # Leaves emitted on the sparse path must all be described.
        assert "chamfer.completeness" in descs
        assert "chamfer.median" in descs
        assert "recall" in descs
        assert "mae3d" in descs
