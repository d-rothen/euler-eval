"""Tests for the programmatic validation interface (euler_eval.validation)."""

import numpy as np
import pytest
import torch

from euler_eval import validation
from euler_eval.metrics.depth_standard import (
    STANDARD_DEPTH_METRIC_KEYS,
    compute_standard_depth_metrics,
)
from euler_eval.validation import (
    DepthValidationAggregator,
    build_validation_gt_dataset,
    evaluate_dense_depth_sample,
    evaluate_sparse_depth_sample,
    summarize_reduced_state,
)


def _grid_depth(height=48, width=64, near=2.0, far=30.0):
    """Smooth positive synthetic depth map."""
    rows = np.linspace(near, far, height, dtype=np.float32)[:, None]
    cols = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    return rows * (1.0 + 0.1 * cols)


# ---------------------------------------------------------------------------
# Dense evaluation
# ---------------------------------------------------------------------------


class TestEvaluateDenseDepthSample:
    def test_perfect_prediction(self):
        gt = _grid_depth()
        result = evaluate_dense_depth_sample(gt.copy(), gt)
        assert result is not None
        assert result.metrics["absrel"] == pytest.approx(0.0, abs=1e-7)
        assert result.metrics["delta1"] == pytest.approx(1.0)
        assert result.valid_pixels == gt.size
        assert result.scale is None and result.shift is None
        assert result.projection is None

    def test_known_relative_error(self):
        gt = _grid_depth()
        pred = gt * 1.05
        result = evaluate_dense_depth_sample(pred, gt)
        assert result.metrics["absrel"] == pytest.approx(0.05, rel=1e-5)
        assert result.metrics["mae"] == pytest.approx(0.05 * gt.mean(), rel=1e-4)
        assert result.metrics["delta1"] == pytest.approx(1.0)

    def test_matches_compute_standard_depth_metrics(self):
        rng = np.random.default_rng(0)
        gt = _grid_depth()
        pred = gt * rng.uniform(0.8, 1.2, size=gt.shape).astype(np.float32)
        expected, expected_pool = compute_standard_depth_metrics(pred, gt)
        result = evaluate_dense_depth_sample(pred, gt)
        for key in STANDARD_DEPTH_METRIC_KEYS:
            assert result.metrics[key] == pytest.approx(expected[key])
        assert result.pool_stats["count"] == expected_pool["count"]

    def test_accepts_torch_chw_tensors(self):
        gt = _grid_depth()
        pred = torch.from_numpy(gt * 1.1).unsqueeze(0)  # (1, H, W)
        result = evaluate_dense_depth_sample(pred, torch.from_numpy(gt))
        assert result.metrics["absrel"] == pytest.approx(0.1, rel=1e-5)

    def test_valid_mask_and_depth_bounds(self):
        gt = _grid_depth()
        pred = gt.copy()
        pred[:10] = gt[:10] * 2.0  # gross error in masked-out region
        mask = np.ones_like(gt, dtype=bool)
        mask[:10] = False
        result = evaluate_dense_depth_sample(pred, gt, valid_mask=mask)
        assert result.metrics["absrel"] == pytest.approx(0.0, abs=1e-7)
        assert result.valid_pixels == int(mask.sum())

        bounded = evaluate_dense_depth_sample(
            gt.copy(), gt, min_depth=5.0, max_depth=20.0
        )
        assert bounded.valid_pixels == int(((gt >= 5.0) & (gt <= 20.0)).sum())

    def test_gt_resolution_mismatch_is_aligned(self):
        pytest.importorskip("cv2")  # align_to_prediction resize fallback
        gt_full = _grid_depth(96, 128)
        pred = gt_full[::2, ::2]
        result = evaluate_dense_depth_sample(pred, gt_full)
        assert result is not None
        assert result.valid_pixels == pred.size
        # Downsampling artifacts only; must stay small.
        assert result.metrics["absrel"] < 0.03

    def test_gt_vae_crop_mismatch_is_aligned(self):
        # Pred dims are multiples of 8 and GT is <8 px larger: top-left crop,
        # no cv2 needed.
        gt_full = _grid_depth(53, 70)
        pred = gt_full[:48, :64] * 1.02
        result = evaluate_dense_depth_sample(pred, gt_full)
        assert result is not None
        assert result.valid_pixels == pred.size
        assert result.metrics["absrel"] == pytest.approx(0.02, rel=1e-4)

    def test_affine_alignment_recovers_scale_shift(self):
        gt = _grid_depth()
        pred = 0.02 * gt + 0.3  # normalized-style affine prediction
        raw = evaluate_dense_depth_sample(pred, gt)
        aligned = evaluate_dense_depth_sample(pred, gt, alignment="affine")
        assert aligned.metrics["absrel"] == pytest.approx(0.0, abs=1e-5)
        assert aligned.scale == pytest.approx(50.0, rel=1e-4)
        assert aligned.shift == pytest.approx(-15.0, rel=1e-3)
        assert raw.metrics["absrel"] > 0.5

    def test_too_few_valid_pixels_returns_none(self):
        gt = np.zeros((16, 16), dtype=np.float32)
        gt[0, :5] = 10.0
        assert evaluate_dense_depth_sample(gt.copy(), gt) is None

    def test_invalid_alignment_mode(self):
        gt = _grid_depth()
        with pytest.raises(ValueError, match="alignment"):
            evaluate_dense_depth_sample(gt, gt, alignment="auto_affine")


# ---------------------------------------------------------------------------
# Sparse evaluation
# ---------------------------------------------------------------------------


def _synthetic_cloud_and_pred(height=48, width=64, n_side=12):
    """Points in the camera frame plus the exactly consistent planar depth."""
    K = np.array(
        [[100.0, 0.0, width / 2.0], [0.0, 100.0, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    us = np.linspace(8, width - 8, n_side)
    vs = np.linspace(8, height - 8, n_side)
    uu, vv = np.meshgrid(us, vs)
    uu = np.rint(uu).astype(np.int64).ravel()
    vv = np.rint(vv).astype(np.int64).ravel()
    z = np.linspace(4.0, 40.0, uu.size)
    x = (uu - K[0, 2]) * z / K[0, 0]
    y = (vv - K[1, 2]) * z / K[1, 1]
    points = np.stack([x, y, z], axis=1)

    pred_planar = np.full((height, width), 15.0, dtype=np.float32)
    pred_planar[vv, uu] = z.astype(np.float32)
    return points, K, pred_planar, (uu, vv, z)


class TestEvaluateSparseDepthSample:
    def test_consistent_planar_prediction_scores_zero_error(self):
        points, K, pred, (uu, vv, z) = _synthetic_cloud_and_pred()
        result = evaluate_sparse_depth_sample(
            pred, points, K, np.eye(4), pred_is_radial=False
        )
        assert result is not None
        assert result.metrics["absrel"] == pytest.approx(0.0, abs=1e-5)
        assert result.metrics["delta1"] == pytest.approx(1.0)
        assert result.projection["projected_pixels"] == result.valid_pixels
        assert result.projection["input_points"] == points.shape[0]

    def test_radial_prediction_skips_conversion(self):
        points, K, pred_planar, (uu, vv, z) = _synthetic_cloud_and_pred()
        radial = np.linalg.norm(points, axis=1)
        pred_radial = np.full_like(pred_planar, 15.0)
        pred_radial[vv, uu] = radial.astype(np.float32)
        result = evaluate_sparse_depth_sample(
            pred_radial, points, K, np.eye(4), pred_is_radial=True
        )
        assert result.metrics["absrel"] == pytest.approx(0.0, abs=1e-5)

    def test_planar_vs_radial_mismatch_is_visible(self):
        """Scoring a planar prediction as radial must degrade off-center points."""
        points, K, pred, _ = _synthetic_cloud_and_pred()
        correct = evaluate_sparse_depth_sample(
            pred, points, K, np.eye(4), pred_is_radial=False
        )
        wrong = evaluate_sparse_depth_sample(
            pred, points, K, np.eye(4), pred_is_radial=True
        )
        assert wrong.metrics["absrel"] > correct.metrics["absrel"]

    def test_composed_lidar_extrinsics(self):
        points_cam, K, pred, _ = _synthetic_cloud_and_pred()
        # Shared-frame poses: camera pose and lidar pose with a known offset.
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = [0.2, -0.1, 0.4]
        lidar_pose = np.eye(4)
        lidar_pose[:3, 3] = [1.0, 0.5, -0.3]
        # source→camera = inv(camera_pose) @ lidar_pose; move points to lidar frame.
        source_to_camera = np.linalg.inv(camera_pose) @ lidar_pose
        points_lidar = (
            np.linalg.inv(source_to_camera)
            @ np.concatenate(
                [points_cam, np.ones((points_cam.shape[0], 1))], axis=1
            ).T
        ).T[:, :3]

        direct = evaluate_sparse_depth_sample(
            pred, points_lidar, K, source_to_camera
        )
        composed = evaluate_sparse_depth_sample(
            pred,
            points_lidar,
            K,
            camera_pose,
            lidar_extrinsics=lidar_pose,
        )
        assert composed.metrics["absrel"] == pytest.approx(
            direct.metrics["absrel"], abs=1e-7
        )
        assert composed.metrics["absrel"] == pytest.approx(0.0, abs=1e-5)

    def test_affine_alignment_on_normalized_prediction(self):
        points, K, pred_planar, (uu, vv, z) = _synthetic_cloud_and_pred()
        radial = np.linalg.norm(points, axis=1)
        pred_norm = np.zeros_like(pred_planar)
        pred_norm[vv, uu] = ((radial - radial.min()) / np.ptp(radial)).astype(
            np.float32
        )
        result = evaluate_sparse_depth_sample(
            pred_norm, points, K, np.eye(4), alignment="affine"
        )
        assert result is not None
        assert result.scale is not None and result.shift is not None
        assert result.metrics["absrel"] == pytest.approx(0.0, abs=1e-4)

    def test_depth_bounds_filter_projected_points(self):
        points, K, pred, (uu, vv, z) = _synthetic_cloud_and_pred()
        result = evaluate_sparse_depth_sample(
            pred, points, K, np.eye(4), min_depth=10.0, max_depth=30.0
        )
        # Bounds apply to the radial GT depth, which is >= planar z.
        radial = np.linalg.norm(points, axis=1)
        assert result.valid_pixels == int(
            ((radial >= 10.0) & (radial <= 30.0)).sum()
        )

    def test_extra_valid_mask(self):
        points, K, pred, (uu, vv, z) = _synthetic_cloud_and_pred()
        mask = np.zeros(pred.shape, dtype=bool)
        mask[: pred.shape[0] // 2] = True
        result = evaluate_sparse_depth_sample(
            pred, points, K, np.eye(4), valid_mask=mask
        )
        assert result.valid_pixels == int((vv < pred.shape[0] // 2).sum())

    def test_no_projected_points_returns_none(self):
        _, K, pred, _ = _synthetic_cloud_and_pred()
        behind = np.array([[0.0, 0.0, -5.0], [1.0, 1.0, -2.0]])
        assert (
            evaluate_sparse_depth_sample(pred, behind, K, np.eye(4)) is None
        )

    def test_accepts_torch_inputs(self):
        points, K, pred, _ = _synthetic_cloud_and_pred()
        result = evaluate_sparse_depth_sample(
            torch.from_numpy(pred).unsqueeze(0),
            torch.from_numpy(points),
            torch.from_numpy(K),
            torch.eye(4),
        )
        assert result.metrics["absrel"] == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestDepthValidationAggregator:
    def _make_samples(self, seed=0, count=6):
        rng = np.random.default_rng(seed)
        gt = _grid_depth()
        samples = []
        for _ in range(count):
            pred = gt * rng.uniform(0.85, 1.25, size=gt.shape).astype(np.float32)
            samples.append(evaluate_dense_depth_sample(pred, gt))
        return samples

    def test_summary_reducers(self):
        samples = self._make_samples()
        agg = DepthValidationAggregator()
        for sample in samples:
            assert agg.update(sample)
        agg.update(None)  # a skipped sample

        summary = agg.summary()
        assert summary["num_samples"] == len(samples)
        assert summary["num_skipped"] == 1
        assert "projection" not in summary
        expected_mean = np.mean([s.metrics["absrel"] for s in samples])
        expected_median = np.median([s.metrics["absrel"] for s in samples])
        standard = summary["standard"]
        assert standard["image_mean"]["absrel"] == pytest.approx(expected_mean)
        assert standard["image_median"]["absrel"] == pytest.approx(expected_median)
        pool_expected = sum(s.pool_stats["sum_absrel"] for s in samples) / sum(
            s.pool_stats["count"] for s in samples
        )
        assert standard["pixel_pool"]["absrel"] == pytest.approx(pool_expected)

    def test_reduced_state_matches_split_aggregation(self):
        samples = self._make_samples(seed=1, count=8)
        full = DepthValidationAggregator()
        left = DepthValidationAggregator()
        right = DepthValidationAggregator()
        for sample in samples:
            full.update(sample)
        for sample in samples[:3]:
            left.update(sample)
        for sample in samples[3:]:
            right.update(sample)

        keys = DepthValidationAggregator.state_keys()
        assert set(left.reduced_state()) == set(keys)
        summed = {
            key: left.reduced_state()[key] + right.reduced_state()[key]
            for key in keys
        }
        rebuilt = summarize_reduced_state(summed)
        local = full.summary()
        assert rebuilt["num_samples"] == local["num_samples"]
        assert rebuilt["valid_pixels"] == local["valid_pixels"]
        for metric in STANDARD_DEPTH_METRIC_KEYS:
            assert rebuilt["standard"]["image_mean"][metric] == pytest.approx(
                local["standard"]["image_mean"][metric]
            )
            assert rebuilt["standard"]["pixel_pool"][metric] == pytest.approx(
                local["standard"]["pixel_pool"][metric]
            )

    def test_projection_totals(self):
        points, K, pred, _ = _synthetic_cloud_and_pred()
        agg = DepthValidationAggregator()
        for _ in range(2):
            agg.update(evaluate_sparse_depth_sample(pred, points, K, np.eye(4)))
        summary = agg.summary()
        assert summary["projection"]["input_points"] == 2 * points.shape[0]
        assert summary["projection"]["projected_pixels"] == summary["valid_pixels"]

    def test_empty_reduced_state_summarizes_to_nan(self):
        state = {key: 0.0 for key in DepthValidationAggregator.state_keys()}
        rebuilt = summarize_reduced_state(state)
        assert rebuilt["num_samples"] == 0
        assert np.isnan(rebuilt["standard"]["image_mean"]["absrel"])
        assert np.isnan(rebuilt["standard"]["pixel_pool"]["absrel"])


# ---------------------------------------------------------------------------
# GT-only dataset builder
# ---------------------------------------------------------------------------


class _CapturedDataset:
    def __init__(self, *, modalities, hierarchical_modalities=None, transforms=None):
        self.modalities = modalities
        self.hierarchical_modalities = hierarchical_modalities or {}
        self.transforms = transforms


class TestBuildValidationGtDataset:
    @pytest.fixture(autouse=True)
    def _capture(self, monkeypatch):
        monkeypatch.setattr(validation, "MultiModalDataset", _CapturedDataset)

    def test_requires_exactly_one_gt_kind(self):
        with pytest.raises(ValueError, match="exactly one"):
            build_validation_gt_dataset()
        with pytest.raises(ValueError, match="exactly one"):
            build_validation_gt_dataset(
                depth_path="/gt/depth", sparse_depth_path="/gt/lidar"
            )

    def test_sparse_requires_projection_calibration(self):
        with pytest.raises(ValueError, match="intrinsics_path"):
            build_validation_gt_dataset(sparse_depth_path="/gt/lidar")

    def test_dense_dataset_modalities(self):
        dataset = build_validation_gt_dataset(
            depth_path="/gt/depth",
            rgb_path="/gt/rgb",
            split="val",
        )
        assert set(dataset.modalities) == {"depth", "rgb"}
        assert dataset.modalities["depth"].split == "val"
        assert dataset.modalities["depth"].modality_type == "depth"
        assert dataset.hierarchical_modalities == {}

    def test_sparse_dataset_modalities_with_inline_selectors(self):
        dataset = build_validation_gt_dataset(
            sparse_depth_path="/data/muses.zip:test#scope=lidar",
            rgb_path="/data/muses.zip#scope=rgb",
            intrinsics_path="/data/muses.zip#scope=intrinsics",
            camera_extrinsics_path="/data/muses.zip#scope=extrinsics",
            lidar_extrinsics_path="/data/muses.zip#scope=lidar_pose",
            split="val",
        )
        sparse = dataset.modalities["sparse_depth"]
        assert sparse.path == "/data/muses.zip"
        assert sparse.split == "test"  # inline selector wins over fallback
        assert sparse.metadata_scope == "lidar"
        assert dataset.modalities["rgb"].split == "val"  # fallback applied
        assert dataset.hierarchical_modalities["intrinsics"].metadata_scope == (
            "intrinsics"
        )
        cam_ext = dataset.hierarchical_modalities["camera_extrinsics"]
        assert cam_ext.metadata_scope == "extrinsics"
        lidar = dataset.hierarchical_modalities["lidar_extrinsics"]
        assert lidar.modality_type == "camera_extrinsics"
        assert lidar.metadata_scope == "lidar_pose"

    def test_transforms_are_forwarded(self):
        marker = [lambda sample: sample]
        dataset = build_validation_gt_dataset(
            depth_path="/gt/depth", transforms=marker
        )
        assert dataset.transforms is marker
