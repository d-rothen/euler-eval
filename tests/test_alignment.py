"""Tests for dimension alignment and scale-and-shift depth alignment."""

import numpy as np
import pytest

from euler_eval.data import (
    align_intrinsics_to_prediction,
    align_to_ground_truth,
    align_to_prediction,
    classify_spatial_alignment,
    compute_scale_and_shift,
    resize_depth_gt,
    resize_mask,
    resize_prediction,
)

# ---------------------------------------------------------------------------
# align_to_prediction
# ---------------------------------------------------------------------------


class TestAlignToPredicition:
    """Tests for VAE-crop-aware dimension alignment."""

    def test_same_dims_noop(self):
        gt = np.ones((64, 128), dtype=np.float32)
        pred = np.ones((64, 128), dtype=np.float32)
        result = align_to_prediction(gt, pred)
        assert result is gt  # exact same object, no copy

    def test_vae_crop_depth(self):
        """GT 375x1242 -> pred 368x1240 (multiple-of-8 crop)."""
        gt = np.random.rand(375, 1242).astype(np.float32)
        pred = np.random.rand(368, 1240).astype(np.float32)
        result = align_to_prediction(gt, pred)
        assert result.shape == (368, 1240)
        # Should be a top-left crop, not a resize
        np.testing.assert_array_equal(result, gt[:368, :1240])

    def test_vae_crop_rgb(self):
        """Same crop logic works for (H, W, 3) RGB arrays."""
        gt = np.random.rand(375, 1242, 3).astype(np.float32)
        pred = np.random.rand(368, 1240, 3).astype(np.float32)
        result = align_to_prediction(gt, pred)
        assert result.shape == (368, 1240, 3)
        np.testing.assert_array_equal(result, gt[:368, :1240])

    def test_vae_crop_bool_mask(self):
        """Bool masks are cropped identically."""
        gt = np.ones((375, 1242), dtype=bool)
        gt[370:, :] = False
        pred = np.zeros((368, 1240), dtype=np.float32)
        result = align_to_prediction(gt, pred)
        assert result.shape == (368, 1240)
        assert result.dtype == bool
        np.testing.assert_array_equal(result, gt[:368, :1240])

    def test_vae_crop_only_height(self):
        """Crop needed on height only (width already multiple of 8)."""
        gt = np.random.rand(133, 256).astype(np.float32)
        pred = np.random.rand(128, 256).astype(np.float32)
        result = align_to_prediction(gt, pred)
        assert result.shape == (128, 256)
        np.testing.assert_array_equal(result, gt[:128, :256])

    def test_vae_crop_only_width(self):
        """Crop needed on width only (height already multiple of 8)."""
        gt = np.random.rand(256, 133).astype(np.float32)
        pred = np.random.rand(256, 128).astype(np.float32)
        result = align_to_prediction(gt, pred)
        assert result.shape == (256, 128)
        np.testing.assert_array_equal(result, gt[:256, :128])

    def test_no_crop_when_delta_too_large(self):
        """Delta >= 8 should fall back to resize, not crop."""
        gt = np.random.rand(200, 200).astype(np.float32)
        pred = np.random.rand(184, 184).astype(np.float32)  # delta=16
        result = align_to_prediction(gt, pred)
        assert result.shape == (184, 184)
        # Should NOT be a simple slice (resize changes values)
        assert not np.array_equal(result, gt[:184, :184])

    def test_resize_fallback_preserves_dtype(self):
        """Resize fallback returns float32 for float32 input."""
        gt = np.random.rand(200, 300).astype(np.float32)
        pred = np.random.rand(100, 150).astype(np.float32)
        result = align_to_prediction(gt, pred)
        assert result.shape == (100, 150)
        assert result.dtype == np.float32

    def test_resize_fallback_bool_mask(self):
        """Bool masks use nearest interpolation and come back as bool."""
        gt = np.ones((200, 300), dtype=bool)
        gt[100:, :] = False
        pred = np.zeros((100, 150), dtype=np.float32)
        result = align_to_prediction(gt, pred)
        assert result.shape == (100, 150)
        assert result.dtype == bool

    def test_pred_not_multiple_of_8_uses_resize(self):
        """If pred dims aren't multiples of 8, can't be a VAE crop."""
        gt = np.random.rand(103, 205).astype(np.float32)
        pred = np.random.rand(99, 201).astype(np.float32)  # not mult of 8
        result = align_to_prediction(gt, pred)
        assert result.shape == (99, 201)

    def test_pred_larger_than_gt_uses_resize(self):
        """Edge case: pred larger than GT (negative delta) -> resize."""
        gt = np.random.rand(64, 64).astype(np.float32)
        pred = np.random.rand(128, 128).astype(np.float32)
        result = align_to_prediction(gt, pred)
        assert result.shape == (128, 128)

    def test_large_downscale_different_aspect_ratio(self):
        """Typical case: GT 1024x2048 -> pred 370x780 (warped aspect ratio)."""
        gt = np.random.rand(1024, 2048).astype(np.float32)
        pred = np.random.rand(370, 780).astype(np.float32)
        with pytest.warns(UserWarning, match="aspect-ratio mismatch"):
            result = align_to_prediction(gt, pred)
        assert result.shape == (370, 780)
        assert result.dtype == np.float32

    def test_large_downscale_rgb(self):
        """Large downscale for RGB arrays uses INTER_AREA."""
        gt = np.random.rand(1024, 2048, 3).astype(np.float32)
        pred = np.random.rand(370, 780, 3).astype(np.float32)
        with pytest.warns(UserWarning, match="aspect-ratio mismatch"):
            result = align_to_prediction(gt, pred)
        assert result.shape == (370, 780, 3)
        assert result.dtype == np.float32

    def test_no_warning_when_aspect_ratio_matches(self):
        """No warning for matching aspect ratios (just different resolution)."""
        gt = np.random.rand(1024, 2048).astype(np.float32)
        pred = np.random.rand(512, 1024).astype(np.float32)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = align_to_prediction(gt, pred)
        assert result.shape == (512, 1024)


# ---------------------------------------------------------------------------
# classify_spatial_alignment
# ---------------------------------------------------------------------------


class TestClassifySpatialAlignment:
    """Tests for alignment method classification."""

    def test_same_dims(self):
        assert classify_spatial_alignment(64, 128, 64, 128) == "none"

    def test_vae_crop(self):
        """GT 375x1242 -> pred 368x1240 (multiple-of-8, delta < 8)."""
        assert classify_spatial_alignment(375, 1242, 368, 1240) == "vae_crop"

    def test_vae_crop_height_only(self):
        assert classify_spatial_alignment(133, 256, 128, 256) == "vae_crop"

    def test_resize_large_delta(self):
        """Delta >= 8 -> resize."""
        assert classify_spatial_alignment(200, 200, 184, 184) == "resize"

    def test_resize_different_aspect(self):
        """Vastly different dimensions -> resize."""
        assert classify_spatial_alignment(1024, 2048, 370, 780) == "resize"

    def test_resize_pred_larger(self):
        """Pred larger than GT (negative delta) -> resize."""
        assert classify_spatial_alignment(64, 64, 128, 128) == "resize"

    def test_resize_pred_not_mult_8(self):
        """Pred dims not multiple of 8 -> resize even if delta < 8."""
        assert classify_spatial_alignment(103, 205, 99, 201) == "resize"


class TestSpatialAlignmentModes:
    """Stating what the model did instead of guessing from shapes."""

    def test_crop_mode_crops_any_smaller_prediction(self):
        assert classify_spatial_alignment(400, 879, 384, 768, mode="crop") == "vae_crop"

    def test_crop_mode_rejects_a_larger_prediction(self):
        with pytest.raises(ValueError, match="'crop' needs a prediction no larger"):
            classify_spatial_alignment(64, 64, 72, 64, mode="crop")

    def test_resize_mode_overrides_the_crop_pattern(self):
        # DrivingStereo half size, 879 px wide, at the next multiple of 8 down.
        assert classify_spatial_alignment(400, 879, 400, 872) == "vae_crop"
        assert (
            classify_spatial_alignment(400, 879, 400, 872, mode="resize") == "resize"
        )

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="spatial_alignment must be one of"):
            classify_spatial_alignment(8, 8, 8, 8, mode="stretch")

    def test_align_to_prediction_follows_the_mode(self):
        gt = np.arange(400 * 879, dtype=np.float32).reshape(400, 879) + 1.0
        pred = np.zeros((400, 872), dtype=np.float32)
        cropped = align_to_prediction(gt, pred)
        np.testing.assert_array_equal(cropped, gt[:, :872])
        resized = align_to_prediction(gt, pred, spatial_alignment="resize")
        assert resized.shape == (400, 872)
        assert not np.array_equal(resized, gt[:, :872])

    def test_intrinsics_follow_the_mode(self):
        K = np.array([[500.0, 0.0, 439.0], [0.0, 500.0, 199.5], [0.0, 0.0, 1.0]])
        np.testing.assert_array_equal(
            align_intrinsics_to_prediction(K, (400, 879), (400, 872)), K
        )
        resized = align_intrinsics_to_prediction(
            K, (400, 879), (400, 872), mode="resize"
        )
        assert resized[0, 0] == pytest.approx(500.0 * 872 / 879)
        assert resized[0, 2] == pytest.approx((439.0 + 0.5) * 872 / 879 - 0.5)


class TestResizeMask:
    def test_downsampling_samples_pixel_centres(self):
        # Halving keeps source columns 1, 3, 5, 7: the centres of each pair,
        # like the image resize and the intrinsics update.
        mask = np.zeros((2, 8), dtype=bool)
        mask[:, 1::2] = True
        assert resize_mask(mask, (1, 4)).all()

    def test_upsampling_repeats_each_pixel(self):
        mask = np.array([[True, False]])
        np.testing.assert_array_equal(
            resize_mask(mask, (1, 4)), [[True, True, False, False]]
        )

    def test_bool_gt_masks_use_it(self):
        mask = np.zeros((4, 8), dtype=bool)
        mask[:, 1::2] = True
        pred = np.zeros((2, 4), dtype=np.float32)
        assert align_to_prediction(mask, pred).all()


class TestResizeDepthGt:
    def test_dense_shrink_ignores_invalid_pixels(self):
        depth = np.zeros((4, 4), dtype=np.float32)
        depth[::2, ::2] = 10.0  # one valid pixel per 2x2 block
        np.testing.assert_allclose(resize_depth_gt(depth, (2, 2)), 10.0)

    def test_dense_shrink_averages_valid_pixels(self):
        depth = np.array(
            [[10.0, 20.0, 0.0, 0.0], [0.0, np.nan, 0.0, 0.0]], dtype=np.float32
        )
        np.testing.assert_allclose(resize_depth_gt(depth, (1, 2)), [[15.0, 0.0]])

    def test_dense_growth_takes_the_nearest_pixel(self):
        depth = np.array([[10.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_equal(
            resize_depth_gt(depth, (1, 4)), [[10.0, 10.0, 0.0, 0.0]]
        )

    def test_sparse_keeps_measurements_and_the_nearest_surface(self):
        depth = np.zeros((4, 4), dtype=np.float32)
        depth[0, 1] = 10.0
        depth[1, 1] = 5.0  # same target pixel, nearer
        depth[3, 3] = 7.0
        resized = resize_depth_gt(depth, (2, 2), sparse=True)
        np.testing.assert_array_equal(resized, [[5.0, 0.0], [0.0, 7.0]])

    def test_sparse_growth_stays_sparse(self):
        depth = np.zeros((2, 2), dtype=np.float32)
        depth[0, 0] = 4.0
        resized = resize_depth_gt(depth, (4, 4), sparse=True)
        assert np.count_nonzero(resized) == 1
        assert resized.max() == 4.0

    def test_sparse_flag_reaches_align_to_prediction(self):
        depth = np.zeros((4, 4), dtype=np.float32)
        depth[1, 1] = 10.0
        pred = np.zeros((2, 2), dtype=np.float32)
        dense = align_to_prediction(depth, pred)
        sparse = align_to_prediction(depth, pred, sparse=True)
        assert dense[0, 0] == pytest.approx(10.0)
        np.testing.assert_array_equal(sparse, [[10.0, 0.0], [0.0, 0.0]])


class TestResizePrediction:
    def test_growth_is_pixel_centre_bilinear(self):
        pred = np.array([[0.0, 2.0]], dtype=np.float32)
        np.testing.assert_allclose(
            resize_prediction(pred, (1, 4)), [[0.0, 0.5, 1.5, 2.0]]
        )

    def test_shrink_averages_areas(self):
        pred = np.array([[1.0, 3.0, 5.0, 7.0]], dtype=np.float32)
        np.testing.assert_allclose(resize_prediction(pred, (1, 2)), [[2.0, 6.0]])

    def test_non_finite_pixels_carry_no_weight(self):
        pred = np.array([[1.0, 3.0, np.nan, np.nan]], dtype=np.float32)
        resized = resize_prediction(pred, (1, 2))
        assert resized[0, 0] == pytest.approx(2.0)
        assert np.isnan(resized[0, 1])

    def test_point_maps(self):
        pts = np.random.rand(4, 6, 3).astype(np.float32)
        assert resize_prediction(pts, (8, 12)).shape == (8, 12, 3)


class TestAlignToGroundTruth:
    def test_equal_shapes_are_untouched(self):
        gt = np.ones((8, 8), dtype=np.float32)
        pred = np.ones((8, 8), dtype=np.float32)
        out_pred, out_gt, method = align_to_ground_truth(pred, gt)
        assert method == "none"
        assert out_pred is pred and out_gt is gt

    def test_crop_trims_the_gt(self):
        gt = np.random.rand(53, 70).astype(np.float32)
        pred = np.random.rand(48, 64).astype(np.float32)
        out_pred, out_gt, method = align_to_ground_truth(pred, gt)
        assert method == "vae_crop"
        assert out_pred is pred
        np.testing.assert_array_equal(out_gt, gt[:48, :64])

    def test_resize_moves_the_prediction_and_keeps_the_gt(self):
        gt = np.zeros((64, 128), dtype=np.float32)
        gt[1::8, 1::8] = 10.0  # sparse
        pred = np.full((32, 64), 10.0, dtype=np.float32)
        out_pred, out_gt, method = align_to_ground_truth(pred, gt)
        assert method == "resize_prediction"
        assert out_gt is gt
        assert out_pred.shape == gt.shape
        np.testing.assert_allclose(out_pred, 10.0)

    def test_resize_mode_resizes_the_crop_pattern(self):
        gt = np.random.rand(53, 70).astype(np.float32)
        pred = np.random.rand(48, 64).astype(np.float32)
        out_pred, out_gt, method = align_to_ground_truth(
            pred, gt, spatial_alignment="resize"
        )
        assert method == "resize_prediction"
        assert out_pred.shape == (53, 70)


# ---------------------------------------------------------------------------
# compute_scale_and_shift
# ---------------------------------------------------------------------------


class TestComputeScaleAndShift:
    """Tests for least-squares affine alignment of depth maps."""

    def test_perfect_affine_recovery(self):
        """Recovers exact (s, t) when pred = (gt - t) / s."""
        gt = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        s_true, t_true = 50.0, 5.0
        pred = (gt - t_true) / s_true  # pred in ~[0.1, 0.7]

        aligned, s, t = compute_scale_and_shift(pred, gt)
        np.testing.assert_allclose(s, s_true, rtol=1e-5)
        np.testing.assert_allclose(t, t_true, rtol=1e-5)
        np.testing.assert_allclose(aligned, gt, rtol=1e-5)

    def test_identity_when_already_aligned(self):
        """If pred == gt, fitted params should be s=1, t=0."""
        gt = np.random.rand(16, 16).astype(np.float32) * 100 + 1
        aligned, s, t = compute_scale_and_shift(gt.copy(), gt)
        np.testing.assert_allclose(s, 1.0, atol=1e-4)
        np.testing.assert_allclose(t, 0.0, atol=1e-3)
        np.testing.assert_allclose(aligned, gt, rtol=1e-5)

    def test_normalized_01_to_metric(self):
        """Typical case: pred in [0, 1], GT in metres."""
        rng = np.random.RandomState(42)
        gt = rng.uniform(5.0, 80.0, size=(64, 64)).astype(np.float32)
        # Simulate model normalization
        pred = ((gt - gt.min()) / (gt.max() - gt.min())).astype(np.float32)

        aligned, s, t = compute_scale_and_shift(pred, gt)
        np.testing.assert_allclose(aligned, gt, rtol=1e-4)
        assert s > 0  # positive scale

    def test_normalized_neg1_pos1_to_metric(self):
        """Pred in [-1, 1], GT in metres."""
        rng = np.random.RandomState(42)
        gt = rng.uniform(5.0, 80.0, size=(64, 64)).astype(np.float32)
        pred = (2.0 * (gt - gt.min()) / (gt.max() - gt.min()) - 1.0).astype(
            np.float32
        )

        aligned, s, t = compute_scale_and_shift(pred, gt)
        np.testing.assert_allclose(aligned, gt, rtol=1e-4)

    def test_inverted_depth(self):
        """Model outputs inverted depth (far=1, near=0) -> negative scale."""
        gt = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
        pred = np.array([[1.0, 0.75, 0.5, 0.25]], dtype=np.float32)  # inverted

        aligned, s, t = compute_scale_and_shift(pred, gt)
        assert s < 0  # negative scale for inversion
        np.testing.assert_allclose(aligned, gt, rtol=1e-5)

    def test_valid_mask_excludes_pixels(self):
        """Only masked pixels contribute to the fit."""
        gt = np.array([[10.0, 20.0], [30.0, 999.0]], dtype=np.float32)
        pred = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        mask = np.array([[True, True], [True, False]], dtype=bool)

        aligned, s, t = compute_scale_and_shift(pred, gt, valid_mask=mask)
        # The fit should match the 3 valid pixels, not the outlier
        np.testing.assert_allclose(aligned[0, 0], gt[0, 0], rtol=1e-4)
        np.testing.assert_allclose(aligned[0, 1], gt[0, 1], rtol=1e-4)
        np.testing.assert_allclose(aligned[1, 0], gt[1, 0], rtol=1e-4)

    def test_max_gt_percentile_ignores_far_depth_outliers(self):
        """Upper-tail GT trimming keeps residual sky leaks out of the fit."""
        gt = np.linspace(5.0, 50.0, num=40 * 40, dtype=np.float32).reshape(40, 40)
        pred = ((gt - 3.0) / 7.0).astype(np.float32)

        gt[:2, :] = 1000.0
        pred[:2, :] = 0.25
        valid_mask = np.ones_like(gt, dtype=bool)

        aligned_untrimmed, _, _ = compute_scale_and_shift(
            pred, gt, valid_mask=valid_mask
        )
        aligned_trimmed, s, t = compute_scale_and_shift(
            pred,
            gt,
            valid_mask=valid_mask,
            max_gt_percentile=95.0,
        )

        np.testing.assert_allclose(s, 7.0, rtol=1e-4)
        np.testing.assert_allclose(t, 3.0, rtol=1e-4)
        np.testing.assert_allclose(aligned_trimmed[2:, :], gt[2:, :], rtol=1e-4)

        untrimmed_mae = float(np.mean(np.abs(aligned_untrimmed[2:, :] - gt[2:, :])))
        trimmed_mae = float(np.mean(np.abs(aligned_trimmed[2:, :] - gt[2:, :])))
        assert trimmed_mae < untrimmed_mae * 0.05

    def test_sky_mask_excluded_from_fit(self):
        """Simulates sky pixels (large GT depth) excluded via mask."""
        rng = np.random.RandomState(123)
        gt = rng.uniform(5.0, 50.0, size=(32, 32)).astype(np.float32)
        # Top rows are "sky" with bogus depth
        gt[:8, :] = 1000.0

        pred = ((gt - 5.0) / 45.0).astype(np.float32)  # normalized
        pred[:8, :] = 0.99  # sky region has arbitrary pred values

        sky_valid = np.ones((32, 32), dtype=bool)
        sky_valid[:8, :] = False  # exclude sky

        fit_mask = (gt > 0) & np.isfinite(gt) & np.isfinite(pred) & sky_valid
        aligned, s, t = compute_scale_and_shift(pred, gt, valid_mask=fit_mask)

        # Non-sky pixels should be well-aligned
        non_sky = gt[8:, :]
        aligned_non_sky = aligned[8:, :]
        np.testing.assert_allclose(aligned_non_sky, non_sky, rtol=1e-3)

    def test_default_mask_excludes_zeros_and_nans(self):
        """When no mask is given, zeros and NaN in GT are auto-excluded."""
        gt = np.array([[0.0, 10.0], [20.0, np.nan]], dtype=np.float32)
        pred = np.array([[0.5, 0.1], [0.2, 0.3]], dtype=np.float32)

        aligned, s, t = compute_scale_and_shift(pred, gt)
        # Should fit on the 2 valid pixels: (0.1, 10) and (0.2, 20)
        np.testing.assert_allclose(s, 100.0, rtol=1e-4)
        np.testing.assert_allclose(t, 0.0, atol=1e-3)

    def test_constant_pred_degeneracy(self):
        """Constant prediction -> aligned output is constant at mean(gt)."""
        gt = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        pred = np.full_like(gt, 0.5)

        aligned, s, t = compute_scale_and_shift(pred, gt)
        # Best fit maps constant pred to mean(gt); aligned should be uniform
        expected = np.mean(gt)
        np.testing.assert_allclose(aligned, expected, atol=1e-4)

    def test_too_few_valid_pixels_returns_unchanged(self):
        """< 2 valid pixels -> returns pred unchanged, s=1, t=0."""
        gt = np.array([[0.0, np.nan], [0.0, 0.0]], dtype=np.float32)
        pred = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)

        aligned, s, t = compute_scale_and_shift(pred, gt)
        np.testing.assert_array_equal(aligned, pred)
        assert s == 1.0
        assert t == 0.0

    def test_output_dtype_float32(self):
        """Aligned output should always be float32."""
        gt = np.random.rand(8, 8).astype(np.float32) * 50 + 1
        pred = np.random.rand(8, 8).astype(np.float32)

        aligned, _, _ = compute_scale_and_shift(pred, gt)
        assert aligned.dtype == np.float32

    def test_large_image_performance(self):
        """Runs on a realistic-size image without error."""
        rng = np.random.RandomState(0)
        gt = rng.uniform(1.0, 300.0, size=(375, 1242)).astype(np.float32)
        pred = ((gt - gt.min()) / (gt.max() - gt.min())).astype(np.float32)

        aligned, s, t = compute_scale_and_shift(pred, gt)
        assert aligned.shape == (375, 1242)
        np.testing.assert_allclose(aligned, gt, rtol=1e-4)


class TestDepthMetadataSparse:
    """``sparse`` comes from the head, else from the loader's annotation."""

    @staticmethod
    def _dataset(head, annotation):
        from types import SimpleNamespace

        return SimpleNamespace(
            get_modality_metadata=lambda name: head,
            get_loader_meta=lambda name: annotation,
        )

    def test_loader_annotation_fills_a_silent_head(self):
        from euler_eval.data import get_depth_metadata

        dataset = self._dataset({"radial_depth": False}, {"meta": {"sparse": True}})
        assert get_depth_metadata(dataset, "gt_depth")["sparse"] is True

    def test_head_wins(self):
        from euler_eval.data import get_depth_metadata

        dataset = self._dataset({"sparse": False}, {"meta": {"sparse": True}})
        assert get_depth_metadata(dataset)["sparse"] is False

    def test_datasets_without_loader_annotations(self):
        from types import SimpleNamespace

        from euler_eval.data import get_depth_metadata

        dataset = SimpleNamespace(get_modality_metadata=lambda name: {})
        assert get_depth_metadata(dataset)["sparse"] is False


class TestSourceGrid:
    """A prediction's recorded ``source_grid`` places it exactly on the GT grid."""

    @staticmethod
    def _grid(affine, size=(8, 12)):
        from euler_eval.data import SourceGrid

        return SourceGrid.from_value({"size": list(size), "affine": affine})

    def test_parse_rejects_what_is_not_a_crop_or_resize(self):
        from euler_eval.data import SourceGrid

        for bad in ({"size": [8, 12]}, {"size": [8, 12], "affine": [[1, 0.2, 0], [0, 1, 0]]},
                    {"size": [0, 12], "affine": [[1, 0, 0], [0, 1, 0]]}):
            with pytest.raises(ValueError):
                SourceGrid.from_value(bad)

    def test_relation(self):
        grid = self._grid([[1, 0, -2], [0, 1, -1]])
        assert grid.relation((6, 8)) == ("crop", (1, 2))
        assert self._grid([[1, 0, 0], [0, 1, 0]]).relation((8, 12)) == ("none", None)
        assert self._grid([[0.5, 0, -0.25], [0, 0.5, -0.25]]).relation((4, 6)) == ("resize", None)
        # Resize to 8x6 (s = 0.5 horizontally only after the crop) is neither.
        assert self._grid([[0.5, 0, -1.25], [0, 0.5, -0.25]]).relation((4, 4))[0] == "warp"

    def test_recorded_centre_crop_trims_the_gt_at_its_origin(self):
        from euler_eval.data import align_to_ground_truth, align_to_prediction

        gt = np.arange(96, dtype=np.float32).reshape(8, 12) + 1
        pred = gt[1:7, 2:10].copy()
        grid = self._grid([[1, 0, -2], [0, 1, -1]])
        p, g, method = align_to_ground_truth(pred, gt, spatial_alignment=grid)
        assert method == "crop"
        np.testing.assert_array_equal(g, pred)
        np.testing.assert_array_equal(align_to_prediction(gt, pred, spatial_alignment=grid), pred)
        # The shapes alone would have called this a top-left crop.
        _, guessed, _ = align_to_ground_truth(pred, gt, spatial_alignment="crop")
        assert not np.array_equal(guessed, pred)

    def test_resize_then_crop_is_warped_exactly(self):
        import cv2

        from euler_eval.data import align_to_ground_truth

        # A linear ramp survives bilinear resampling exactly, away from edges.
        yy, xx = np.mgrid[0:40, 0:60].astype(np.float32)
        gt = 10 + 0.5 * xx + 0.25 * yy
        resized = cv2.resize(gt, (30, 20), interpolation=cv2.INTER_LINEAR)  # s = 0.5
        pred = resized[2:18, 3:27]  # then a crop at (2, 3)
        # Composite source -> prediction mapping: crop @ resize.
        grid = self._grid([[0.5, 0, -0.25 - 3], [0, 0.5, -0.25 - 2]], size=(40, 60))
        warped, g, method = align_to_ground_truth(pred, gt, spatial_alignment=grid)
        assert method == "warp_prediction"
        covered = np.isfinite(warped)
        assert covered[10:30, 10:50].all() and not covered[:2].any() and not covered[:, :4].any()
        inner = (slice(8, 32), slice(10, 50))
        np.testing.assert_allclose(warped[inner], gt[inner], atol=1e-3)

    def test_sparse_gt_is_forward_mapped_through_the_grid(self):
        from euler_eval.data import align_to_prediction

        gt = np.zeros((8, 12), dtype=np.float32)
        gt[2, 4], gt[3, 5], gt[6, 10] = 7.0, 3.0, 9.0
        # A 2x shrink then a crop of one row and one column.
        grid = self._grid([[0.5, 0, -0.25 - 1], [0, 0.5, -0.25 - 1]])
        pred = np.zeros((3, 5), dtype=np.float32)
        out = align_to_prediction(gt, pred, spatial_alignment=grid, sparse=True)
        # (2,4) and (3,5) land on pixel (0,1); the nearer surface (3 m) wins.
        assert out[0, 1] == 3.0 and out[2, 4] == 9.0
        assert np.count_nonzero(out) == 2 and set(out[out > 0]) <= {3.0, 7.0, 9.0}

    def test_intrinsics_move_with_the_grid(self):
        from euler_eval.data import prediction_intrinsics

        K = np.array([[100.0, 0, 5.5], [0, 100.0, 3.5], [0, 0, 1]])
        grid = self._grid([[0.5, 0, -1.25], [0, 0.5, -1.25]])
        np.testing.assert_allclose(
            prediction_intrinsics(K, grid),
            [[50, 0, 0.5 * 5.5 - 1.25], [0, 50, 0.5 * 3.5 - 1.25], [0, 0, 1]],
        )
        assert prediction_intrinsics(K, "auto") is K

    def test_auto_reads_the_file_then_the_head_and_explicit_modes_win(self):
        from euler_eval.data import SourceGrid, sample_spatial_alignment

        head = self._grid([[1, 0, 0], [0, 1, 0]])
        file_value = {"size": [8, 12], "affine": [[1, 0, -2], [0, 1, -1]]}
        sample = {"attributes": {"pred": {"source_grid": file_value}}}
        assert sample_spatial_alignment(sample, "auto", head) == SourceGrid.from_value(file_value)
        assert sample_spatial_alignment({"attributes": {"pred": {}}}, "auto", head) is head
        assert sample_spatial_alignment({}, "auto") == "auto"
        assert sample_spatial_alignment(sample, "resize", head) == "resize"

    def test_a_grid_for_another_source_size_falls_back_to_guessing(self):
        from euler_eval.data import classify_spatial_alignment

        grid = self._grid([[1, 0, 0], [0, 1, 0]], size=(20, 20))
        with pytest.warns(UserWarning, match="source_grid"):
            assert classify_spatial_alignment(8, 12, 8, 8, mode=grid) == "vae_crop"


class TestInvalidSentinel:
    def test_declared_sentinel_comes_from_the_head_then_the_loader(self):
        from types import SimpleNamespace

        from euler_eval.data import declared_invalid_sentinel

        def dataset(head, annotation=None):
            return SimpleNamespace(
                get_modality_metadata=lambda name: head,
                get_loader_meta=lambda name: annotation,
            )

        loader = {"meta": {"invalid": {"non_finite": False, "sentinel": 0}}}
        assert declared_invalid_sentinel(dataset({}, loader), "gt") == 0.0
        head = {"invalid": {"non_finite": True, "sentinel": None}}
        assert declared_invalid_sentinel(dataset(head, loader), "gt") is None
        assert declared_invalid_sentinel(dataset({"invalid": {"sentinel": 65535}}), "gt") == 65535.0
        assert declared_invalid_sentinel(dataset({}), "gt") is None

    def test_mask_invalid_sentinel(self):
        from euler_eval.data import mask_invalid_sentinel

        values = np.array([[1.0, 65535.0], [0.0, 2.0]], dtype=np.float32)
        gt = mask_invalid_sentinel(values, 65535.0, fill=0.0)
        np.testing.assert_array_equal(gt, [[1, 0], [0, 2]])
        assert values[0, 1] == 65535.0  # the input is not modified
        pred = mask_invalid_sentinel(values, 0.0, fill=np.nan)
        assert np.isnan(pred[1, 0]) and np.isfinite(pred).sum() == 3
        assert mask_invalid_sentinel(values, None, fill=0.0) is values
