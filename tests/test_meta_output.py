"""Tests for the structured meta block in eval.json output."""

from euler_eval.cli import (
    _clean_metric_tree,
    _DEPTH_EVAL_AXES,
    _DEPTH_EVAL_DESCRIPTIONS,
    _RGB_EVAL_AXES,
    _RGB_EVAL_DESCRIPTIONS,
    _RAYS_EVAL_AXES,
    _RAYS_EVAL_DESCRIPTIONS,
)


class TestMetaBlockStructure:
    """Verify _clean_metric_tree handles meta blocks correctly."""

    def test_meta_block_preserved(self):
        """Core meta fields survive cleaning."""
        meta = {
            "version": "1.4.1",
            "modality": "depth",
            "device": "cuda",
            "gt": {
                "path": "/data/gt",
                "split": "test",
                "dimensions": {"height": 1024, "width": 2048},
            },
            "pred": {
                "path": "/data/pred",
                "dimensions": {"height": 370, "width": 780},
            },
            "spatial_alignment": {
                "method": "resize",
                "evaluated_dimensions": {"height": 370, "width": 780},
            },
            "modality_params": {
                "radial_depth": True,
                "scale_to_meters": 1.0,
            },
            "eval_params": {
                "sky_masking": False,
                "depth_alignment_mode": "auto_affine",
                "batch_size": 16,
                "num_workers": 4,
            },
        }
        cleaned = _clean_metric_tree(meta)
        assert cleaned["version"] == "1.4.1"
        assert cleaned["gt"]["dimensions"]["height"] == 1024
        assert cleaned["pred"]["dimensions"]["width"] == 780
        assert cleaned["spatial_alignment"]["method"] == "resize"
        assert cleaned["modality_params"]["radial_depth"] is True
        assert cleaned["eval_params"]["sky_masking"] is False

    def test_none_split_pruned(self):
        """None-valued splits are pruned by _clean_metric_tree (expected)."""
        meta = {
            "gt": {"path": "/data/gt", "split": None},
            "pred": {"path": "/data/pred", "split": "test"},
        }
        cleaned = _clean_metric_tree(meta)
        assert "split" not in cleaned["gt"]
        assert cleaned["pred"]["split"] == "test"

    def test_empty_eval_params_pruned(self):
        """Empty eval_params dict is pruned."""
        meta = {
            "version": "1.0.0",
            "eval_params": {},
        }
        cleaned = _clean_metric_tree(meta)
        assert "eval_params" not in cleaned

    def test_meta_in_full_save_dict(self):
        """meta sits alongside metricSet and dataset_info without conflict."""
        save_dict = {
            "metricSet": {
                "metricNamespace": "depth.eval",
                "producerKey": "euler-eval",
            },
            "dataset_info": {"num_pairs": 10},
            "meta": {
                "version": "1.4.1",
                "gt": {"dimensions": {"height": 100, "width": 200}},
                "pred": {"dimensions": {"height": 50, "width": 100}},
                "spatial_alignment": {"method": "resize"},
            },
            "depth": {"eval": {"raw": {}, "aligned": {}}},
        }
        cleaned = _clean_metric_tree(save_dict)
        assert "meta" in cleaned
        assert "metricSet" in cleaned
        assert "dataset_info" in cleaned
        assert cleaned["meta"]["gt"]["dimensions"]["height"] == 100

    def test_rgb_meta_with_rgb_range(self):
        """RGB meta includes rgb_range in modality_params."""
        meta = {
            "version": "1.4.1",
            "modality": "rgb",
            "modality_params": {"rgb_range": [0.0, 1.0]},
        }
        cleaned = _clean_metric_tree(meta)
        assert cleaned["modality_params"]["rgb_range"] == [0.0, 1.0]

    def test_rays_meta_with_fov_domain(self):
        """Rays meta includes fov_domain and threshold_deg."""
        meta = {
            "version": "1.4.1",
            "modality": "rays",
            "modality_params": {
                "fov_domain": "lfov",
                "threshold_deg": 20.0,
            },
        }
        cleaned = _clean_metric_tree(meta)
        assert cleaned["modality_params"]["fov_domain"] == "lfov"
        assert cleaned["modality_params"]["threshold_deg"] == 20.0


class TestAxisDeclarations:
    """Verify axis declarations follow the metric-namespacing convention."""

    def test_depth_axes_structure(self):
        """depth.eval declares alignment (required) and category (optional)."""
        assert "alignment" in _DEPTH_EVAL_AXES
        assert "category" in _DEPTH_EVAL_AXES

        alignment = _DEPTH_EVAL_AXES["alignment"]
        assert alignment["position"] == 0
        assert alignment["optional"] is False
        assert "raw" in alignment["values"]
        assert "aligned" in alignment["values"]

        category = _DEPTH_EVAL_AXES["category"]
        assert category["position"] == 1
        assert category["optional"] is True
        assert set(category["values"]) == {
            "image_quality",
            "depth_metrics",
            "geometric_metrics",
        }

    def test_rgb_axes_structure(self):
        """rgb.eval declares a single optional category axis."""
        assert "category" in _RGB_EVAL_AXES
        assert len(_RGB_EVAL_AXES) == 1

        category = _RGB_EVAL_AXES["category"]
        assert category["position"] == 0
        assert category["optional"] is True
        assert "image_quality" in category["values"]
        assert "edge_f1" in category["values"]
        assert "tail_errors" in category["values"]
        assert "high_frequency" in category["values"]
        assert "depth_binned_photometric" in category["values"]

    def test_rays_axes_empty(self):
        """rays.eval has no axes (flat namespace)."""
        assert _RAYS_EVAL_AXES == {}

    def test_axis_positions_are_contiguous(self):
        """Axis positions start at 0 and are contiguous."""
        for axes in (_DEPTH_EVAL_AXES, _RGB_EVAL_AXES):
            if not axes:
                continue
            positions = sorted(a["position"] for a in axes.values())
            assert positions == list(range(len(positions)))

    def test_axis_values_are_nonempty_strings(self):
        """Every axis has at least one value, all lowercase strings."""
        for axes in (_DEPTH_EVAL_AXES, _RGB_EVAL_AXES, _RAYS_EVAL_AXES):
            for name, decl in axes.items():
                assert len(decl["values"]) >= 1, f"axis {name} has no values"
                for v in decl["values"]:
                    assert isinstance(v, str) and v == v.lower(), (
                        f"axis {name} value {v!r} must be lowercase string"
                    )


class TestMetricDescriptions:
    """Verify metric descriptions have valid structure and directions."""

    def _check_descriptions(self, descriptions):
        valid_scales = {"linear", "log", "percentage", "binary"}
        for key, desc in descriptions.items():
            assert isinstance(key, str) and len(key) > 0
            if "isHigherBetter" in desc:
                assert isinstance(desc["isHigherBetter"], bool)
            if "scale" in desc:
                assert desc["scale"] in valid_scales, (
                    f"{key}: invalid scale {desc['scale']!r}"
                )
            if "min" in desc:
                assert isinstance(desc["min"], (int, float))
            if "max" in desc:
                assert isinstance(desc["max"], (int, float))
            if "displayName" in desc:
                assert isinstance(desc["displayName"], str)
            if "unit" in desc:
                assert isinstance(desc["unit"], str)

    def test_depth_descriptions_valid(self):
        self._check_descriptions(_DEPTH_EVAL_DESCRIPTIONS)

    def test_rgb_descriptions_valid(self):
        self._check_descriptions(_RGB_EVAL_DESCRIPTIONS)

    def test_rays_descriptions_valid(self):
        self._check_descriptions(_RAYS_EVAL_DESCRIPTIONS)

    def test_depth_key_metrics_have_direction(self):
        """Core depth metrics declare isHigherBetter."""
        for key in ("psnr", "ssim", "lpips", "absrel.median", "rmse.median"):
            assert "isHigherBetter" in _DEPTH_EVAL_DESCRIPTIONS[key], (
                f"depth description {key} missing isHigherBetter"
            )

    def test_rgb_key_metrics_have_direction(self):
        """Core RGB metrics declare isHigherBetter."""
        for key in ("psnr", "ssim", "lpips", "f1"):
            assert "isHigherBetter" in _RGB_EVAL_DESCRIPTIONS[key]

    def test_rays_key_metrics_have_direction(self):
        """Core rays metrics declare isHigherBetter."""
        for key in ("rho_a.mean", "angular_error.mean_angle"):
            assert "isHigherBetter" in _RAYS_EVAL_DESCRIPTIONS[key]

    def test_metricset_envelope_includes_axes_and_descriptions(self):
        """metricSet dict includes axes and metricDescriptions fields."""
        depth_metric_set = {
            "metricNamespace": "depth.eval",
            "producerKey": "euler-eval",
            "producerVersion": "1.7.0",
            "sourceKind": "computed",
            "metadata": {},
            "axes": _DEPTH_EVAL_AXES,
            "metricDescriptions": _DEPTH_EVAL_DESCRIPTIONS,
        }
        assert "axes" in depth_metric_set
        assert "metricDescriptions" in depth_metric_set
        assert depth_metric_set["axes"]["alignment"]["position"] == 0
        assert "psnr" in depth_metric_set["metricDescriptions"]
