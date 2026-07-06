"""Tests for euler-loading modality construction in dataset builders."""

from types import SimpleNamespace

from euler_eval import data


class _CapturedDataset:
    def __init__(self, *, modalities, hierarchical_modalities=None):
        self.modalities = modalities
        self.hierarchical_modalities = hierarchical_modalities or {}


def _install_captured_dataset(monkeypatch):
    monkeypatch.setattr(data, "MultiModalDataset", _CapturedDataset)


def _assert_modality(
    modality, *, key, split=None, used_as=None, loader=None, scope=None
):
    assert modality.metadata_scope == (scope or key)
    assert modality.modality_type == key
    assert modality.split == split
    assert modality.used_as == used_as
    if loader is not None:
        assert modality.loader is loader


def test_depth_builder_sets_modality_scopes(monkeypatch):
    _install_captured_dataset(monkeypatch)

    def sky_loader(path, meta=None):
        return None

    monkeypatch.setattr(
        data,
        "_resolve_sky_mask_loader",
        lambda path, *, modality_key="segmentation": sky_loader,
    )

    dataset = data.build_depth_eval_dataset(
        gt_depth_path="/datasets/shared",
        pred_depth_path="/predictions/shared",
        calibration_path="/datasets/calibration",
        segmentation_path="/datasets/segmentation",
        gt_depth_split="test",
        pred_depth_split="val",
        calibration_split="calib",
        segmentation_split="seg",
    )

    _assert_modality(dataset.modalities["gt"], key="depth", split="test")
    _assert_modality(
        dataset.modalities["pred"],
        key="depth",
        split="val",
        used_as="output",
    )
    _assert_modality(
        dataset.hierarchical_modalities["calibration"],
        key="calibration",
        split="calib",
    )
    _assert_modality(
        dataset.hierarchical_modalities["segmentation"],
        key="segmentation",
        split="seg",
        loader=sky_loader,
    )


def test_sparse_depth_builder_sets_projection_modality_scopes(monkeypatch):
    _install_captured_dataset(monkeypatch)

    dataset = data.build_sparse_depth_eval_dataset(
        gt_sparse_depth_path="/datasets/shared",
        pred_depth_path="/predictions/shared",
        intrinsics_path="/datasets/shared",
        camera_extrinsics_path="/datasets/shared",
        gt_sparse_depth_split="test",
        pred_depth_split="val",
        intrinsics_split="cam",
        camera_extrinsics_split="pose",
    )

    _assert_modality(dataset.modalities["gt"], key="sparse_depth", split="test")
    _assert_modality(
        dataset.modalities["pred"],
        key="depth",
        split="val",
        used_as="output",
    )
    _assert_modality(
        dataset.hierarchical_modalities["intrinsics"],
        key="intrinsics",
        split="cam",
    )
    _assert_modality(
        dataset.hierarchical_modalities["camera_extrinsics"],
        key="camera_extrinsics",
        split="pose",
    )
    assert "segmentation" not in dataset.hierarchical_modalities


def test_sparse_depth_builder_adds_optional_lidar_extrinsics(monkeypatch):
    _install_captured_dataset(monkeypatch)

    dataset = data.build_sparse_depth_eval_dataset(
        gt_sparse_depth_path="/datasets/shared",
        pred_depth_path="/predictions/shared",
        intrinsics_path="/datasets/shared",
        camera_extrinsics_path="/datasets/shared",
        lidar_extrinsics_path="/datasets/shared",
        lidar_extrinsics_split="pose",
    )

    _assert_modality(
        dataset.hierarchical_modalities["lidar_extrinsics"],
        key="camera_extrinsics",
        split="pose",
        scope="lidar_extrinsics",
    )


def test_points_3d_sparse_builder_sets_pointmap_and_projection_scopes(monkeypatch):
    _install_captured_dataset(monkeypatch)

    dataset = data.build_points_3d_sparse_eval_dataset(
        gt_sparse_depth_path="/datasets/shared",
        pred_points_3d_path="/predictions/shared",
        intrinsics_path="/datasets/shared",
        camera_extrinsics_path="/datasets/shared",
        lidar_extrinsics_path="/datasets/shared",
        gt_sparse_depth_split="test",
        pred_points_3d_split="val",
        lidar_extrinsics_split="pose",
    )

    # GT stays a sparse pointcloud; the prediction is loaded as a point map.
    _assert_modality(dataset.modalities["gt"], key="sparse_depth", split="test")
    _assert_modality(
        dataset.modalities["pred"],
        key="points_3d",
        split="val",
        used_as="output",
    )
    _assert_modality(
        dataset.hierarchical_modalities["intrinsics"], key="intrinsics"
    )
    _assert_modality(
        dataset.hierarchical_modalities["camera_extrinsics"],
        key="camera_extrinsics",
    )
    _assert_modality(
        dataset.hierarchical_modalities["lidar_extrinsics"],
        key="camera_extrinsics",
        split="pose",
        scope="lidar_extrinsics",
    )


def test_sparse_depth_builder_can_load_relative_depth_prediction_scope(monkeypatch):
    _install_captured_dataset(monkeypatch)

    dataset = data.build_sparse_depth_eval_dataset(
        gt_sparse_depth_path="/datasets/shared",
        pred_depth_path="/predictions/shared",
        intrinsics_path="/datasets/shared",
        camera_extrinsics_path="/datasets/shared",
        pred_depth_metadata_scope="relative_depth",
    )

    _assert_modality(
        dataset.modalities["pred"],
        key="depth",
        scope="relative_depth",
        used_as="output",
    )


def test_sparse_depth_builder_uses_semantic_segmentation_scope(monkeypatch):
    _install_captured_dataset(monkeypatch)
    calls = []

    def sky_loader(path, meta=None):
        return None

    def resolve_sky_mask_loader(path, *, modality_key="segmentation"):
        calls.append((path, modality_key))
        return sky_loader

    monkeypatch.setattr(
        data,
        "_resolve_sky_mask_loader",
        resolve_sky_mask_loader,
    )

    dataset = data.build_sparse_depth_eval_dataset(
        gt_sparse_depth_path="/datasets/shared",
        pred_depth_path="/predictions/shared",
        intrinsics_path="/datasets/shared",
        camera_extrinsics_path="/datasets/shared",
        segmentation_path="/datasets/shared",
        segmentation_modality_key="semantic_segmentation",
    )

    assert calls == [("/datasets/shared", "semantic_segmentation")]
    _assert_modality(
        dataset.hierarchical_modalities["segmentation"],
        key="semantic_segmentation",
        loader=sky_loader,
    )


def test_rgb_builder_sets_rgb_and_auxiliary_modality_scopes(monkeypatch):
    _install_captured_dataset(monkeypatch)

    dataset = data.build_rgb_eval_dataset(
        gt_rgb_path="/datasets/shared",
        pred_rgb_path="/predictions/shared",
        gt_depth_path="/datasets/shared",
        gt_rgb_split="test",
        pred_rgb_split="val",
        gt_depth_split="depth",
    )

    _assert_modality(dataset.modalities["gt"], key="rgb", split="test")
    _assert_modality(
        dataset.modalities["pred"],
        key="rgb",
        split="val",
        used_as="output",
    )
    _assert_modality(dataset.modalities["gt_depth"], key="depth", split="depth")


def test_rays_builder_sets_rays_modality_scopes(monkeypatch):
    _install_captured_dataset(monkeypatch)

    dataset = data.build_rays_eval_dataset(
        gt_rays_path="/datasets/shared",
        pred_rays_path="/predictions/shared",
        calibration_path="/datasets/calibration",
        gt_rays_split="test",
        pred_rays_split="val",
        calibration_split="calib",
    )

    _assert_modality(dataset.modalities["gt"], key="rays", split="test")
    _assert_modality(
        dataset.modalities["pred"],
        key="rays",
        split="val",
        used_as="output",
    )
    _assert_modality(
        dataset.hierarchical_modalities["calibration"],
        key="calibration",
        split="calib",
    )


def test_sky_mask_loader_resolution_uses_segmentation_scope(monkeypatch):
    calls = []

    def fake_index_dataset_from_path(path, **kwargs):
        calls.append((path, kwargs))
        return {"euler_loading": {"loader": "vkitti2"}}

    def sky_mask(path, meta=None):
        return None

    monkeypatch.setattr(data, "index_dataset_from_path", fake_index_dataset_from_path)
    monkeypatch.setattr(
        data,
        "resolve_loader_module",
        lambda name: SimpleNamespace(sky_mask=sky_mask),
    )

    assert data._resolve_sky_mask_loader("/datasets/shared") is sky_mask
    assert calls == [
        ("/datasets/shared", {"metadata_scope": "segmentation"}),
    ]


def test_sky_mask_loader_resolution_strips_inline_split(monkeypatch):
    calls = []

    def fake_index_dataset_from_path(path, **kwargs):
        calls.append((path, kwargs))
        return {"euler_loading": {"loader": "vkitti2"}}

    def sky_mask(path, meta=None):
        return None

    monkeypatch.setattr(data, "index_dataset_from_path", fake_index_dataset_from_path)
    monkeypatch.setattr(
        data,
        "resolve_loader_module",
        lambda name: SimpleNamespace(sky_mask=sky_mask),
    )

    assert data._resolve_sky_mask_loader("/datasets/shared.zip:fog_day") is sky_mask
    assert calls == [
        ("/datasets/shared.zip", {"metadata_scope": "segmentation"}),
    ]


def test_sky_mask_loader_resolution_accepts_semantic_segmentation_scope(monkeypatch):
    calls = []

    def fake_index_dataset_from_path(path, **kwargs):
        calls.append((path, kwargs))
        return {"euler_loading": {"loader": "muses"}}

    def sky_mask(path, meta=None):
        return None

    monkeypatch.setattr(data, "index_dataset_from_path", fake_index_dataset_from_path)
    monkeypatch.setattr(
        data,
        "resolve_loader_module",
        lambda name: SimpleNamespace(sky_mask=sky_mask),
    )

    assert (
        data._resolve_sky_mask_loader(
            "/datasets/shared",
            modality_key="semantic_segmentation",
        )
        is sky_mask
    )
    assert calls == [
        ("/datasets/shared", {"metadata_scope": "semantic_segmentation"}),
    ]


def test_modality_parses_inline_split_and_scope():
    modality = data._modality(
        path="/datasets/frame_camera_trainvaltest.zip:fog_day#scope=intrinsics",
        modality_key="intrinsics",
    )

    assert modality.path == "/datasets/frame_camera_trainvaltest.zip"
    assert modality.split == "fog_day"
    assert modality.metadata_scope == "intrinsics"
