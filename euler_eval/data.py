"""Data loading bridge between euler_loading and euler-eval.

Provides utilities to build MultiModalDataset instances from config
and convert loaded tensors to the numpy formats expected by euler-eval
metrics for depth, RGB, rays, and points_3d.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from ds_crawler import get_dataset_contract, index_dataset_from_path
from euler_loading import Modality, MultiModalDataset, resolve_loader_module

from .config_paths import build_modality, parse_modality_path
from .metrics.utils import convert_planar_to_radial

# ---------------------------------------------------------------------------
# Tensor → numpy conversions
# ---------------------------------------------------------------------------


def to_numpy_depth(data: Any) -> np.ndarray:
    """Convert depth data to ``(H, W)`` float32 numpy array.

    Accepts torch tensors ``(1, H, W)`` or ``(H, W)``, or numpy arrays.
    """
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data, dtype=np.float32)

    if arr.ndim == 2:
        return arr.astype(np.float32)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(
                f"Unsupported depth shape {arr.shape}. Expected (H,W), (1,H,W), or (H,W,1)."
            )
        return arr.astype(np.float32)

    if arr.ndim == 4:
        if arr.shape[0] == 1 and arr.shape[1] == 1:
            arr = arr[0, 0]
        elif arr.shape[0] == 1 and arr.shape[-1] == 1:
            arr = arr[0, ..., 0]
        else:
            raise ValueError(
                f"Unsupported depth shape {arr.shape}. Expected a single-sample/single-channel tensor."
            )
        return arr.astype(np.float32)

    raise ValueError(f"Unsupported depth rank {arr.ndim} for shape {arr.shape}.")


def to_numpy_rgb(data: Any) -> np.ndarray:
    """Convert RGB data to ``(H, W, 3)`` float32 numpy array.

    Accepts torch tensors ``(3, H, W)`` or ``(H, W, 3)``, or numpy arrays.
    """
    tensor_input = isinstance(data, torch.Tensor)
    if tensor_input:
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data, dtype=np.float32)

    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 3:
        raise ValueError(
            f"Unsupported RGB shape {arr.shape}. Expected (H,W,3), (3,H,W), or single-sample variants."
        )

    # Tensor RGB from loaders is typically CHW.
    if tensor_input and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.shape[-1] == 3 and arr.shape[0] != 3:
        pass  # HWC
    elif arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))  # CHW
    elif arr.shape[0] == 3 and arr.shape[-1] == 3:
        raise ValueError(
            f"Ambiguous RGB layout for shape {arr.shape}. "
            "Please provide explicit HWC data or tensor CHW data."
        )
    else:
        raise ValueError(
            f"Unsupported RGB shape {arr.shape}. Expected channel dimension of size 3."
        )

    return arr.astype(np.float32)


def to_numpy_mask(data: Any) -> np.ndarray:
    """Convert mask data to ``(H, W)`` bool numpy array.

    Accepts torch tensors ``(1, H, W)`` or ``(H, W)``, or numpy arrays.
    """
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data)

    if arr.ndim == 2:
        return arr.astype(bool)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(
                f"Unsupported mask shape {arr.shape}. Expected (H,W), (1,H,W), or (H,W,1)."
            )
        return arr.astype(bool)

    if arr.ndim == 4:
        if arr.shape[0] == 1 and arr.shape[1] == 1:
            arr = arr[0, 0]
        elif arr.shape[0] == 1 and arr.shape[-1] == 1:
            arr = arr[0, ..., 0]
        else:
            raise ValueError(
                f"Unsupported mask shape {arr.shape}. Expected a single-sample/single-channel tensor."
            )
        return arr.astype(bool)

    raise ValueError(f"Unsupported mask rank {arr.ndim} for shape {arr.shape}.")


def to_numpy_intrinsics(data: Any) -> np.ndarray:
    """Convert intrinsics to ``(3, 3)`` float32 numpy array."""
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy().astype(np.float32)
    else:
        arr = np.asarray(data, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape != (3, 3):
        raise ValueError(f"Unsupported intrinsics shape {arr.shape}. Expected (3,3).")
    return arr


def to_numpy_extrinsics(data: Any) -> np.ndarray:
    """Convert camera extrinsics to a ``(4, 4)`` float32 transform matrix."""
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy().astype(np.float32)
    else:
        arr = np.asarray(data, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape == (3, 4):
        arr = np.vstack([arr, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)])
    if arr.shape != (4, 4):
        raise ValueError(
            f"Unsupported camera extrinsics shape {arr.shape}. Expected (4,4) or (3,4)."
        )
    return arr.astype(np.float32, copy=False)


def compose_sensor_to_camera_extrinsics(
    source_sensor_extrinsics: Any,
    camera_extrinsics: Any,
) -> np.ndarray:
    """Compose per-sensor poses into a source-to-camera transform.

    Both inputs are interpreted as sensor-to-common-frame transforms, e.g.
    ``T_common_from_lidar`` and ``T_common_from_camera``.  The returned matrix
    maps source sensor points directly into the camera frame:

    ``T_camera_from_source = inv(T_common_from_camera) @ T_common_from_source``.
    """
    source_to_common = to_numpy_extrinsics(source_sensor_extrinsics).astype(
        np.float64
    )
    camera_to_common = to_numpy_extrinsics(camera_extrinsics).astype(np.float64)
    common_to_camera = np.linalg.inv(camera_to_common)
    return (common_to_camera @ source_to_common).astype(np.float32)


def to_numpy_point_cloud(data: Any) -> np.ndarray:
    """Convert point-cloud data to ``(N, C)`` float32 numpy array.

    The first three columns are interpreted as ``x, y, z`` in metres. Extra
    columns such as intensity/ring/timestamp are preserved for callers that
    need them, but projection utilities use only ``xyz``.
    """
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy().astype(np.float32)
    else:
        arr = np.asarray(data, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(
            f"Unsupported point-cloud shape {arr.shape}. Expected (N,C) with C >= 3."
        )
    return arr.astype(np.float32, copy=False)


def to_numpy_directions(data: Any) -> np.ndarray:
    """Convert direction map data to ``(H, W, 3)`` float32 numpy array.

    Accepts torch tensors ``(3, H, W)`` or ``(H, W, 3)``, or numpy
    arrays.  The returned vectors are **not** re-normalized; callers
    should normalize if needed.
    """
    tensor_input = isinstance(data, torch.Tensor)
    if tensor_input:
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data, dtype=np.float32)

    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 3:
        raise ValueError(
            f"Unsupported direction map shape {arr.shape}. "
            "Expected (H,W,3), (3,H,W), or single-sample variants."
        )

    # Tensor from loaders is typically CHW.
    if tensor_input and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.shape[-1] == 3 and arr.shape[0] != 3:
        pass  # HWC
    elif arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))  # CHW
    elif arr.shape[0] == 3 and arr.shape[-1] == 3:
        raise ValueError(
            f"Ambiguous direction map layout for shape {arr.shape}. "
            "Please provide explicit HWC data or tensor CHW data."
        )
    else:
        raise ValueError(
            f"Unsupported direction map shape {arr.shape}. "
            "Expected channel dimension of size 3."
        )

    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Post-load depth processing
# ---------------------------------------------------------------------------


def process_depth(
    depth: np.ndarray,
    is_radial: bool,
    intrinsics_K: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply planar→radial conversion to a depth map.

    Args:
        depth: Depth array ``(H, W)`` in metres.
        is_radial: If True, depth is already radial (Euclidean).
                   If False and ``intrinsics_K`` is provided, convert.
        intrinsics_K: ``(3, 3)`` camera matrix. Required when
                      ``is_radial`` is False.

    Returns:
        Depth in metres, ``(H, W)`` float32.
    """
    if not is_radial and intrinsics_K is not None:
        intrinsics_dict = {
            "fx": float(intrinsics_K[0, 0]),
            "fy": float(intrinsics_K[1, 1]),
            "cx": float(intrinsics_K[0, 2]),
            "cy": float(intrinsics_K[1, 2]),
        }
        depth = convert_planar_to_radial(depth, intrinsics_dict)
    return depth.astype(np.float32)


def project_point_cloud_to_depth_map(
    point_cloud: np.ndarray,
    intrinsics_K: np.ndarray,
    camera_extrinsics: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Project a sparse point cloud into the camera image plane.

    Args:
        point_cloud: ``(N, C)`` array whose first three columns are source-frame
            ``x, y, z`` coordinates in metres.
        intrinsics_K: Camera intrinsics matrix.
        camera_extrinsics: Source-to-camera transform. For MUSES this is the
            ``lidar2rgb`` transform returned by the ``camera_extrinsics`` loader.
        image_shape: Target dense prediction shape as ``(height, width)``.

    Returns:
        ``(depth_map, valid_mask, metadata)``. ``depth_map`` stores radial
        camera depth in metres at projected pixels, choosing the nearest point
        by camera ``z`` when multiple points land on the same pixel.
    """
    height, width = image_shape
    depth = np.zeros((height, width), dtype=np.float32)
    valid_mask = np.zeros((height, width), dtype=bool)

    xyz = np.asarray(point_cloud[:, :3], dtype=np.float32)
    finite_xyz = np.isfinite(xyz).all(axis=1)
    if not finite_xyz.any():
        return depth, valid_mask, {
            "input_points": int(point_cloud.shape[0]),
            "finite_points": 0,
            "in_front_points": 0,
            "in_image_points": 0,
            "projected_pixels": 0,
        }

    xyz = xyz[finite_xyz]
    ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
    points_h = np.concatenate([xyz, ones], axis=1)
    camera_points = (camera_extrinsics @ points_h.T).T[:, :3]

    finite_camera = np.isfinite(camera_points).all(axis=1)
    in_front = finite_camera & (camera_points[:, 2] > 1e-8)
    if not in_front.any():
        return depth, valid_mask, {
            "input_points": int(point_cloud.shape[0]),
            "finite_points": int(finite_xyz.sum()),
            "in_front_points": 0,
            "in_image_points": 0,
            "projected_pixels": 0,
        }

    camera_points = camera_points[in_front]
    z = camera_points[:, 2]
    x = camera_points[:, 0]
    y = camera_points[:, 1]

    u = intrinsics_K[0, 0] * (x / z) + intrinsics_K[0, 2]
    v = intrinsics_K[1, 1] * (y / z) + intrinsics_K[1, 2]
    u_px = np.rint(u).astype(np.int64)
    v_px = np.rint(v).astype(np.int64)
    in_image = (u_px >= 0) & (u_px < width) & (v_px >= 0) & (v_px < height)
    if not in_image.any():
        return depth, valid_mask, {
            "input_points": int(point_cloud.shape[0]),
            "finite_points": int(finite_xyz.sum()),
            "in_front_points": int(in_front.sum()),
            "in_image_points": 0,
            "projected_pixels": 0,
        }

    camera_points = camera_points[in_image]
    z = z[in_image]
    u_px = u_px[in_image]
    v_px = v_px[in_image]
    radial_depth = np.linalg.norm(camera_points, axis=1).astype(np.float32)
    flat = v_px * width + u_px

    order = np.lexsort((z, flat))
    sorted_flat = flat[order]
    keep = np.concatenate([[True], sorted_flat[1:] != sorted_flat[:-1]])
    chosen = order[keep]
    chosen_flat = flat[chosen]

    depth.reshape(-1)[chosen_flat] = radial_depth[chosen]
    valid_mask.reshape(-1)[chosen_flat] = True

    return depth, valid_mask, {
        "input_points": int(point_cloud.shape[0]),
        "finite_points": int(finite_xyz.sum()),
        "in_front_points": int(in_front.sum()),
        "in_image_points": int(in_image.sum()),
        "projected_pixels": int(chosen_flat.size),
    }


def project_point_cloud_to_point_map(
    point_cloud: np.ndarray,
    intrinsics_K: np.ndarray,
    camera_extrinsics: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Project a sparse point cloud into the camera frame as a ``(H, W, 3)`` map.

    This is the 3D companion of :func:`project_point_cloud_to_depth_map`: instead
    of storing radial depth at each hit pixel it stores the full camera-frame
    ``(X, Y, Z)`` coordinate, so downstream code can compute 3D point metrics
    (Chamfer / F-score, per-pixel 3D error) against a ``points_3d`` prediction.
    When several points land on the same pixel the nearest one by camera ``z``
    is kept, matching the depth-map projector.

    Args:
        point_cloud: ``(N, C)`` array whose first three columns are source-frame
            ``x, y, z`` coordinates in metres.
        intrinsics_K: Camera intrinsics matrix.
        camera_extrinsics: Source-to-camera transform (e.g. MUSES ``lidar2rgb``).
        image_shape: Target dense prediction shape as ``(height, width)``.

    Returns:
        ``(point_map, valid_mask, metadata)``. ``point_map`` is a
        ``(H, W, 3)`` float32 array holding the camera-frame point at each hit
        pixel (``0`` elsewhere); ``valid_mask`` marks the hit pixels.  The
        ``metadata`` keys mirror :func:`project_point_cloud_to_depth_map`.
    """
    height, width = image_shape
    point_map = np.zeros((height, width, 3), dtype=np.float32)
    valid_mask = np.zeros((height, width), dtype=bool)

    empty_meta = {
        "input_points": int(point_cloud.shape[0]),
        "finite_points": 0,
        "in_front_points": 0,
        "in_image_points": 0,
        "projected_pixels": 0,
    }

    xyz = np.asarray(point_cloud[:, :3], dtype=np.float32)
    finite_xyz = np.isfinite(xyz).all(axis=1)
    if not finite_xyz.any():
        return point_map, valid_mask, empty_meta

    xyz = xyz[finite_xyz]
    ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
    points_h = np.concatenate([xyz, ones], axis=1)
    camera_points = (camera_extrinsics @ points_h.T).T[:, :3]

    finite_camera = np.isfinite(camera_points).all(axis=1)
    in_front = finite_camera & (camera_points[:, 2] > 1e-8)
    empty_meta["finite_points"] = int(finite_xyz.sum())
    if not in_front.any():
        return point_map, valid_mask, empty_meta

    camera_points = camera_points[in_front]
    z = camera_points[:, 2]
    x = camera_points[:, 0]
    y = camera_points[:, 1]

    u = intrinsics_K[0, 0] * (x / z) + intrinsics_K[0, 2]
    v = intrinsics_K[1, 1] * (y / z) + intrinsics_K[1, 2]
    u_px = np.rint(u).astype(np.int64)
    v_px = np.rint(v).astype(np.int64)
    in_image = (u_px >= 0) & (u_px < width) & (v_px >= 0) & (v_px < height)
    empty_meta["in_front_points"] = int(in_front.sum())
    if not in_image.any():
        return point_map, valid_mask, empty_meta

    camera_points = camera_points[in_image]
    z = z[in_image]
    u_px = u_px[in_image]
    v_px = v_px[in_image]
    flat = v_px * width + u_px

    # Keep the nearest point (smallest camera z) per pixel, matching the
    # depth-map projector's occlusion handling.
    order = np.lexsort((z, flat))
    sorted_flat = flat[order]
    keep = np.concatenate([[True], sorted_flat[1:] != sorted_flat[:-1]])
    chosen = order[keep]
    chosen_flat = flat[chosen]

    point_map.reshape(-1, 3)[chosen_flat] = camera_points[chosen].astype(np.float32)
    valid_mask.reshape(-1)[chosen_flat] = True

    return point_map, valid_mask, {
        "input_points": int(point_cloud.shape[0]),
        "finite_points": int(finite_xyz.sum()),
        "in_front_points": int(in_front.sum()),
        "in_image_points": int(in_image.sum()),
        "projected_pixels": int(chosen_flat.size),
    }


# ---------------------------------------------------------------------------
# Dimension alignment
# ---------------------------------------------------------------------------


#: How a GT/prediction resolution mismatch is reconciled; see
#: :func:`classify_spatial_alignment`.
SPATIAL_ALIGNMENT_MODES = ("auto", "crop", "resize")

#: ``spatial_alignment.method`` when the prediction, not the GT, was resized
#: (dense depth and points_3d, see :func:`align_to_ground_truth`).
PREDICTION_RESIZE_METHOD = "resize_prediction"

#: ``spatial_alignment.mode`` when a prediction's recorded ``source_grid``
#: decided the alignment (see :class:`SourceGrid`).
SOURCE_GRID_MODE = "source_grid"

#: ``spatial_alignment.method`` for a recorded crop, at any origin.
RECORDED_CROP_METHOD = "crop"

#: ``spatial_alignment.method`` for a recorded relation that is neither a crop
#: nor a whole-image resize (e.g. a resize, then a crop): the prediction is
#: resampled onto the GT grid (``warp_prediction``) or the GT onto the
#: prediction's (``warp``) through the exact pixel mapping.
PREDICTION_WARP_METHOD = "warp_prediction"
WARP_METHOD = "warp"


@dataclass(frozen=True)
class SourceGrid:
    """A prediction's recorded ``source_grid``: how its pixel grid derives from
    the image it was computed from, which is the GT's grid.

    The prediction pixel at ``(x', y')`` sits at source position
    ``((x' - tx) / sx, (y' - ty) / sy)``; pixel centres are at integer
    coordinates (euler-dataset-contract's ``source_grid``, written by
    euler-inference). The camera on the prediction grid is :meth:`matrix`
    ``@ K``.
    """

    source_size: tuple[int, int]
    sx: float
    tx: float
    sy: float
    ty: float

    @classmethod
    def from_value(cls, value: Any) -> "SourceGrid":
        """Parse a ``source_grid`` mapping (or return a :class:`SourceGrid`)."""
        if isinstance(value, SourceGrid):
            return value
        try:
            height, width = (int(v) for v in value["size"])
            (sx, sxy, tx), (syx, sy, ty) = value["affine"]
            numbers = [float(v) for v in (sx, sxy, tx, syx, sy, ty)]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "source_grid must have 'size' [height, width] and 'affine' "
                f"[[sx, 0, tx], [0, sy, ty]], got {value!r}"
            ) from exc
        if (
            height <= 0
            or width <= 0
            or not np.all(np.isfinite(numbers))
            or numbers[1] != 0
            or numbers[3] != 0
            or numbers[0] <= 0
            or numbers[4] <= 0
        ):
            raise ValueError(
                "source_grid must record a crop or resize: a positive size and "
                f"an affine [[sx, 0, tx], [0, sy, ty]] with sx, sy > 0, got {value!r}"
            )
        return cls((height, width), numbers[0], numbers[2], numbers[4], numbers[5])

    def matrix(self) -> np.ndarray:
        """``3x3`` pixel-grid transform from the source to the prediction grid."""
        return np.array(
            [[self.sx, 0.0, self.tx], [0.0, self.sy, self.ty], [0.0, 0.0, 1.0]]
        )

    def to_prediction(self) -> np.ndarray:
        """``2x3`` affine from source to prediction pixel coordinates."""
        return self.matrix()[:2]

    def to_source(self) -> np.ndarray:
        """``2x3`` affine from prediction to source pixel coordinates."""
        return np.array(
            [
                [1.0 / self.sx, 0.0, -self.tx / self.sx],
                [0.0, 1.0 / self.sy, -self.ty / self.sy],
            ]
        )

    def relation(
        self, pred_shape: tuple[int, int]
    ) -> tuple[str, Optional[tuple[int, int]]]:
        """Classify a prediction of *pred_shape* on this grid.

        Returns ``("none" | "crop" | "resize" | "warp", origin)``: ``origin``
        is the ``(top, left)`` source pixel of a crop, else ``None``.
        ``resize`` is a whole-image resize to *pred_shape*.
        """
        height, width = self.source_size
        pred_h, pred_w = int(pred_shape[0]), int(pred_shape[1])
        if (
            self.sx == 1
            and self.sy == 1
            and float(self.tx).is_integer()
            and float(self.ty).is_integer()
        ):
            top, left = int(-self.ty), int(-self.tx)
            if 0 <= top and 0 <= left and top + pred_h <= height and left + pred_w <= width:
                if (top, left, pred_h, pred_w) == (0, 0, height, width):
                    return "none", None
                return "crop", (top, left)
        sx, sy = pred_w / width, pred_h / height
        if np.allclose(
            [self.sx, self.tx, self.sy, self.ty],
            [sx, 0.5 * sx - 0.5, sy, 0.5 * sy - 0.5],
            rtol=1e-9,
            atol=1e-9,
        ):
            return "resize", None
        return "warp", None


#: A run-wide mode (:data:`SPATIAL_ALIGNMENT_MODES`) or one sample's recorded grid.
SpatialAlignment = Union[str, SourceGrid]


def pred_head_source_grid(
    dataset: Any, modality: str = "pred"
) -> Optional[SourceGrid]:
    """Return the ``source_grid`` a prediction dataset's head records, if any."""
    lookup = getattr(dataset, "get_modality_metadata", None)
    if not callable(lookup):
        return None
    try:
        meta = lookup(modality)
    except Exception:
        return None
    value = meta.get("source_grid") if isinstance(meta, Mapping) else None
    return SourceGrid.from_value(value) if value is not None else None


def sample_spatial_alignment(
    sample: Mapping[str, Any],
    mode: SpatialAlignment,
    head_grid: Optional[SourceGrid] = None,
    *,
    modality: str = "pred",
) -> SpatialAlignment:
    """Return how one sample's prediction goes onto the GT grid.

    An explicit ``crop`` or ``resize`` *mode* applies as configured. Under
    ``auto`` the prediction's recorded ``source_grid`` decides: its file's
    attributes, else its dataset head (*head_grid*); ``auto`` guesses from
    the shapes only when neither records one.
    """
    if not isinstance(mode, str) or mode != "auto":
        return mode
    attributes = sample.get("attributes") if isinstance(sample, Mapping) else None
    entry = attributes.get(modality) if isinstance(attributes, Mapping) else None
    value = entry.get("source_grid") if isinstance(entry, Mapping) else None
    if value is not None:
        return SourceGrid.from_value(value)
    return head_grid if head_grid is not None else mode


def spatial_alignment_mode(alignment: SpatialAlignment) -> str:
    """The ``spatial_alignment.mode`` to report for *alignment*."""
    return SOURCE_GRID_MODE if isinstance(alignment, SourceGrid) else alignment


def prediction_intrinsics(
    intrinsics_K: Optional[np.ndarray], alignment: SpatialAlignment
) -> Optional[np.ndarray]:
    """The camera on the prediction grid: ``T @ K`` for a recorded grid.

    Without a recorded grid the intrinsics are returned as they are, which is
    right for an uncropped or top-left-cropped prediction.
    """
    if intrinsics_K is None or not isinstance(alignment, SourceGrid):
        return intrinsics_K
    return alignment.matrix() @ np.asarray(intrinsics_K, dtype=np.float64)


def validate_spatial_alignment(mode: str) -> str:
    """Return *mode* when it is one of :data:`SPATIAL_ALIGNMENT_MODES`."""
    if mode not in SPATIAL_ALIGNMENT_MODES:
        raise ValueError(
            f"spatial_alignment must be one of {list(SPATIAL_ALIGNMENT_MODES)}, "
            f"got {mode!r}"
        )
    return mode


def classify_spatial_alignment(
    gt_h: int,
    gt_w: int,
    pred_h: int,
    pred_w: int,
    *,
    mode: SpatialAlignment = "auto",
    reference: str = "GT",
) -> str:
    """Classify how a GT and a prediction of these shapes go onto one grid.

    Returns ``"none"`` for equal shapes, ``"vae_crop"`` when the prediction
    covers the top-left ``pred_h x pred_w`` pixels of the GT, so the GT is
    cropped, or ``"resize"`` when one of them is resampled onto the other.

    *mode* states what the model did to its input. The shapes alone cannot
    tell a crop from a resize to the same size:

    - ``"auto"`` guesses: a crop when the prediction's sides are multiples of
      8 and at most 7 px shorter than the GT's (what latent models that crop
      produce), else a resize.
    - ``"crop"``: always a top-left crop. A prediction larger than the GT
      raises.
    - ``"resize"``: always a resize.
    - a :class:`SourceGrid` (the prediction's recorded ``source_grid``) knows:
      ``"none"``, ``"crop"`` at any origin, ``"resize"`` of the whole image,
      or ``"warp"`` for anything else. A grid recorded for another source
      size than the GT's cannot place the prediction, so the shapes are
      guessed from as under ``"auto"``, with a warning.

    *reference* names the image in the GT's place in messages, e.g.
    ``"input"`` for the degraded input (:func:`align_with_input`).
    """
    if isinstance(mode, SourceGrid):
        if (gt_h, gt_w) == mode.source_size:
            relation, _ = mode.relation((pred_h, pred_w))
            return {"crop": RECORDED_CROP_METHOD, "warp": WARP_METHOD}.get(
                relation, relation
            )
        import warnings

        warnings.warn(
            f"The prediction's source_grid describes a {mode.source_size[0]}x"
            f"{mode.source_size[1]} source, but the {reference} is {gt_h}x{gt_w}; "
            "guessing the alignment from the shapes instead.",
            stacklevel=2,
        )
        mode = "auto"
    validate_spatial_alignment(mode)
    if gt_h == pred_h and gt_w == pred_w:
        return "none"
    dh = gt_h - pred_h
    dw = gt_w - pred_w
    if mode == "crop":
        if dh < 0 or dw < 0:
            raise ValueError(
                "spatial_alignment 'crop' needs a prediction no larger than "
                f"the {reference}, got prediction {pred_h}x{pred_w} for "
                f"{reference} {gt_h}x{gt_w}"
            )
        return "vae_crop"
    if mode == "resize":
        return "resize"
    if pred_h % 8 == 0 and pred_w % 8 == 0 and 0 <= dh < 8 and 0 <= dw < 8:
        return "vae_crop"
    return "resize"


def align_intrinsics_to_prediction(
    intrinsics_K: np.ndarray,
    gt_shape: tuple[int, int],
    pred_shape: tuple[int, int],
    *,
    mode: SpatialAlignment = "auto",
) -> np.ndarray:
    """Adjust a camera matrix for the alignment :func:`align_to_prediction` did.

    Intrinsics describe the GT image they were captured with, so any resolution
    change applied to the GT map must be applied to them too before they are
    used to unproject the *evaluated* plane.

    - ``none``: returned unchanged.
    - ``vae_crop``: unchanged as well — a top-left crop keeps the pixel origin,
      so the principal point does not move.
    - ``resize``: focal lengths and principal point are scaled with OpenCV's
      continuous-coordinate convention (``x' = (x + 0.5)·s − 0.5``), matching
      the ``cv2.resize`` that :func:`align_to_prediction` performs.

    Dense depth and points_3d keep the GT grid
    (:func:`align_to_ground_truth`), so their intrinsics need no adjustment.

    Args:
        intrinsics_K: ``(3, 3)`` camera matrix for the GT image.
        gt_shape: ``(height, width)`` of the GT map before alignment.
        pred_shape: ``(height, width)`` of the prediction, i.e. the evaluated
            plane.
        mode: The ``spatial_alignment`` mode the GT was aligned with.

    Returns:
        ``(3, 3)`` camera matrix describing the evaluated plane.
    """
    K = np.asarray(intrinsics_K, dtype=np.float64).copy()
    gt_h, gt_w = int(gt_shape[0]), int(gt_shape[1])
    pred_h, pred_w = int(pred_shape[0]), int(pred_shape[1])

    if isinstance(mode, SourceGrid) and (gt_h, gt_w) == mode.source_size:
        # The recorded mapping moves the camera exactly, crops included.
        return mode.matrix() @ K
    if classify_spatial_alignment(gt_h, gt_w, pred_h, pred_w, mode=mode) != "resize":
        return K
    if gt_h <= 0 or gt_w <= 0:
        raise ValueError(f"Invalid GT shape for intrinsics rescaling: {gt_shape}")

    scale_x = pred_w / gt_w
    scale_y = pred_h / gt_h

    K[0, 0] *= scale_x
    K[0, 1] *= scale_x  # skew shares the x axis
    K[1, 1] *= scale_y
    K[0, 2] = (K[0, 2] + 0.5) * scale_x - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * scale_y - 0.5
    return K


def _warn_aspect_ratio(
    gt_shape: tuple[int, ...], pred_shape: tuple[int, ...]
) -> None:
    """Warn when GT and prediction aspect ratios differ by more than 5 %."""
    gt_h, gt_w = gt_shape[:2]
    pred_h, pred_w = pred_shape[:2]
    gt_ar = gt_w / gt_h
    pred_ar = pred_w / pred_h
    if abs(gt_ar - pred_ar) / max(gt_ar, pred_ar) > 0.05:
        import warnings

        warnings.warn(
            f"Significant aspect-ratio mismatch: GT {gt_w}x{gt_h} "
            f"({gt_ar:.3f}) vs pred {pred_w}x{pred_h} ({pred_ar:.3f}). "
            f"Metrics may not be directly comparable to native-resolution "
            f"evaluations.",
            stacklevel=3,
        )


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a boolean mask to ``shape = (height, width)``.

    Each target pixel takes the source pixel whose centre is nearest to its
    own (``cv2.INTER_NEAREST_EXACT``), the pixel-centre convention of the
    image resizes and of :func:`align_intrinsics_to_prediction`. Plain
    ``INTER_NEAREST`` floors ``i / s`` instead and shifts the mask by up to
    half a source pixel against the image.
    """
    import cv2

    height, width = int(shape[0]), int(shape[1])
    mask = np.asarray(mask)
    if mask.shape[:2] == (height, width):
        return mask.astype(bool, copy=False)
    resized = cv2.resize(
        np.ascontiguousarray(mask, dtype=np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST_EXACT,
    )
    return resized.astype(bool)


def _resize_dense(
    values: np.ndarray,
    shape: tuple[int, int],
    *,
    valid: Optional[np.ndarray] = None,
    fill: float = float("nan"),
) -> np.ndarray:
    """Area-average when shrinking, bilinear otherwise, at pixel centres.

    With a ``(H, W)`` *valid* mask, invalid pixels carry no weight: the valid
    values and their support are resized separately and divided, and target
    pixels without support get *fill*.
    """
    import cv2

    height, width = int(shape[0]), int(shape[1])
    src_h, src_w = values.shape[:2]
    interp = (
        cv2.INTER_AREA if height <= src_h and width <= src_w else cv2.INTER_LINEAR
    )
    if valid is None:
        return cv2.resize(
            np.ascontiguousarray(values, dtype=np.float32),
            (width, height),
            interpolation=interp,
        )
    weight = valid.astype(np.float32)
    mask = valid if values.ndim == 2 else valid[..., None]
    clean = np.where(mask, values, 0).astype(np.float32)
    numerator = cv2.resize(clean, (width, height), interpolation=interp)
    support = cv2.resize(weight, (width, height), interpolation=interp)
    if numerator.ndim == 3:
        support = support[..., None]
    out = np.full_like(numerator, fill)
    np.divide(numerator, support, out=out, where=support > 1e-6)
    return out


def _crop_origin(
    method: str,
    mode: SpatialAlignment,
    pred_shape: tuple[int, int],
) -> Optional[tuple[int, int]]:
    """``(top, left)`` of the GT region a cropped prediction covers, or ``None``."""
    if method == "vae_crop":
        return 0, 0
    if method == RECORDED_CROP_METHOD and isinstance(mode, SourceGrid):
        return mode.relation(pred_shape)[1]
    return None


def _warp_dense(
    values: np.ndarray,
    affine: np.ndarray,
    shape: tuple[int, int],
    *,
    valid: Optional[np.ndarray] = None,
    fill: float = float("nan"),
) -> np.ndarray:
    """Resample ``(H, W)`` or ``(H, W, C)`` *values* onto a grid of *shape*.

    *affine* (``2x3``) maps a pixel position of *values* to the target grid,
    pixel centres at integer coordinates. Invalid pixels (non-finite, or
    outside *valid*) carry no weight; a target pixel with none of the values
    behind it gets *fill*. Where the target is coarser, values are first
    area-averaged, as :func:`_resize_dense` does for a resize; the rest is
    bilinear.
    """
    import cv2

    values = np.asarray(values, dtype=np.float32)
    matrix = np.asarray(affine, dtype=np.float64).copy()
    if valid is None:
        valid = (
            np.isfinite(values) if values.ndim == 2 else np.isfinite(values).all(axis=-1)
        )
    src_h, src_w = values.shape[:2]
    fx, fy = min(matrix[0, 0], 1.0), min(matrix[1, 1], 1.0)
    new_w, new_h = max(1, int(round(src_w * fx))), max(1, int(round(src_h * fy)))
    if (new_h, new_w) != (src_h, src_w):
        values = _resize_dense(values, (new_h, new_w), valid=valid, fill=float("nan"))
        valid = (
            np.isfinite(values) if values.ndim == 2 else np.isfinite(values).all(axis=-1)
        )
        # Compose with the inverse of that resize: x = (x_r - 0.5 f + 0.5) / f.
        rx, ry = new_w / src_w, new_h / src_h
        matrix = np.array(
            [
                [matrix[0, 0] / rx, 0.0, matrix[0, 2] + matrix[0, 0] * (0.5 - 0.5 * rx) / rx],
                [0.0, matrix[1, 1] / ry, matrix[1, 2] + matrix[1, 1] * (0.5 - 0.5 * ry) / ry],
            ]
        )
    height, width = int(shape[0]), int(shape[1])
    mask = valid if values.ndim == 2 else valid[..., None]
    clean = np.ascontiguousarray(np.where(mask, values, 0), dtype=np.float32)
    weight = np.ascontiguousarray(valid, dtype=np.float32)
    options = dict(
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    numerator = cv2.warpAffine(clean, matrix, (width, height), **options)
    support = cv2.warpAffine(weight, matrix, (width, height), **options)
    if numerator.ndim == 2 and clean.ndim == 3:  # cv2 drops a single channel
        numerator = numerator[..., None]
    if numerator.ndim == 3:
        support = support[..., None]
    out = np.full_like(numerator, fill)
    np.divide(numerator, support, out=out, where=support > 1e-6)
    return out


def _warp_nearest(
    values: np.ndarray, affine: np.ndarray, shape: tuple[int, int], fill: Any
) -> np.ndarray:
    """Nearest-pixel counterpart of :func:`_warp_dense`, for masks and labels."""
    import cv2

    values = np.asarray(values)
    source = values.astype(np.uint8) if values.dtype == bool else values
    out = cv2.warpAffine(
        np.ascontiguousarray(source),
        np.asarray(affine, dtype=np.float64),
        (int(shape[1]), int(shape[0])),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )
    return out.astype(bool) if values.dtype == bool else out


def _forward_map_sparse(
    depth: np.ndarray, affine: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Move each valid depth sample to its nearest target pixel under *affine*.

    The general-mapping counterpart of euler-loading's
    ``resize_raster(interpolation="sparse")``: the nearest surface wins where
    several land on one pixel, and every output value is a measurement.
    """
    depth = np.asarray(depth)
    height, width = int(shape[0]), int(shape[1])
    valid = np.isfinite(depth) & (depth > 0)
    ys, xs = np.nonzero(valid)
    tx = np.floor(affine[0][0] * xs + affine[0][2] + 0.5).astype(np.int64)
    ty = np.floor(affine[1][1] * ys + affine[1][2] + 0.5).astype(np.int64)
    keep = (tx >= 0) & (tx < width) & (ty >= 0) & (ty < height)
    flat = np.full(height * width, np.inf)
    np.minimum.at(flat, ty[keep] * width + tx[keep], depth[ys[keep], xs[keep]])
    flat[~np.isfinite(flat)] = 0
    dtype = depth.dtype if depth.dtype.kind == "f" else np.float32
    return flat.reshape(height, width).astype(dtype)


def warp_prediction(
    pred: np.ndarray, grid: SourceGrid, shape: tuple[int, int]
) -> np.ndarray:
    """Resample a dense prediction onto its source grid of *shape*.

    For depth ``(H, W)`` and point maps ``(H, W, C)`` with a recorded
    :class:`SourceGrid`. GT pixels the prediction does not cover become NaN,
    so they drop out of the metrics.
    """
    pred = np.asarray(pred)
    resized = _warp_dense(pred, grid.to_source(), shape)
    dtype = pred.dtype if pred.dtype.kind == "f" else np.float32
    return resized.astype(dtype, copy=False)


def _warp_gt_onto_prediction(
    gt: np.ndarray, grid: SourceGrid, shape: tuple[int, int], *, sparse: bool
) -> np.ndarray:
    """Bring a GT array onto a prediction grid its recorded mapping describes."""
    affine = grid.to_prediction()
    if gt.dtype == bool:
        return _warp_nearest(gt, affine, shape, fill=0)
    if gt.ndim == 2:
        if sparse:
            return _forward_map_sparse(gt, affine, shape)
        valid = np.isfinite(gt) & (gt > 0)
        clean = np.where(valid, gt, 0).astype(np.float32)
        dtype = gt.dtype if gt.dtype.kind == "f" else np.float32
        if grid.sx >= 1 and grid.sy >= 1:
            # Enlarging takes the nearest GT pixel, as resize_depth_gt does.
            return _warp_nearest(clean, affine, shape, fill=0.0).astype(dtype)
        return _warp_dense(clean, affine, shape, valid=valid, fill=0.0).astype(dtype)
    warped = _warp_dense(gt, affine, shape)
    return warped.astype(gt.dtype, copy=False) if gt.dtype.kind == "f" else warped


def resize_prediction(pred: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a dense prediction onto the GT grid ``shape = (height, width)``.

    For depth ``(H, W)`` and point maps ``(H, W, C)``. Shrinking averages
    areas and enlarging is bilinear, both at pixel centres
    (``align_corners=False``). Non-finite pixels carry no weight; target
    pixels with no finite source become NaN.
    """
    pred = np.asarray(pred)
    if pred.shape[:2] == (int(shape[0]), int(shape[1])):
        return pred
    finite = np.isfinite(pred) if pred.ndim == 2 else np.isfinite(pred).all(axis=-1)
    resized = _resize_dense(pred, shape, valid=None if finite.all() else finite)
    dtype = pred.dtype if pred.dtype.kind == "f" else np.float32
    return resized.astype(dtype, copy=False)


def _sparse_resize_raster():
    """Return euler-loading's ``resize_raster`` if it can resize sparse depth."""
    import inspect

    try:
        from euler_loading import resize_raster
    except ImportError:
        resize_raster = None
    if (
        resize_raster is None
        or "invalid_value" not in inspect.signature(resize_raster).parameters
    ):
        raise RuntimeError(
            "Resizing sparse GT depth needs euler-loading's "
            "resize_raster(interpolation='sparse') from euler-loading 2.36.0, "
            "which euler-eval requires. Upgrade euler-loading."
        )
    return resize_raster


def resize_depth_gt(
    depth: np.ndarray, shape: tuple[int, int], *, sparse: bool = False
) -> np.ndarray:
    """Resize GT depth to ``shape = (height, width)`` without mixing in invalid pixels.

    Valid GT is finite and ``> 0``, as everywhere in euler-eval. Invalid
    pixels never contribute to a valid one, and come out as ``0``.

    - *sparse* (a GT head with ``"sparse": true``, such as lidar-derived
      depth): every valid pixel moves to the nearest target pixel, and where
      several land on one the nearest surface wins (euler-loading's
      ``resize_raster(..., interpolation="sparse")``). Every output value is
      a measurement; none is spread or duplicated.
    - dense: shrinking area-averages the valid pixels only; enlarging takes
      the source pixel whose centre is nearest.
    """
    import cv2

    depth = np.asarray(depth)
    height, width = int(shape[0]), int(shape[1])
    if depth.shape[:2] == (height, width):
        return depth
    valid = np.isfinite(depth) & (depth > 0)
    clean = np.where(valid, depth, 0).astype(np.float32)
    if sparse:
        resized = _sparse_resize_raster()(
            clean,
            layout="HW",
            size=(height, width),
            interpolation="sparse",
            invalid_value=0.0,
        )
    elif height >= depth.shape[0] and width >= depth.shape[1]:
        resized = cv2.resize(
            clean, (width, height), interpolation=cv2.INTER_NEAREST_EXACT
        )
    else:
        resized = _resize_dense(clean, (height, width), valid=valid, fill=0.0)
    dtype = depth.dtype if depth.dtype.kind == "f" else np.float32
    return np.asarray(resized).astype(dtype, copy=False)


def align_to_prediction(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    spatial_alignment: SpatialAlignment = "auto",
    sparse: bool = False,
) -> np.ndarray:
    """Bring a GT array onto the prediction's grid.

    :func:`classify_spatial_alignment` with *spatial_alignment* decides
    between a top-left crop and a resize. A resize depends on the array:

    - boolean masks: :func:`resize_mask`, nearest pixel centre;
    - ``(H, W)`` depth: :func:`resize_depth_gt`, which keeps invalid pixels
      out of valid ones; *sparse* selects its sparse mode;
    - ``(H, W, C)`` images and direction maps: area average when shrinking,
      bilinear when enlarging.

    A recorded :class:`SourceGrid` crops at its origin, resizes as above for a
    whole-image resize, and otherwise resamples through its exact mapping
    (sparse depth forward-mapped, nearest surface first).

    Dense depth and points_3d are evaluated on the GT grid instead
    (:func:`align_to_ground_truth`). This direction serves RGB, rays, GT
    masks and the projection plane of sparse GT.

    Args:
        gt: Ground-truth array.
        pred: Prediction array whose spatial dimensions are the target.
        spatial_alignment: ``auto``, ``crop``, ``resize`` or the sample's
            recorded :class:`SourceGrid`.
        sparse: The GT is sparse depth.

    Returns:
        GT array with spatial dimensions matching *pred*.
    """
    gt_h, gt_w = gt.shape[:2]
    pred_h, pred_w = pred.shape[:2]
    method = classify_spatial_alignment(
        gt_h, gt_w, pred_h, pred_w, mode=spatial_alignment
    )
    if method == "none":
        return gt
    origin = _crop_origin(method, spatial_alignment, (pred_h, pred_w))
    if origin is not None:
        top, left = origin
        return gt[top : top + pred_h, left : left + pred_w]
    if method == WARP_METHOD:
        return _warp_gt_onto_prediction(
            gt, spatial_alignment, (pred_h, pred_w), sparse=sparse
        )

    _warn_aspect_ratio(gt.shape, pred.shape)
    if gt.dtype == bool:
        return resize_mask(gt, (pred_h, pred_w))
    if gt.ndim == 2:
        return resize_depth_gt(gt, (pred_h, pred_w), sparse=sparse)

    import cv2

    # INTER_AREA averages source pixels when shrinking and avoids the
    # aliasing INTER_LINEAR produces at large scale ratios.
    is_downscale = (pred_h * pred_w) < (gt_h * gt_w)
    interp = cv2.INTER_AREA if is_downscale else cv2.INTER_LINEAR
    return cv2.resize(gt, (pred_w, pred_h), interpolation=interp)


def align_to_ground_truth(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    spatial_alignment: SpatialAlignment = "auto",
) -> tuple[np.ndarray, np.ndarray, str]:
    """Put a dense prediction and its GT on the GT's pixel grid.

    For a crop (:func:`classify_spatial_alignment`), the GT is cropped to the
    top-left region the prediction covers. Otherwise the prediction is
    resized onto the GT grid (:func:`resize_prediction`) and the GT stays as
    measured: no GT value is interpolated, invalid pixels stay invalid, and
    the GT's intrinsics describe the evaluated grid unchanged.

    A recorded :class:`SourceGrid` (*spatial_alignment*) crops the GT at its
    origin, or resamples the prediction through its exact mapping
    (:func:`warp_prediction`) when it is neither a crop nor a whole-image
    resize.

    Returns:
        ``(pred, gt, method)``, with *method* ``"none"``, ``"vae_crop"``,
        :data:`RECORDED_CROP_METHOD`, :data:`PREDICTION_RESIZE_METHOD` or
        :data:`PREDICTION_WARP_METHOD`.
    """
    gt_h, gt_w = gt.shape[:2]
    pred_h, pred_w = pred.shape[:2]
    method = classify_spatial_alignment(
        gt_h, gt_w, pred_h, pred_w, mode=spatial_alignment
    )
    origin = _crop_origin(method, spatial_alignment, (pred_h, pred_w))
    if origin is not None:
        top, left = origin
        return pred, gt[top : top + pred_h, left : left + pred_w], method
    if method == "resize":
        _warn_aspect_ratio(gt.shape, pred.shape)
        return resize_prediction(pred, (gt_h, gt_w)), gt, PREDICTION_RESIZE_METHOD
    if method == WARP_METHOD:
        return (
            warp_prediction(pred, spatial_alignment, (gt_h, gt_w)),
            gt,
            PREDICTION_WARP_METHOD,
        )
    return pred, gt, method


def classify_upscaling_alignment(
    pred_shape: tuple[int, int], input_shape: tuple[int, int]
) -> str:
    """Name what :func:`align_by_upscaling` does to a prediction and its input.

    Returns ``"none"``, ``"upscale_input"`` or ``"upscale_prediction"``.
    """
    pred_shape = tuple(pred_shape[:2])
    input_shape = tuple(input_shape[:2])
    if pred_shape == input_shape:
        return "none"
    if input_shape[0] * input_shape[1] <= pred_shape[0] * pred_shape[1]:
        return "upscale_input"
    return "upscale_prediction"


def align_by_upscaling(
    pred: np.ndarray, input_image: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bring a prediction and its degraded input onto one pixel grid.

    The image with fewer pixels is resized to the other's height and width
    with bilinear interpolation (the prediction's grid wins a tie); neither
    is cropped. Equal shapes are returned unchanged. Warns when the aspect
    ratios differ by more than 5 %, since the resize then also stretches.

    Returns:
        ``(pred, input_image)`` with equal spatial dimensions.
    """
    method = classify_upscaling_alignment(pred.shape, input_image.shape)
    if method == "none":
        return pred, input_image

    import warnings

    import cv2

    source, target = (
        (input_image, pred) if method == "upscale_input" else (pred, input_image)
    )
    source_h, source_w = source.shape[:2]
    target_h, target_w = target.shape[:2]
    source_ar = source_w / source_h
    target_ar = target_w / target_h
    if abs(source_ar - target_ar) / max(source_ar, target_ar) > 0.05:
        warnings.warn(
            f"Significant aspect-ratio mismatch between prediction "
            f"{pred.shape[1]}x{pred.shape[0]} and input "
            f"{input_image.shape[1]}x{input_image.shape[0]}; the smaller one is "
            "stretched onto the larger one's grid.",
            stacklevel=2,
        )
    resized = cv2.resize(
        np.ascontiguousarray(source, dtype=np.float32),
        (target_w, target_h),
        interpolation=cv2.INTER_LINEAR,
    )
    if method == "upscale_input":
        return pred, resized
    return resized, input_image


#: ``method.input_alignment`` when the degraded input was cropped to the
#: pixels a cropped prediction covers (:func:`align_with_input`).
INPUT_CROP_METHOD = "crop_input"

#: ``method.input_alignment`` when the input was resampled onto the prediction
#: grid through a recorded mapping that is neither a crop nor a whole-image
#: resize (:func:`align_with_input`).
INPUT_WARP_METHOD = "warp_input"


def _input_method(
    spatial_method: str, pred_shape: tuple[int, ...], input_shape: tuple[int, ...]
) -> str:
    """The ``input_alignment`` for a :func:`classify_spatial_alignment` result."""
    if spatial_method == "none":
        return "none"
    if spatial_method in ("vae_crop", RECORDED_CROP_METHOD):
        return INPUT_CROP_METHOD
    if spatial_method == WARP_METHOD:
        return INPUT_WARP_METHOD
    return classify_upscaling_alignment(pred_shape, input_shape)


def classify_input_alignment(
    pred_shape: tuple[int, ...],
    input_shape: tuple[int, ...],
    *,
    spatial_alignment: SpatialAlignment = "auto",
) -> str:
    """Name what :func:`align_with_input` does to a prediction and its input.

    Returns ``"none"``, :data:`INPUT_CROP_METHOD`, ``"upscale_input"``,
    ``"upscale_prediction"`` or :data:`INPUT_WARP_METHOD`.
    """
    in_h, in_w = (int(n) for n in input_shape[:2])
    pred_h, pred_w = (int(n) for n in pred_shape[:2])
    spatial_method = classify_spatial_alignment(
        in_h, in_w, pred_h, pred_w, mode=spatial_alignment, reference="input"
    )
    return _input_method(spatial_method, pred_shape, input_shape)


def align_with_input(
    pred: np.ndarray,
    input_image: np.ndarray,
    *,
    spatial_alignment: SpatialAlignment = "auto",
) -> tuple[np.ndarray, np.ndarray, str]:
    """Bring a prediction and the degraded input it restores onto one grid.

    The input is the image the prediction was computed from, so it goes
    where the GT goes in :func:`classify_spatial_alignment`: the prediction's
    recorded ``source_grid`` (*spatial_alignment* a :class:`SourceGrid`), an
    explicit ``crop`` or ``resize``, or ``auto``'s guess from the shapes
    decides, the same for both references.

    - A crop (top-left for ``vae_crop``, at its recorded origin otherwise) is
      compared with exactly the input pixels it covers: the input is cropped
      (:data:`INPUT_CROP_METHOD`) and nothing is resampled. Upscaling the
      prediction instead would stretch it over pixels it never saw, shifting
      its content by up to the size difference.
    - A resize upscales the smaller image onto the other's grid
      (:func:`align_by_upscaling`).
    - A recorded mapping that is neither, such as a resize followed by a
      crop, resamples the input onto the prediction grid through it
      (:data:`INPUT_WARP_METHOD`).

    Returns:
        ``(pred, input_image, method)``: both images with equal spatial
        dimensions, and *method* as :func:`classify_input_alignment` names it.
    """
    in_h, in_w = input_image.shape[:2]
    pred_h, pred_w = pred.shape[:2]
    spatial_method = classify_spatial_alignment(
        in_h, in_w, pred_h, pred_w, mode=spatial_alignment, reference="input"
    )
    method = _input_method(spatial_method, pred.shape, input_image.shape)
    if method == "none":
        return pred, input_image, method
    if method == INPUT_CROP_METHOD:
        top, left = _crop_origin(spatial_method, spatial_alignment, (pred_h, pred_w))
        return pred, input_image[top : top + pred_h, left : left + pred_w], method
    if method == INPUT_WARP_METHOD:
        warped = _warp_dense(
            input_image, spatial_alignment.to_prediction(), (pred_h, pred_w)
        )
        return pred, warped, method
    aligned_pred, aligned_input = align_by_upscaling(pred, input_image)
    return aligned_pred, aligned_input, method


def validate_sky_depth(sky_depth: Optional[float]) -> None:
    """Require a finite positive depth cap when sky-depth processing is enabled."""
    if sky_depth is not None and (not np.isfinite(sky_depth) or sky_depth <= 0):
        raise ValueError("sky_depth must be a finite positive value")


def compute_scale_and_shift(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    max_gt_percentile: Optional[float] = None,
) -> tuple[np.ndarray, float, float]:
    """Align predicted depth to GT via least-squares scale and shift.

    Solves ``min_{s,t}  Σ (s·pred + t − gt)²`` over valid pixels and
    returns the aligned prediction together with the fitted parameters.

    Args:
        pred: Predicted depth ``(H, W)``.
        gt: Ground-truth depth in metres ``(H, W)``.
        valid_mask: Optional ``(H, W)`` bool mask.  When *None*, all
                    finite positive pixels in both arrays are used.
        max_gt_percentile: Optional upper GT-depth percentile for the fit.
            Pixels above this percentile are ignored when enough samples
            remain, which is useful for suppressing residual sky-depth
            outliers during masked evaluation.

    Returns:
        ``(aligned, scale, shift)`` where *aligned* is
        ``(s·pred + t).astype(float32)`` and *scale*/*shift* are the
        fitted affine parameters.
    """
    if valid_mask is None:
        valid_mask = (gt > 0) & np.isfinite(gt) & np.isfinite(pred)

    n_valid = int(valid_mask.sum())
    if n_valid < 2:
        return pred.copy(), 1.0, 0.0

    if max_gt_percentile is not None:
        if not 0.0 < max_gt_percentile <= 100.0:
            raise ValueError(
                f"max_gt_percentile must be in (0, 100], got {max_gt_percentile}."
            )

        gt_valid = gt[valid_mask].astype(np.float64)
        gt_cap = float(np.percentile(gt_valid, max_gt_percentile))
        trimmed_mask = valid_mask & (gt <= gt_cap)
        trimmed_count = int(trimmed_mask.sum())
        if trimmed_count >= 2:
            valid_mask = trimmed_mask
            n_valid = trimmed_count

    pred_valid = pred[valid_mask].astype(np.float64)
    gt_valid = gt[valid_mask].astype(np.float64)

    # Solve  [pred, 1] @ [s, t]^T = gt
    A = np.stack([pred_valid, np.ones(n_valid, dtype=np.float64)], axis=-1)
    params, _, _, _ = np.linalg.lstsq(A, gt_valid, rcond=None)
    s, t = float(params[0]), float(params[1])

    aligned = (s * pred.astype(np.float64) + t).astype(np.float32)
    return aligned, s, t


# ---------------------------------------------------------------------------
# Loader resolution helpers
# ---------------------------------------------------------------------------


def _resolve_sky_mask_loader(
    segmentation_path: str,
    *,
    modality_key: str = "segmentation",
) -> Callable[..., Any]:
    """Resolve the ``sky_mask`` loader for a segmentation dataset.

    The segmentation dataset's ds-crawler index declares a loader module
    (e.g. ``"vkitti2"``).  This function imports that module and returns
    its ``sky_mask`` function so that raw class-segmentation data is
    converted to a boolean sky mask at load time.
    """
    modality = _modality(path=segmentation_path, modality_key=modality_key)
    try:
        index = index_dataset_from_path(
            modality.path,
            metadata_scope=modality.metadata_scope,
        )
    except FileNotFoundError:
        index = index_dataset_from_path(modality.path)
    # Resolve euler_loading metadata the same way euler-loading does: via the
    # contract addon API (addons.euler_loading in the ds-crawler index).
    euler_meta = None
    try:
        contract = get_dataset_contract(dict(index))
        euler_meta = contract.get_addon("euler_loading")
    except Exception:
        pass
    if not isinstance(euler_meta, Mapping):
        euler_meta = {}
    loader_name = euler_meta.get("loader")
    if loader_name is None:
        raise ValueError(
            f"Segmentation dataset at {modality.path!r} does not declare "
            f"an 'euler_loading.loader' in its ds-crawler index."
        )
    module = resolve_loader_module(loader_name)
    sky_fn = getattr(module, "sky_mask", None)
    if sky_fn is None or not callable(sky_fn):
        raise ValueError(
            f"Loader module {loader_name!r} (resolved from {modality.path!r}) "
            f"does not expose a 'sky_mask' function."
        )
    return sky_fn


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def _modality(
    *,
    path: str,
    modality_key: str,
    split: Optional[str] = None,
    metadata_scope: Optional[str] = None,
    loader: Optional[Callable[..., Any]] = None,
    used_as: Optional[str] = None,
) -> Modality:
    """Create an euler-loading modality with an explicit metadata selector."""
    return build_modality(
        path=path,
        loader=loader,
        used_as=used_as,
        modality_key=modality_key,
        metadata_scope=metadata_scope,
        split=split,
    )


def _gt_segmentation_modality(
    path: str,
    *,
    modality_key: str = "segmentation",
    split: Optional[str] = None,
) -> Modality:
    """Return GT segmentation as an ID-paired modality whose loader yields sky.

    Segmentation is per frame, so it is a regular modality, paired with the
    evaluated frames by hierarchy path and file ID like depth. As a
    hierarchical modality each sample would receive every mask of its folder,
    and a frame could be masked with another frame's sky. Being ID-paired, a
    frame without its own segmentation is not evaluated.
    """
    return _modality(
        path=path,
        modality_key=modality_key,
        loader=_resolve_sky_mask_loader(path, modality_key=modality_key),
        split=split,
    )


def build_depth_eval_dataset(
    gt_depth_path: str,
    pred_depth_path: str,
    calibration_path: Optional[str] = None,
    segmentation_path: Optional[str] = None,
    gt_depth_split: Optional[str] = None,
    pred_depth_split: Optional[str] = None,
    calibration_split: Optional[str] = None,
    segmentation_split: Optional[str] = None,
    pred_depth_metadata_scope: Optional[str] = None,
    segmentation_modality_key: str = "segmentation",
    pred_sky_mask_path: Optional[str] = None,
    pred_sky_mask_split: Optional[str] = None,
) -> MultiModalDataset:
    """Build a MultiModalDataset for depth evaluation.

    The returned dataset yields samples with keys ``"gt"``, ``"pred"``,
    and optionally ``"calibration"``, ``"segmentation"`` (GT sky, True = sky,
    ID-paired like ``"pred_sky_mask"``), and ``"pred_sky_mask"`` (prediction,
    True = sky).

    Loaders are resolved automatically from each dataset directory's
    ds-crawler index metadata.

    Args:
        gt_depth_path: Path to GT depth dataset root.
        pred_depth_path: Path to prediction depth dataset root.
        calibration_path: Optional path to calibration dataset.
        segmentation_path: Optional path to GT segmentation dataset.
        gt_depth_split: Optional split name for the GT depth modality.
        pred_depth_split: Optional split name for the prediction depth modality.
        calibration_split: Optional split name for the calibration modality.
        segmentation_split: Optional split name for the segmentation modality.
        pred_depth_metadata_scope: Optional metadata scope for the prediction
            depth modality, useful when loading depth-like outputs such as
            ``relative_depth`` or ``affine_depth`` through the depth loader.
        pred_sky_mask_path: Optional predicted ``sky_mask`` dataset, paired
            with depth by sample ID using its own loader metadata.
        pred_sky_mask_split: Optional split for the predicted sky mask.

    Returns:
        A MultiModalDataset instance.
    """
    modalities = {
        "gt": _modality(
            path=gt_depth_path,
            modality_key="depth",
            split=gt_depth_split,
        ),
        "pred": _modality(
            path=pred_depth_path,
            modality_key="depth",
            metadata_scope=pred_depth_metadata_scope,
            used_as="output",
            split=pred_depth_split,
        ),
    }
    if pred_sky_mask_path is not None:
        modalities["pred_sky_mask"] = _modality(
            path=pred_sky_mask_path,
            modality_key="sky_mask",
            used_as="output",
            split=pred_sky_mask_split,
        )

    hierarchical = {}
    if calibration_path is not None:
        hierarchical["calibration"] = _modality(
            path=calibration_path,
            modality_key="calibration",
            split=calibration_split,
        )
    if segmentation_path is not None:
        modalities["segmentation"] = _gt_segmentation_modality(
            segmentation_path,
            modality_key=segmentation_modality_key,
            split=segmentation_split,
        )

    return MultiModalDataset(
        modalities=modalities,
        hierarchical_modalities=hierarchical if hierarchical else None,
    )


def build_sparse_depth_eval_dataset(
    gt_sparse_depth_path: str,
    pred_depth_path: str,
    intrinsics_path: str,
    camera_extrinsics_path: str,
    lidar_extrinsics_path: Optional[str] = None,
    segmentation_path: Optional[str] = None,
    gt_sparse_depth_split: Optional[str] = None,
    pred_depth_split: Optional[str] = None,
    intrinsics_split: Optional[str] = None,
    camera_extrinsics_split: Optional[str] = None,
    lidar_extrinsics_split: Optional[str] = None,
    segmentation_split: Optional[str] = None,
    pred_depth_metadata_scope: Optional[str] = None,
    segmentation_modality_key: str = "segmentation",
    pred_sky_mask_path: Optional[str] = None,
    pred_sky_mask_split: Optional[str] = None,
) -> MultiModalDataset:
    """Build a MultiModalDataset for sparse pointcloud depth evaluation.

    The returned dataset yields dense predicted depth under ``"pred"``,
    sparse GT point clouds under ``"gt"``, and hierarchical camera
    calibration under ``"intrinsics"`` and ``"camera_extrinsics"``.
    When ``lidar_extrinsics_path`` is supplied, ``camera_extrinsics`` and
    ``lidar_extrinsics`` are treated as per-sensor poses in a shared frame and
    composed into the source-to-camera transform used for projection.  Without
    ``lidar_extrinsics_path``, ``camera_extrinsics`` remains the direct
    source-to-camera transform, such as MUSES ``lidar2rgb``.
    ``pred_depth_metadata_scope`` can select a depth-like prediction scope
    such as ``relative_depth`` or ``affine_depth`` while still loading it
    as a dense depth map. Segmentation remains optional for sparse depth; when
    given it is ID-paired with the frames (True = sky).
    ``pred_sky_mask_path`` optionally adds an ID-paired ``"pred_sky_mask"``
    prediction (True = sky), with its own ``pred_sky_mask_split`` selector.
    """
    modalities = {
        "gt": _modality(
            path=gt_sparse_depth_path,
            modality_key="sparse_depth",
            split=gt_sparse_depth_split,
        ),
        "pred": _modality(
            path=pred_depth_path,
            modality_key="depth",
            metadata_scope=pred_depth_metadata_scope,
            used_as="output",
            split=pred_depth_split,
        ),
    }
    if pred_sky_mask_path is not None:
        modalities["pred_sky_mask"] = _modality(
            path=pred_sky_mask_path,
            modality_key="sky_mask",
            used_as="output",
            split=pred_sky_mask_split,
        )

    if segmentation_path is not None:
        modalities["segmentation"] = _gt_segmentation_modality(
            segmentation_path,
            modality_key=segmentation_modality_key,
            split=segmentation_split,
        )
    hierarchical = _sparse_projection_hierarchical(
        intrinsics_path=intrinsics_path,
        camera_extrinsics_path=camera_extrinsics_path,
        lidar_extrinsics_path=lidar_extrinsics_path,
        intrinsics_split=intrinsics_split,
        camera_extrinsics_split=camera_extrinsics_split,
        lidar_extrinsics_split=lidar_extrinsics_split,
    )

    return MultiModalDataset(
        modalities=modalities,
        hierarchical_modalities=hierarchical,
    )


def _sparse_projection_hierarchical(
    intrinsics_path: str,
    camera_extrinsics_path: str,
    lidar_extrinsics_path: Optional[str] = None,
    intrinsics_split: Optional[str] = None,
    camera_extrinsics_split: Optional[str] = None,
    lidar_extrinsics_split: Optional[str] = None,
) -> dict:
    """Build the hierarchical calibration modalities shared by sparse-GT evals.

    Both the sparse-depth and sparse ``points_3d`` datasets project a LiDAR
    cloud into the prediction plane, so they need identical ``intrinsics`` /
    ``camera_extrinsics`` (optionally ``lidar_extrinsics``) hierarchical
    modalities. GT segmentation is ID-paired and added by the builders.
    """
    hierarchical = {
        "intrinsics": _modality(
            path=intrinsics_path,
            modality_key="intrinsics",
            split=intrinsics_split,
        ),
        "camera_extrinsics": _modality(
            path=camera_extrinsics_path,
            modality_key="camera_extrinsics",
            split=camera_extrinsics_split,
        ),
    }
    if lidar_extrinsics_path is not None:
        lidar_scope = (
            parse_modality_path(
                lidar_extrinsics_path,
                split=lidar_extrinsics_split,
            ).metadata_scope
            or "lidar_extrinsics"
        )
        hierarchical["lidar_extrinsics"] = _modality(
            path=lidar_extrinsics_path,
            modality_key="camera_extrinsics",
            metadata_scope=lidar_scope,
            split=lidar_extrinsics_split,
        )
    return hierarchical


def build_points_3d_sparse_eval_dataset(
    gt_sparse_depth_path: str,
    pred_points_3d_path: str,
    intrinsics_path: str,
    camera_extrinsics_path: str,
    lidar_extrinsics_path: Optional[str] = None,
    segmentation_path: Optional[str] = None,
    gt_sparse_depth_split: Optional[str] = None,
    pred_points_3d_split: Optional[str] = None,
    intrinsics_split: Optional[str] = None,
    camera_extrinsics_split: Optional[str] = None,
    lidar_extrinsics_split: Optional[str] = None,
    segmentation_split: Optional[str] = None,
    segmentation_modality_key: str = "segmentation",
) -> MultiModalDataset:
    """Build a MultiModalDataset for points_3d-vs-sparse-pointcloud evaluation.

    Like :func:`build_sparse_depth_eval_dataset`, but the prediction under
    ``"pred"`` is a per-pixel ``(H, W, 3)`` point map (``points_3d``) that is
    scored *directly* against the sparse GT cloud (no depth unprojection). The
    sparse GT and hierarchical ``intrinsics`` / ``camera_extrinsics`` (plus
    optional ``lidar_extrinsics``, and the ID-paired ``segmentation``) are
    configured exactly as for the sparse-depth path.
    """
    modalities = {
        "gt": _modality(
            path=gt_sparse_depth_path,
            modality_key="sparse_depth",
            split=gt_sparse_depth_split,
        ),
        "pred": _modality(
            path=pred_points_3d_path,
            modality_key="points_3d",
            used_as="output",
            split=pred_points_3d_split,
        ),
    }
    if segmentation_path is not None:
        modalities["segmentation"] = _gt_segmentation_modality(
            segmentation_path,
            modality_key=segmentation_modality_key,
            split=segmentation_split,
        )
    hierarchical = _sparse_projection_hierarchical(
        intrinsics_path=intrinsics_path,
        camera_extrinsics_path=camera_extrinsics_path,
        lidar_extrinsics_path=lidar_extrinsics_path,
        intrinsics_split=intrinsics_split,
        camera_extrinsics_split=camera_extrinsics_split,
        lidar_extrinsics_split=lidar_extrinsics_split,
    )
    return MultiModalDataset(
        modalities=modalities,
        hierarchical_modalities=hierarchical,
    )


def build_rgb_eval_dataset(
    gt_rgb_path: Optional[str],
    pred_rgb_path: str,
    gt_depth_path: Optional[str] = None,
    calibration_path: Optional[str] = None,
    segmentation_path: Optional[str] = None,
    gt_rgb_split: Optional[str] = None,
    pred_rgb_split: Optional[str] = None,
    gt_depth_split: Optional[str] = None,
    calibration_split: Optional[str] = None,
    segmentation_split: Optional[str] = None,
    segmentation_modality_key: str = "segmentation",
    input_rgb_path: Optional[str] = None,
    input_rgb_split: Optional[str] = None,
) -> MultiModalDataset:
    """Build a MultiModalDataset for RGB evaluation.

    The returned dataset yields samples with key ``"pred"``, ``"gt"`` (the
    clear reference) and/or ``"input"`` (the degraded input the prediction
    restores), and optionally ``"gt_depth"``, ``"calibration"``,
    ``"segmentation"``. Every modality is paired with the prediction by ID.

    Loaders are resolved automatically from each dataset directory's
    ds-crawler index metadata.

    Args:
        gt_rgb_path: Path to GT RGB dataset root, or ``None`` when only the
            input is a reference.
        pred_rgb_path: Path to prediction RGB dataset root.
        gt_depth_path: Optional GT depth path for depth-binned metrics.
        calibration_path: Optional calibration dataset path.
        segmentation_path: Optional GT segmentation dataset path.
        gt_rgb_split: Optional split name for the GT RGB modality.
        pred_rgb_split: Optional split name for the prediction RGB modality.
        gt_depth_split: Optional split name for the GT depth modality.
        calibration_split: Optional split name for the calibration modality.
        segmentation_split: Optional split name for the segmentation modality.
        segmentation_modality_key: Config key of the segmentation input.
        input_rgb_path: Optional path to the degraded input RGB dataset root.
        input_rgb_split: Optional split name for the input RGB modality.

    Returns:
        A MultiModalDataset instance.

    Raises:
        ValueError: If neither ``gt_rgb_path`` nor ``input_rgb_path`` is set.
    """
    if gt_rgb_path is None and input_rgb_path is None:
        raise ValueError("RGB evaluation needs gt.rgb or gt.input_rgb")
    modalities: dict[str, Modality] = {}
    if gt_rgb_path is not None:
        modalities["gt"] = _modality(
            path=gt_rgb_path,
            modality_key="rgb",
            split=gt_rgb_split,
        )
    if input_rgb_path is not None:
        modalities["input"] = _modality(
            path=input_rgb_path,
            modality_key="rgb",
            split=input_rgb_split,
        )
    modalities["pred"] = _modality(
        path=pred_rgb_path,
        modality_key="rgb",
        used_as="output",
        split=pred_rgb_split,
    )
    if gt_depth_path is not None:
        modalities["gt_depth"] = _modality(
            path=gt_depth_path,
            modality_key="depth",
            split=gt_depth_split,
        )

    hierarchical = {}
    if calibration_path is not None:
        hierarchical["calibration"] = _modality(
            path=calibration_path,
            modality_key="calibration",
            split=calibration_split,
        )
    if segmentation_path is not None:
        modalities["segmentation"] = _gt_segmentation_modality(
            segmentation_path,
            modality_key=segmentation_modality_key,
            split=segmentation_split,
        )

    return MultiModalDataset(
        modalities=modalities,
        hierarchical_modalities=hierarchical if hierarchical else None,
    )


class PartialInputDataset:
    """Every frame of *complete*, with an optional input where a frame has it.

    euler-loading intersects every modality by ID, so an input such as GT
    segmentation that covers only some frames removes the others from the
    pairing. This dataset keeps them: *complete* is built without the input
    and *with_input* with it, from the same other inputs, so its frames are a
    subset of *complete*'s. A frame of *with_input* is read from it, any
    other frame from *complete*, whose sample then has no entry for the
    input. Frames keep *complete*'s order and identities.

    Public attributes such as ``get_modality_metadata`` and ``modality_paths``
    come from *with_input*, which describes every input.
    """

    def __init__(self, complete: Any, with_input: Any) -> None:
        self._complete = complete
        self._with_input = with_input
        index = {
            with_input.sample_full_id(j): j for j in range(len(with_input))
        }
        self._source = [
            index.pop(complete.sample_full_id(i), None) for i in range(len(complete))
        ]
        if index:
            raise ValueError(
                f"{len(index)} frames with the optional input are not among the "
                "frames without it; both datasets must share their other inputs"
            )

    @property
    def frames_without_input(self) -> int:
        return sum(source is None for source in self._source)

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source = self._source[index]
        if source is None:
            return self._complete[index]
        return self._with_input[source]

    def sample_full_id(self, index: int, modality_name: Optional[str] = None) -> str:
        return self._complete.sample_full_id(index, modality_name)

    def __getattr__(self, name: str) -> Any:
        # Private and dunder lookups (pickling, ``_lookups``) are not forwarded:
        # they would describe *with_input*'s frames, not this dataset's.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._with_input, name)


def count_ids_dropped_by(
    dataset: Any, modality: str, *, optional: tuple[str, ...] = ()
) -> Optional[int]:
    """Count the sample IDs that *modality* removes from a dataset's pairing.

    Regular modalities are intersected by ID, so an incomplete optional
    modality (such as ``segmentation`` or ``pred_sky_mask``) shrinks the
    evaluated set, which euler-loading only logs. Returns how many IDs the
    required modalities share but *modality* lacks. The other *optional*
    modalities are left out of that intersection, so two inputs missing the
    same frame both count it. ``None`` when the dataset does not expose its
    per-modality lookups or has no such modality.
    """
    lookups = getattr(dataset, "_lookups", None)
    if not isinstance(lookups, Mapping) or modality not in lookups:
        return None
    required = [
        set(ids)
        for name, ids in lookups.items()
        if name != modality and name not in optional
    ]
    if not required:
        return 0
    return len(set.intersection(*required) - set(lookups[modality]))


def get_depth_metadata(
    dataset: MultiModalDataset, modality: str = "gt"
) -> dict[str, Any]:
    """Extract depth modality metadata from a dataset.

    Args:
        dataset: The dataset holding the depth modality.
        modality: Its key in the dataset (``"gt_depth"`` beside RGB GT).

    Returns:
        Dict with ``radial_depth`` (bool) and ``sparse`` (bool, the
        ``"sparse"`` flag for lidar-like depth; selects the sparse mode of
        :func:`resize_depth_gt`). ``sparse`` comes from the head, else from
        the loader's annotation, which states it for loaders that always
        return sparse maps (DrivingStereo), as euler-loading's preprocessor
        reads it. ``invalid_sentinel`` is the declared invalid value
        (:func:`declared_invalid_sentinel`), or ``None``.
    """
    meta = dataset.get_modality_metadata(modality)
    sparse = meta.get("sparse")
    if sparse is None:
        sparse = _loader_annotation_meta(dataset, modality).get("sparse", False)

    return {
        "radial_depth": bool(meta.get("radial_depth", True)),
        "sparse": sparse is True,
        "invalid_sentinel": declared_invalid_sentinel(dataset, modality),
    }


def declared_invalid_sentinel(dataset: Any, modality: str) -> Optional[float]:
    """The finite value a modality declares for pixels without a valid value.

    Reads euler-dataset-contract's ``invalid.sentinel`` from the head's meta,
    else from the loader annotation's meta (as for ``sparse``). ``None`` when
    neither declares one, or declares ``null``.
    """
    lookup = getattr(dataset, "get_modality_metadata", None)
    try:
        meta = lookup(modality) if callable(lookup) else None
    except Exception:
        meta = None
    meta = meta if isinstance(meta, Mapping) else {}
    invalid = (
        meta["invalid"]
        if "invalid" in meta
        else _loader_annotation_meta(dataset, modality).get("invalid")
    )
    sentinel = invalid.get("sentinel") if isinstance(invalid, Mapping) else None
    if (
        isinstance(sentinel, bool)
        or not isinstance(sentinel, (int, float))
        or not np.isfinite(sentinel)
    ):
        return None
    return float(sentinel)


def mask_invalid_sentinel(
    values: np.ndarray, sentinel: Optional[float], *, fill: float
) -> np.ndarray:
    """Replace the declared invalid *sentinel* in *values* with *fill*.

    euler-eval treats GT depth that is non-finite or not positive as invalid
    and drops non-finite predictions, so GT uses ``fill=0`` and predictions
    ``fill=NaN``. A sentinel those rules already exclude costs nothing.
    """
    if sentinel is None:
        return values
    values = np.asarray(values)
    hit = values == sentinel
    if not hit.any() or (fill == sentinel):
        return values
    out = values.astype(values.dtype if values.dtype.kind == "f" else np.float32, copy=True)
    out[hit] = fill
    return out


def _loader_annotation_meta(dataset: Any, modality: str) -> dict[str, Any]:
    """Return the ``meta`` of a modality's loader annotation, or ``{}``."""
    lookup = getattr(dataset, "get_loader_meta", None)
    if not callable(lookup):
        return {}
    try:
        annotation = lookup(modality)
    except Exception:
        return {}
    meta = annotation.get("meta") if isinstance(annotation, dict) else None
    return meta if isinstance(meta, dict) else {}


def get_sparse_depth_metadata(dataset: MultiModalDataset) -> dict[str, Any]:
    """Extract sparse GT and dense prediction metadata for sparse-depth eval."""
    sparse_meta = dataset.get_modality_metadata("gt")
    pred_meta = dataset.get_modality_metadata("pred")

    return {
        "representation": sparse_meta.get("representation", "point_cloud"),
        "coordinate_unit": sparse_meta.get("coordinate_unit", "meters"),
        "point_columns": sparse_meta.get("columns"),
        "pred_radial_depth": bool(pred_meta.get("radial_depth", True)),
    }


def get_rgb_metadata(dataset: MultiModalDataset) -> dict[str, Any]:
    """Extract RGB modality metadata from a dataset's reference.

    The reference is the clear GT when configured, else the degraded input.

    Returns:
        Dict with ``rgb_range`` ([min, max]).
    """
    reference = "gt" if "gt" in dataset.modality_paths() else "input"
    meta = dataset.get_modality_metadata(reference)
    return {
        "rgb_range": meta.get("rgb_range", [0.0, 1.0]),
    }


# ---------------------------------------------------------------------------
# Rays (spherical direction map) dataset construction
# ---------------------------------------------------------------------------


def build_rays_eval_dataset(
    gt_rays_path: str,
    pred_rays_path: str,
    calibration_path: Optional[str] = None,
    gt_rays_split: Optional[str] = None,
    pred_rays_split: Optional[str] = None,
    calibration_split: Optional[str] = None,
) -> MultiModalDataset:
    """Build a MultiModalDataset for rays (spherical direction map) evaluation.

    The returned dataset yields samples with keys ``"gt"``, ``"pred"``,
    and optionally ``"calibration"``.

    Args:
        gt_rays_path: Path to GT ray direction map dataset root.
        pred_rays_path: Path to predicted ray direction map dataset root.
        calibration_path: Optional path to calibration dataset (used for
            FoV domain classification).
        gt_rays_split: Optional split name for the GT rays modality.
        pred_rays_split: Optional split name for the prediction rays modality.
        calibration_split: Optional split name for the calibration modality.

    Returns:
        A MultiModalDataset instance.
    """
    modalities = {
        "gt": _modality(
            path=gt_rays_path,
            modality_key="rays",
            split=gt_rays_split,
        ),
        "pred": _modality(
            path=pred_rays_path,
            modality_key="rays",
            used_as="output",
            split=pred_rays_split,
        ),
    }

    hierarchical = {}
    if calibration_path is not None:
        hierarchical["calibration"] = _modality(
            path=calibration_path,
            modality_key="calibration",
            split=calibration_split,
        )

    return MultiModalDataset(
        modalities=modalities,
        hierarchical_modalities=hierarchical if hierarchical else None,
    )


def get_rays_metadata(dataset: MultiModalDataset) -> dict[str, Any]:
    """Extract rays modality metadata from a dataset.

    Returns:
        Dict with ``fov_domain`` (``"sfov"``, ``"lfov"``, or ``"pano"``).
        Falls back to ``"lfov"`` when metadata is not available.
    """
    meta = dataset.get_modality_metadata("gt")
    return {
        "fov_domain": meta.get("fov_domain", None),
    }


# ---------------------------------------------------------------------------
# Points-3D (per-pixel 3D point map) construction and geometry
# ---------------------------------------------------------------------------


def to_numpy_points_3d(data: Any) -> np.ndarray:
    """Convert a per-pixel 3D point map to ``(H, W, 3)`` float32 numpy array.

    A ``points_3d`` map stores camera-frame ``(X, Y, Z)`` coordinates in
    metres at every pixel.  Accepts torch tensors ``(3, H, W)`` or
    ``(H, W, 3)``, or numpy arrays, plus single-sample variants.  The
    returned points are **not** re-scaled or re-centred; callers apply gauge
    alignment explicitly.
    """
    # Layout handling is identical to direction maps (HWC vs CHW, batch dim).
    return to_numpy_directions(data)


def unproject_depth_to_points(
    depth: np.ndarray,
    intrinsics_K: np.ndarray,
    depth_is_radial: bool = False,
) -> np.ndarray:
    """Unproject a depth map into a ``(H, W, 3)`` camera-frame point map.

    Args:
        depth: Depth map ``(H, W)`` in metres.
        intrinsics_K: ``(3, 3)`` pinhole camera matrix.
        depth_is_radial: If True, ``depth`` stores Euclidean (radial)
            distance from the camera centre and is converted back to planar
            ``Z`` before unprojection.  If False, ``depth`` is planar ``Z``.

    Returns:
        ``(H, W, 3)`` float32 point map.  Pixels with non-finite or
        non-positive depth are set to ``0``.
    """
    height, width = depth.shape
    fx = float(intrinsics_K[0, 0])
    fy = float(intrinsics_K[1, 1])
    cx = float(intrinsics_K[0, 2])
    cy = float(intrinsics_K[1, 2])

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x_norm = (u - cx) / fx
    y_norm = (v - cy) / fy

    z = depth.astype(np.float32)
    if depth_is_radial:
        ray_norm = np.sqrt(x_norm**2 + y_norm**2 + 1.0)
        z = z / ray_norm.astype(np.float32)

    points = np.stack([x_norm * z, y_norm * z, z], axis=-1).astype(np.float32)
    invalid = ~(np.isfinite(depth) & (depth > 0))
    points[invalid] = 0.0
    return points


def fit_scale_transform(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
) -> float:
    """Fit a single global isotropic scale mapping *pred* onto *gt*.

    Solves ``min_s Σ ‖s·p_i − g_i‖²`` in closed form, which gives
    ``s = Σ(p_i·g_i) / Σ‖p_i‖²``.

    Args:
        pred_points: ``(N, 3)`` predicted points.
        gt_points: ``(N, 3)`` ground-truth points.

    Returns:
        The fitted scalar scale ``s`` (``1.0`` if degenerate).
    """
    pred = np.asarray(pred_points, dtype=np.float64)
    gt = np.asarray(gt_points, dtype=np.float64)
    if pred.shape[0] < 1:
        return 1.0
    denom = float(np.sum(pred * pred))
    if denom <= 1e-12:
        return 1.0
    return float(np.sum(pred * gt) / denom)


def fit_similarity_transform(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    with_scale: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit a similarity transform ``g ≈ s·R·p + t`` (Umeyama, 1991).

    Closed-form least-squares estimate of scale ``s``, rotation ``R``
    (``3×3``), and translation ``t`` (``3``) that best map predicted points
    onto ground-truth points, given known correspondence.

    Args:
        pred_points: ``(N, 3)`` predicted (source) points.
        gt_points: ``(N, 3)`` ground-truth (target) points.
        with_scale: If False, fix ``s = 1`` (rigid transform).

    Returns:
        ``(s, R, t)``.  Falls back to identity if the problem is degenerate
        (fewer than 3 points or a rank-deficient point set).
    """
    pred = np.asarray(pred_points, dtype=np.float64)
    gt = np.asarray(gt_points, dtype=np.float64)

    identity = (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
    n = pred.shape[0]
    if n < 3:
        return identity

    mu_p = pred.mean(axis=0)
    mu_g = gt.mean(axis=0)
    pred_c = pred - mu_p
    gt_c = gt - mu_g

    var_p = float(np.mean(np.sum(pred_c**2, axis=1)))
    if var_p <= 1e-12:
        return identity

    sigma = (gt_c.T @ pred_c) / n
    U, D, Vt = np.linalg.svd(sigma)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt

    if with_scale:
        s = float(np.trace(np.diag(D) @ S) / var_p)
    else:
        s = 1.0

    t = mu_g - s * (R @ mu_p)
    return s, R.astype(np.float32), t.astype(np.float32)


def apply_point_transform(
    points: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Apply ``s·R·p + t`` to a point array of shape ``(..., 3)``."""
    pts = np.asarray(points, dtype=np.float32)
    rotated = pts @ np.asarray(rotation, dtype=np.float32).T
    return (scale * rotated + np.asarray(translation, dtype=np.float32)).astype(
        np.float32
    )


def decompose_point_errors(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
) -> dict[str, np.ndarray]:
    """Decompose per-point 3D errors relative to the GT ray direction.

    For each corresponding pair, the error vector ``d = pred − gt`` is split
    into the component **along** the GT ray (radial ≈ depth error) and the
    component **perpendicular** to it (lateral ≈ camera-model error).

    Args:
        pred_points: ``(N, 3)`` predicted points.
        gt_points: ``(N, 3)`` ground-truth points.

    Returns:
        Dict of per-point ``(N,)`` arrays: ``euclidean`` (‖d‖), ``radial``
        (signed along the GT ray), ``lateral`` (perpendicular magnitude),
        ``relative`` (‖d‖ / ‖gt‖), and ``gt_norm`` (‖gt‖).
    """
    pred = np.asarray(pred_points, dtype=np.float64)
    gt = np.asarray(gt_points, dtype=np.float64)

    diff = pred - gt
    euclidean = np.linalg.norm(diff, axis=1)
    gt_norm = np.linalg.norm(gt, axis=1)
    safe_norm = np.clip(gt_norm, 1e-8, None)
    ray = gt / safe_norm[:, None]

    radial = np.sum(diff * ray, axis=1)
    lateral_vec = diff - radial[:, None] * ray
    lateral = np.linalg.norm(lateral_vec, axis=1)
    relative = euclidean / safe_norm

    return {
        "euclidean": euclidean.astype(np.float32),
        "radial": radial.astype(np.float32),
        "lateral": lateral.astype(np.float32),
        "relative": relative.astype(np.float32),
        "gt_norm": gt_norm.astype(np.float32),
    }


def build_points_3d_eval_dataset(
    pred_points_3d_path: str,
    gt_points_3d_path: Optional[str] = None,
    gt_depth_path: Optional[str] = None,
    intrinsics_path: Optional[str] = None,
    calibration_path: Optional[str] = None,
    segmentation_path: Optional[str] = None,
    pred_points_3d_split: Optional[str] = None,
    gt_points_3d_split: Optional[str] = None,
    gt_depth_split: Optional[str] = None,
    intrinsics_split: Optional[str] = None,
    calibration_split: Optional[str] = None,
    segmentation_split: Optional[str] = None,
    segmentation_modality_key: str = "segmentation",
) -> MultiModalDataset:
    """Build a MultiModalDataset for points_3d (3D point map) evaluation.

    The prediction is always a ``(H, W, 3)`` camera-frame point map under
    ``"pred"``.  The ground truth comes from one of two sources:

    - **Explicit point map** — pass ``gt_points_3d_path``; ``"gt"`` is loaded as
      a ``points_3d`` modality directly.
    - **Synthesized from depth** — pass ``gt_depth_path`` (and intrinsics via
      ``intrinsics_path`` or ``calibration_path``); ``"gt"`` is loaded as a
      ``depth`` modality and the evaluator unprojects it to a point map on the
      fly. Use this when no precomputed GT point map exists.

    Optional ``"calibration"``/``"intrinsics"`` are used for FoV-domain
    classification of the angular metrics (and are *required* for the depth
    synthesis path), and ``"segmentation"`` enables sky masking.
    """
    if gt_points_3d_path is not None:
        gt_modality = _modality(
            path=gt_points_3d_path,
            modality_key="points_3d",
            split=gt_points_3d_split,
        )
    elif gt_depth_path is not None:
        gt_modality = _modality(
            path=gt_depth_path,
            modality_key="depth",
            split=gt_depth_split,
        )
    else:
        raise ValueError(
            "build_points_3d_eval_dataset requires either gt_points_3d_path or "
            "gt_depth_path (with intrinsics) for the ground-truth point map."
        )

    modalities = {
        "gt": gt_modality,
        "pred": _modality(
            path=pred_points_3d_path,
            modality_key="points_3d",
            used_as="output",
            split=pred_points_3d_split,
        ),
    }

    hierarchical = {}
    if intrinsics_path is not None:
        hierarchical["intrinsics"] = _modality(
            path=intrinsics_path,
            modality_key="intrinsics",
            split=intrinsics_split,
        )
    if calibration_path is not None:
        hierarchical["calibration"] = _modality(
            path=calibration_path,
            modality_key="calibration",
            split=calibration_split,
        )
    if segmentation_path is not None:
        modalities["segmentation"] = _gt_segmentation_modality(
            segmentation_path,
            modality_key=segmentation_modality_key,
            split=segmentation_split,
        )

    return MultiModalDataset(
        modalities=modalities,
        hierarchical_modalities=hierarchical if hierarchical else None,
    )


def get_points_3d_metadata(dataset: MultiModalDataset) -> dict[str, Any]:
    """Extract points_3d modality metadata from a dataset.

    Returns:
        Dict with ``fov_domain`` (``"sfov"``/``"lfov"``/``"pano"`` or None)
        and ``coordinate_unit`` (default ``"meters"``).
    """
    meta = dataset.get_modality_metadata("gt")
    return {
        "fov_domain": meta.get("fov_domain", None),
        "coordinate_unit": meta.get("coordinate_unit", "meters"),
    }
