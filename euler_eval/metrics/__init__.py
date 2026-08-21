"""Depth, RGB, rays, and points_3d evaluation metrics package.

This package provides various metrics for comparing depth maps, RGB images,
spherical direction maps (camera rays), and per-pixel 3D point maps:

Depth metrics:
- Image quality: PSNR, SSIM, LPIPS, FID, KID
- Depth-specific: AbsRel, RMSE, Scale-Invariant Log Error
- Geometric: Normal Consistency, Depth-Edge F1

RGB metrics:
- Image quality: PSNR, SSIM, LPIPS
- Edge F1
- Tail errors (p95/p99)
- High-frequency energy ratio
- Depth-binned photometric error (MAE/MSE per depth bin)
- Domain-specific no-reference metrics: NIQE and FADE (dehazing)

Rays metrics:
- ρ_A: AUC of angular accuracy curve between predicted and GT ray directions

Points-3D metrics:
- Point errors with absolute/relative threshold accuracy
- Error decomposition (along-ray vs lateral)
- Geometric: point-normal angles, point-edge F1
- Cloud distance and F-score AUC
"""

from .absrel import aggregate_absrel, compute_absrel
from .daniel_error import compute_sce
from .depth_binned_error import (
    aggregate_depth_binned_errors,
    compute_depth_binned_mae,
    compute_depth_binned_mse,
    compute_depth_binned_photometric_error,
)
from .depth_edge_f1 import aggregate_edge_f1, compute_depth_edge_f1
from .depth_standard import (
    STANDARD_DEPTH_METRIC_KEYS,
    append_standard_depth_metrics,
    compute_standard_depth_metrics,
    init_standard_depth_store,
    summarize_standard_depth_store,
)
from .fade import compute_fade
from .fid_kid import FIDKIDMetric, compute_clean_fid
from .gpu_depth_batch import GPUDepthMetricsBatcher
from .gpu_image_batch import GPUImageMetricsBatcher
from .high_freq_energy import (
    aggregate_high_freq_metrics,
    compute_high_freq_energy_comparison,
    compute_high_freq_energy_ratio,
)
from .lpips_metric import LPIPSMetric
from .niqe import compute_niqe
from .normal_consistency import (
    aggregate_normal_consistency,
    compute_normal_angles,
)
from .points3d_cloud import (
    DEFAULT_MAX_POINTS as POINTS3D_DEFAULT_MAX_POINTS,
    FSCORE_THRESHOLDS as POINTS3D_FSCORE_THRESHOLDS,
    compute_cloud_distance_metrics,
    compute_fscore_auc,
    compute_sparse_cloud_distance_metrics,
)
from .points3d_decomposition import compute_decomposition_metrics
from .points3d_distance import (
    ABS_THRESHOLDS as POINTS3D_ABS_THRESHOLDS,
    REL_THRESHOLDS as POINTS3D_REL_THRESHOLDS,
    compute_point_error_metrics,
    threshold_key as points3d_threshold_key,
)
from .points3d_geometry import (
    compute_point_edge_f1,
    compute_point_normal_angles,
    points_to_normals,
)
from .psnr import compute_psnr
from .rgb_edge_f1 import (
    aggregate_rgb_edge_f1,
    compute_rgb_edge_f1,
    detect_rgb_edges,
)
from .rgb_lpips import RGBLPIPSMetric
from .rgb_psnr_ssim import compute_rgb_psnr, compute_rgb_ssim
from .rho_a import (
    FOV_THRESHOLDS,
    aggregate_angular_errors,
    aggregate_rho_a,
    classify_fov_domain,
    compute_angular_errors,
    compute_rho_a,
    get_threshold_for_domain,
)
from .rmse import aggregate_rmse, compute_rmse_per_pixel
from .scale_invariant_log import (
    aggregate_silog,
    compute_scale_invariant_log_error,
    compute_silog_per_pixel,
)
from .ssim import compute_ssim
from .tail_errors import compute_tail_errors
from .utils import (
    _BENCHMARK_BIN_NAMES,
    convert_planar_to_radial,
    get_benchmark_depth_bins,
    get_depth_bins,
)

__all__ = [
    # Depth image quality metrics
    "compute_psnr",
    "compute_ssim",
    "LPIPSMetric",
    "compute_clean_fid",
    "FIDKIDMetric",
    # Depth-specific metrics
    "compute_absrel",
    "aggregate_absrel",
    "compute_rmse_per_pixel",
    "aggregate_rmse",
    "compute_scale_invariant_log_error",
    "compute_silog_per_pixel",
    "aggregate_silog",
    # Depth geometric metrics
    "compute_normal_angles",
    "aggregate_normal_consistency",
    "compute_depth_edge_f1",
    "aggregate_edge_f1",
    "STANDARD_DEPTH_METRIC_KEYS",
    "compute_standard_depth_metrics",
    "init_standard_depth_store",
    "append_standard_depth_metrics",
    "summarize_standard_depth_store",
    # RGB image quality metrics
    "compute_rgb_psnr",
    "compute_rgb_ssim",
    "RGBLPIPSMetric",
    "compute_sce",
    "compute_niqe",
    "compute_fade",
    # RGB depth-binned metrics
    "compute_depth_binned_mae",
    "compute_depth_binned_mse",
    "compute_depth_binned_photometric_error",
    "aggregate_depth_binned_errors",
    # RGB edge metrics
    "compute_rgb_edge_f1",
    "detect_rgb_edges",
    "aggregate_rgb_edge_f1",
    # RGB tail errors
    "compute_tail_errors",
    # RGB high-frequency metrics
    "compute_high_freq_energy_ratio",
    "compute_high_freq_energy_comparison",
    "aggregate_high_freq_metrics",
    # Rays metrics
    "compute_angular_errors",
    "compute_rho_a",
    "aggregate_rho_a",
    "aggregate_angular_errors",
    "classify_fov_domain",
    "get_threshold_for_domain",
    "FOV_THRESHOLDS",
    # Points-3D metrics
    "compute_point_error_metrics",
    "points3d_threshold_key",
    "POINTS3D_ABS_THRESHOLDS",
    "POINTS3D_REL_THRESHOLDS",
    "compute_decomposition_metrics",
    "points_to_normals",
    "compute_point_normal_angles",
    "compute_point_edge_f1",
    "compute_cloud_distance_metrics",
    "compute_fscore_auc",
    "compute_sparse_cloud_distance_metrics",
    "POINTS3D_FSCORE_THRESHOLDS",
    "POINTS3D_DEFAULT_MAX_POINTS",
    # Utilities
    "convert_planar_to_radial",
    "get_depth_bins",
    "get_benchmark_depth_bins",
    "_BENCHMARK_BIN_NAMES",
    # GPU-batched image metrics
    "GPUImageMetricsBatcher",
    "GPUDepthMetricsBatcher",
]
