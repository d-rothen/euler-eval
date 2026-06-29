"""Radial/lateral error decomposition for the ``points_3d`` modality.

The per-point 3D error vector ``d = pred − gt`` is decomposed (by
:func:`euler_eval.data.decompose_point_errors`) into the component **along**
the GT ray (radial, ≈ depth error) and the component **perpendicular** to it
(lateral, ≈ camera-model error).  This module turns those per-point arrays
into per-image summaries, including ``lateral_fraction`` — a single scalar in
``[0, 1]`` indicating whether the error is depth-dominated (→0) or
camera-model-dominated (→1).

The angular ray error and ρ_A scores that round out this category reuse
``euler_eval.metrics.rho_a`` directly on the point-map directions.
"""

from typing import Optional

import numpy as np


def compute_decomposition_metrics(
    radial: np.ndarray,
    lateral: np.ndarray,
) -> Optional[dict]:
    """Per-image radial/lateral decomposition metrics.

    Args:
        radial: ``(N,)`` signed per-point radial errors (along the GT ray).
        lateral: ``(N,)`` non-negative per-point lateral errors.

    Returns:
        Dict of scalar metrics, or ``None`` if there are no valid points.
    """
    rad = np.asarray(radial, dtype=np.float64).reshape(-1)
    lat = np.asarray(lateral, dtype=np.float64).reshape(-1)
    if rad.size == 0:
        return None

    radial_mae = float(np.mean(np.abs(rad)))
    lateral_mae = float(np.mean(lat))
    denom = radial_mae + lateral_mae
    lateral_fraction = float(lateral_mae / denom) if denom > 0 else 0.0

    return {
        "radial_mae": radial_mae,
        "radial_rmse": float(np.sqrt(np.mean(rad**2))),
        "lateral_mae": lateral_mae,
        "lateral_rmse": float(np.sqrt(np.mean(lat**2))),
        "lateral_fraction": lateral_fraction,
    }
