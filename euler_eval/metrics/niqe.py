"""Natural Image Quality Evaluator (NIQE).

This is a NumPy/SciPy implementation of the two-scale natural-scene
statistics model from Mittal, Soundararajan, and Bovik.  The pristine model
parameters are bundled from the official LIVE release so the metric does not
download weights at runtime.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy import ndimage
from scipy.special import gamma

from ._natural_scene_models import (
    NIQE_PRISTINE_COVARIANCE,
    NIQE_PRISTINE_MEAN,
)

_BLOCK_SIZE = 96
_AGGD_SHAPES = np.arange(0.2, 10.001, 0.001, dtype=np.float64)
_AGGD_RATIO = gamma(2.0 / _AGGD_SHAPES) ** 2 / (
    gamma(1.0 / _AGGD_SHAPES) * gamma(3.0 / _AGGD_SHAPES)
)


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Validate an evaluator RGB image and restore its 8-bit representation."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            f"Expected an RGB image with shape (H, W, 3), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Natural-scene metrics require finite RGB values")

    array = array.astype(np.float64, copy=False)
    tolerance = 1e-6
    if array.size and (
        float(np.min(array)) < -tolerance or float(np.max(array)) > 1.0 + tolerance
    ):
        raise ValueError("Natural-scene metrics expect RGB values in the [0, 1] range")
    return np.floor(np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _rgb_to_gray_255(image: np.ndarray) -> np.ndarray:
    """Match ``double(rgb2gray(uint8_rgb))`` used by the reference code."""
    weights = np.array(
        [0.298936021293775, 0.587043074451121, 0.114020904255103],
        dtype=np.float64,
    )
    gray = np.tensordot(image.astype(np.float64), weights, axes=([2], [0]))
    return np.floor(np.clip(gray, 0.0, 255.0) + 0.5)


@lru_cache(maxsize=1)
def _gaussian_window() -> np.ndarray:
    coordinates = np.arange(-3, 4, dtype=np.float64)
    xx, yy = np.meshgrid(coordinates, coordinates)
    sigma = 7.0 / 6.0
    window = np.exp(-((xx * xx) + (yy * yy)) / (2.0 * sigma * sigma))
    return window / np.sum(window)


def _estimate_aggd_parameters(values: np.ndarray) -> tuple[float, float, float]:
    """Estimate asymmetric generalized Gaussian distribution parameters."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    negative = flat[flat < 0.0]
    positive = flat[flat > 0.0]
    left_std = float(np.sqrt(np.mean(negative * negative))) if negative.size else 0.0
    right_std = float(np.sqrt(np.mean(positive * positive))) if positive.size else 0.0

    mean_square = float(np.mean(flat * flat))
    if not np.isfinite(mean_square) or mean_square <= np.finfo(np.float64).eps:
        # A constant block has a degenerate limiting distribution.  Returning
        # zero scales keeps the image-level fit well-defined.
        return float(_AGGD_SHAPES[-1]), 0.0, 0.0

    epsilon = np.finfo(np.float64).eps
    gamma_hat = max(left_std, epsilon) / max(right_std, epsilon)
    r_hat = float(np.mean(np.abs(flat))) ** 2 / mean_square
    denominator = (gamma_hat * gamma_hat + 1.0) ** 2
    normalized_r_hat = r_hat * (gamma_hat**3 + 1.0) * (gamma_hat + 1.0) / denominator
    index = int(np.argmin((_AGGD_RATIO - normalized_r_hat) ** 2))
    alpha = float(_AGGD_SHAPES[index])
    scale = float(np.sqrt(gamma(1.0 / alpha) / gamma(3.0 / alpha)))
    return alpha, left_std * scale, right_std * scale


def _compute_block_features(block: np.ndarray) -> np.ndarray:
    alpha, beta_left, beta_right = _estimate_aggd_parameters(block)
    features = [alpha, (beta_left + beta_right) / 2.0]

    for shift in ((0, 1), (1, 0), (1, 1), (1, -1)):
        shifted = np.roll(block, shift=shift, axis=(0, 1))
        alpha, beta_left, beta_right = _estimate_aggd_parameters(block * shifted)
        mean = (beta_right - beta_left) * gamma(2.0 / alpha) / gamma(1.0 / alpha)
        features.extend((alpha, float(mean), beta_left, beta_right))
    return np.asarray(features, dtype=np.float64)


def _cubic_kernel(distance: np.ndarray) -> np.ndarray:
    absolute = np.abs(distance)
    absolute_squared = absolute * absolute
    absolute_cubed = absolute_squared * absolute
    inner = (1.5 * absolute_cubed - 2.5 * absolute_squared + 1.0) * (absolute <= 1.0)
    outer = (-0.5 * absolute_cubed + 2.5 * absolute_squared - 4.0 * absolute + 2.0) * (
        (absolute > 1.0) & (absolute <= 2.0)
    )
    return inner + outer


def _resize_weights_indices(
    input_length: int, output_length: int, scale: float
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Build MATLAB-compatible antialiased bicubic resize weights."""
    kernel_width = 4.0 / scale if scale < 1.0 else 4.0
    output_coordinates = np.arange(1, output_length + 1, dtype=np.float64)
    input_coordinates = output_coordinates / scale + 0.5 * (1.0 - 1.0 / scale)
    left = np.floor(input_coordinates - kernel_width / 2.0)
    support = int(math.ceil(kernel_width)) + 2
    indices = left[:, None] + np.arange(support, dtype=np.float64)[None, :]
    distances = input_coordinates[:, None] - indices
    weights = (
        scale * _cubic_kernel(distances * scale)
        if scale < 1.0
        else _cubic_kernel(distances)
    )
    weights /= np.sum(weights, axis=1, keepdims=True)

    if np.any(weights[:, 0] == 0.0):
        weights = weights[:, 1:-1]
        indices = indices[:, 1:-1]
    if np.any(weights[:, -1] == 0.0):
        weights = weights[:, :-2]
        indices = indices[:, :-2]

    symmetric_start = int(-np.min(indices) + 1)
    symmetric_end = int(np.max(indices) - input_length)
    indices = (indices + symmetric_start - 1).astype(np.int64)
    return weights, indices, symmetric_start, symmetric_end


def _matlab_resize_half(image: np.ndarray) -> np.ndarray:
    """Resize a grayscale image by 0.5 using MATLAB-style bicubic filtering."""
    scale = 0.5
    input_height, input_width = image.shape
    output_height = int(math.ceil(input_height * scale))
    output_width = int(math.ceil(input_width * scale))
    weights_h, indices_h, pad_h_start, pad_h_end = _resize_weights_indices(
        input_height, output_height, scale
    )
    weights_w, indices_w, pad_w_start, pad_w_end = _resize_weights_indices(
        input_width, output_width, scale
    )

    padded = np.pad(
        image,
        ((pad_h_start, pad_h_end), (0, 0)),
        mode="symmetric",
    )
    intermediate = np.empty((output_height, input_width), dtype=np.float64)
    for row in range(output_height):
        intermediate[row] = np.tensordot(
            weights_h[row], padded[indices_h[row]], axes=(0, 0)
        )

    padded = np.pad(
        intermediate,
        ((0, 0), (pad_w_start, pad_w_end)),
        mode="symmetric",
    )
    output = np.empty((output_height, output_width), dtype=np.float64)
    for column in range(output_width):
        output[:, column] = np.tensordot(
            padded[:, indices_w[column]], weights_w[column], axes=(1, 0)
        )
    return output


def _extract_niqe_features(gray: np.ndarray) -> np.ndarray:
    height, width = gray.shape
    block_rows = height // _BLOCK_SIZE
    block_columns = width // _BLOCK_SIZE
    if block_rows == 0 or block_columns == 0:
        raise ValueError("NIQE requires an image of at least 96x96 pixels")

    gray = gray[: block_rows * _BLOCK_SIZE, : block_columns * _BLOCK_SIZE]
    scale_features = []
    current = gray
    window = _gaussian_window()

    for scale in (1, 2):
        local_mean = ndimage.correlate(current, window, mode="nearest")
        local_variance = np.abs(
            ndimage.correlate(current * current, window, mode="nearest")
            - local_mean * local_mean
        )
        normalized = (current - local_mean) / (np.sqrt(local_variance) + 1.0)
        block_height = _BLOCK_SIZE // scale
        block_width = _BLOCK_SIZE // scale
        features = []
        for column in range(block_columns):
            for row in range(block_rows):
                block = normalized[
                    row * block_height : (row + 1) * block_height,
                    column * block_width : (column + 1) * block_width,
                ]
                features.append(_compute_block_features(block))
        scale_features.append(np.asarray(features, dtype=np.float64))
        if scale == 1:
            current = _matlab_resize_half(current / 255.0) * 255.0

    return np.concatenate(scale_features, axis=1)


def compute_niqe(image: np.ndarray) -> float:
    """Compute the no-reference NIQE score for an RGB image in ``[0, 1]``.

    Lower scores indicate closer agreement with the model of pristine natural
    images. Images smaller than the reference model's 96-pixel block size are
    rejected rather than resized because adapting the block size changes the
    metric.
    """
    gray = _rgb_to_gray_255(_to_uint8_rgb(image))
    features = _extract_niqe_features(gray)
    finite_rows = features[np.all(np.isfinite(features), axis=1)]
    if finite_rows.size == 0:
        raise ValueError("NIQE could not extract finite natural-scene features")

    distorted_mean = np.mean(finite_rows, axis=0)
    if finite_rows.shape[0] == 1:
        distorted_covariance = np.zeros((36, 36), dtype=np.float64)
    else:
        distorted_covariance = np.cov(finite_rows, rowvar=False)
    average_covariance = (NIQE_PRISTINE_COVARIANCE + distorted_covariance) / 2.0
    difference = NIQE_PRISTINE_MEAN - distorted_mean
    inverse_covariance = np.linalg.pinv(average_covariance, hermitian=True)
    squared_distance = float(difference @ inverse_covariance @ difference)
    if squared_distance < 0.0 and abs(squared_distance) < 1e-10:
        squared_distance = 0.0
    if not np.isfinite(squared_distance) or squared_distance < 0.0:
        raise ValueError("NIQE produced an invalid model distance")
    return float(np.sqrt(squared_distance))
