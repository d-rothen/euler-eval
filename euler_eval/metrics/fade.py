"""Fog Aware Density Evaluator (FADE).

FADE is a no-reference perceptual fog-density metric.  This implementation
follows the twelve patch features and foggy/fog-free multivariate Gaussian
models from Choi, You, and Bovik.  Lower values indicate less perceived fog.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import ndimage

from ._natural_scene_models import (
    FADE_FOGFREE_COVARIANCE,
    FADE_FOGFREE_MEAN,
    FADE_FOGGY_COVARIANCE,
    FADE_FOGGY_MEAN,
)
from .niqe import _gaussian_window, _rgb_to_gray_255, _to_uint8_rgb

_PATCH_SIZE = 8


def _split_blocks(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    if height % _PATCH_SIZE or width % _PATCH_SIZE:
        raise ValueError("FADE feature input must align to its 8x8 patch grid")
    return image.reshape(
        height // _PATCH_SIZE,
        _PATCH_SIZE,
        width // _PATCH_SIZE,
        _PATCH_SIZE,
    ).transpose(0, 2, 1, 3)


def _sample_nan_variance(blocks: np.ndarray) -> np.ndarray:
    finite = np.isfinite(blocks)
    counts = np.sum(finite, axis=(-2, -1))
    sums = np.sum(np.where(finite, blocks, 0.0), axis=(-2, -1))
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    centered = np.where(finite, blocks - means[..., None, None], 0.0)
    sum_squares = np.sum(centered * centered, axis=(-2, -1))
    result = np.full(counts.shape, np.nan, dtype=np.float64)
    result[counts == 1] = 0.0
    valid = counts > 1
    result[valid] = sum_squares[valid] / (counts[valid] - 1.0)
    return result


@lru_cache(maxsize=1)
def _contrast_kernels() -> tuple[np.ndarray, np.ndarray]:
    sigma = 3.25
    # MATLAB's -9.75:1:9.75 contains 20 samples and ends at 9.25.
    coordinates = -9.75 + np.arange(20, dtype=np.float64)
    gaussian = np.exp(-(coordinates * coordinates) / (2.0 * sigma * sigma))
    gaussian /= np.sum(gaussian)
    kernel = ((coordinates * coordinates) / sigma**4 - 1.0 / sigma**2) * gaussian
    kernel -= np.sum(kernel) / kernel.size
    kernel /= np.sum(0.5 * coordinates * coordinates * kernel)
    return kernel.reshape(1, -1), kernel.reshape(-1, 1)


def _add_contrast_border(image: np.ndarray) -> np.ndarray:
    # Equivalent to FADE's border_in(..., 20): copy ten samples from each edge.
    vertical = np.concatenate((image[:10], image, image[-10:]), axis=0)
    return np.concatenate(
        (vertical[:, :10], vertical, vertical[:, -10:]),
        axis=1,
    )


def _convolve_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Match MATLAB ``conv2(..., 'same')`` alignment for even 1-D kernels."""
    if kernel.shape[0] == 1:
        return ndimage.convolve1d(
            image, kernel.reshape(-1), axis=1, mode="constant", cval=0.0
        )
    return ndimage.convolve1d(
        image, kernel.reshape(-1), axis=0, mode="constant", cval=0.0
    )


def _contrast_energy(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = image.astype(np.float64, copy=False)
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    channels = (
        0.299 * red + 0.587 * green + 0.114 * blue,
        0.5 * (red + green) - blue,
        red - green,
    )
    thresholds = (
        9.225496406318721e-4 * 255.0,
        8.969246659629488e-4 * 255.0,
        2.069284034165411e-4 * 255.0,
    )
    horizontal, vertical = _contrast_kernels()
    outputs = []
    for channel, threshold in zip(channels, thresholds):
        padded = _add_contrast_border(channel)
        response_x = _convolve_same(padded, horizontal)
        response_y = _convolve_same(padded, vertical)
        contrast = np.sqrt(response_x * response_x + response_y * response_y)
        contrast = contrast[10:-10, 10:-10]
        maximum = float(np.max(contrast))
        if maximum <= np.finfo(np.float64).eps:
            outputs.append(np.zeros_like(channel))
            continue
        normalized = contrast * maximum / (contrast + maximum * 0.1)
        outputs.append(np.maximum(normalized - threshold, 0.0))
    return tuple(outputs)


def _block_entropy(gray_uint8: np.ndarray) -> np.ndarray:
    blocks = _split_blocks(gray_uint8)
    result = np.empty(blocks.shape[:2], dtype=np.float64)
    for row in range(blocks.shape[0]):
        for column in range(blocks.shape[1]):
            counts = np.bincount(blocks[row, column].reshape(-1), minlength=256)
            probabilities = counts[counts > 0].astype(np.float64) / 64.0
            result[row, column] = -float(np.sum(probabilities * np.log2(probabilities)))
    return result


def _saturation(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float64) / 255.0
    maximum = np.max(rgb, axis=-1)
    minimum = np.min(rgb, axis=-1)
    return np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 0.0,
    )


def _extract_fade_features(image: np.ndarray) -> np.ndarray:
    image = _to_uint8_rgb(image)
    height = (image.shape[0] // _PATCH_SIZE) * _PATCH_SIZE
    width = (image.shape[1] // _PATCH_SIZE) * _PATCH_SIZE
    if height < 16 or width < 16:
        raise ValueError("FADE requires an image of at least 16x16 pixels")
    image = image[:height, :width]
    image_float = image.astype(np.float64)
    red, green, blue = (
        image_float[..., 0],
        image_float[..., 1],
        image_float[..., 2],
    )
    gray = _rgb_to_gray_255(image)

    window = _gaussian_window()
    local_mean = ndimage.correlate(gray, window, mode="nearest")
    second_moment = ndimage.correlate(gray * gray, window, mode="nearest")
    local_sigma = np.sqrt(np.abs(second_moment - local_mean * local_mean))
    mscn = (gray - local_mean) / (local_sigma + 1.0)
    coefficient_variation = np.divide(
        local_sigma,
        local_mean,
        out=np.zeros_like(local_sigma),
        where=np.abs(local_mean) > np.finfo(np.float64).eps,
    )

    mscn_variance = _sample_nan_variance(_split_blocks(mscn))
    vertical_pair = mscn * np.roll(mscn, shift=1, axis=0)
    negative_pair = vertical_pair.copy()
    negative_pair[negative_pair > 0.0] = np.nan
    positive_pair = vertical_pair.copy()
    positive_pair[positive_pair < 0.0] = np.nan
    negative_variance = _sample_nan_variance(_split_blocks(negative_pair))
    positive_variance = _sample_nan_variance(_split_blocks(positive_pair))

    mean_sigma = np.mean(_split_blocks(local_sigma), axis=(-2, -1))
    mean_variation = np.mean(_split_blocks(coefficient_variation), axis=(-2, -1))
    contrast_gray, contrast_by, contrast_rg = _contrast_energy(image)
    mean_contrast_gray = np.mean(_split_blocks(contrast_gray), axis=(-2, -1))
    mean_contrast_by = np.mean(_split_blocks(contrast_by), axis=(-2, -1))
    mean_contrast_rg = np.mean(_split_blocks(contrast_rg), axis=(-2, -1))

    dark_channel = np.min(image_float / 255.0, axis=-1)
    mean_dark_channel = np.mean(_split_blocks(dark_channel), axis=(-2, -1))
    mean_saturation = np.mean(_split_blocks(_saturation(image)), axis=(-2, -1))

    red_green = red - green
    blue_yellow = 0.5 * (red + green) - blue
    red_green_blocks = _split_blocks(red_green)
    blue_yellow_blocks = _split_blocks(blue_yellow)
    red_green_std = np.std(red_green_blocks, axis=(-2, -1), ddof=1)
    blue_yellow_std = np.std(blue_yellow_blocks, axis=(-2, -1), ddof=1)
    colorfulness = np.sqrt(red_green_std**2 + blue_yellow_std**2)
    colorfulness += 0.3 * np.sqrt(
        np.mean(red_green_blocks, axis=(-2, -1)) ** 2
        + np.mean(blue_yellow_blocks, axis=(-2, -1)) ** 2
    )

    feature_maps = (
        mscn_variance,
        positive_variance,
        negative_variance,
        mean_sigma,
        mean_variation,
        mean_contrast_gray,
        mean_contrast_by,
        mean_contrast_rg,
        _block_entropy(gray.astype(np.uint8)),
        mean_dark_channel,
        mean_saturation,
        colorfulness,
    )
    features = np.column_stack([feature.reshape(-1) for feature in feature_maps])
    return np.log1p(features)


@lru_cache(maxsize=2)
def _model_distance_components(kind: str) -> tuple[np.ndarray, np.ndarray, float]:
    covariance = FADE_FOGFREE_COVARIANCE if kind == "fogfree" else FADE_FOGGY_COVARIANCE
    half_covariance = covariance / 2.0
    inverse = np.linalg.inv(half_covariance)
    ones = np.ones(12, dtype=np.float64)
    inverse_ones = inverse @ ones
    return inverse, inverse_ones, float(ones @ inverse_ones)


def _distance_to_model(features: np.ndarray, mean: np.ndarray, kind: str) -> float:
    difference = mean[None, :] - features
    inverse, inverse_ones, ones_quadratic = _model_distance_components(kind)
    base_distance = np.einsum("ij,jk,ik->i", difference, inverse, difference)
    patch_variance = np.var(features, axis=1, ddof=1)
    alpha = patch_variance / 2.0
    projected = difference @ inverse_ones
    correction = alpha * projected * projected / (1.0 + alpha * ones_quadratic)
    squared_distance = base_distance - correction
    squared_distance[(squared_distance < 0.0) & (squared_distance > -1e-10)] = 0.0
    distances = np.sqrt(squared_distance)
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        raise ValueError("FADE could not compute a finite model distance")
    return float(np.mean(finite))


def compute_fade(image: np.ndarray) -> float:
    """Compute the no-reference FADE fog-density score for RGB in ``[0, 1]``.

    The image is cropped at its bottom and right edges to the 8x8 reference
    patch grid. Lower scores indicate lower perceptual fog density.
    """
    features = _extract_fade_features(image)
    finite_features = features[np.all(np.isfinite(features), axis=1)]
    if finite_features.size == 0:
        raise ValueError("FADE could not extract finite fog-aware features")
    foggy_level = _distance_to_model(finite_features, FADE_FOGFREE_MEAN, "fogfree")
    fogfree_level = _distance_to_model(finite_features, FADE_FOGGY_MEAN, "foggy")
    score = foggy_level / (fogfree_level + 1.0)
    if not np.isfinite(score):
        raise ValueError("FADE produced a non-finite score")
    return float(score)
