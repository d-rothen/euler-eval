"""Tests for Hautière's blind contrast-restoration descriptors e, r̄ and σ."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from euler_eval.metrics import (
    HautiereScorer,
    compute_hautiere,
    hautiere as hautiere_mod,
    load_hautiere,
)
from euler_eval.metrics.hautiere import (
    VISIBILITY_THRESHOLD,
    WINDOW_SIZE,
    WINDOW_STEP,
    visible_edges,
)


def _reference_visible_edges(grey: np.ndarray) -> np.ndarray:
    """Literal transcription of Hautière et al. (2008) Eqs. 20-23.

    Köhler's method in WINDOW_SIZE windows stepping WINDOW_STEP (Hautière,
    Aubert and Jourlin 2006, section 3.2), V*4 couples, Weber contrast
    (their Eq. 21), visible when 2 C(s0) > 5 %.
    """
    height, width = grey.shape

    def starts(length):
        values = list(range(0, length - WINDOW_SIZE + 1, WINDOW_STEP))
        if values[-1] != length - WINDOW_SIZE:
            values.append(length - WINDOW_SIZE)
        return values

    def weber(threshold, level):
        largest = max(threshold, level)
        return abs(threshold - level) / largest if largest else 0.0

    edges = np.zeros(grey.shape, dtype=bool)
    for top in starts(height):
        for left in starts(width):
            couples = []
            for row in range(top, top + WINDOW_SIZE):
                for column in range(left, left + WINDOW_SIZE):
                    if column + 1 < left + WINDOW_SIZE:
                        couples.append(((row, column), (row, column + 1)))
                    if row + 1 < top + WINDOW_SIZE:
                        couples.append(((row, column), (row + 1, column)))
            best_threshold, best_mean = None, -1.0
            for threshold in range(256):
                contrasts = [
                    min(weber(threshold, int(grey[x])), weber(threshold, int(grey[y])))
                    for x, y in couples
                    if min(grey[x], grey[y]) <= threshold < max(grey[x], grey[y])
                ]
                if contrasts:
                    mean = sum(contrasts) / len(contrasts)
                    if mean > best_mean:
                        best_threshold, best_mean = threshold, mean
            if best_threshold is not None and 2 * best_mean > VISIBILITY_THRESHOLD:
                for x, y in couples:
                    if min(grey[x], grey[y]) <= best_threshold < max(grey[x], grey[y]):
                        edges[x] = edges[y] = True
    return edges


def _pattern_rgb(height: int, width: int) -> np.ndarray:
    rows, columns = np.indices((height, width))
    image = np.stack(
        [
            (rows * 3 + columns * 5) % 256,
            (rows * 7 + columns * 11 + 23) % 256,
            (rows * 13 + columns * 17 + 101) % 256,
        ],
        axis=-1,
    )
    return image.astype(np.float64) / 255.0


def _grey_rgb(levels: np.ndarray) -> np.ndarray:
    return np.repeat((np.asarray(levels, dtype=np.float64) / 255.0)[..., None], 3, -1)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_visible_edges_match_the_papers_equations(seed):
    rng = np.random.default_rng(seed)
    height, width = rng.integers(WINDOW_SIZE, 15, size=2)
    if seed == 0:
        grey = rng.integers(0, 256, size=(height, width))
    else:
        # Smooth ramps with a step: realistic, mostly small couple spans.
        grey = np.cumsum(rng.integers(-3, 4, size=(height, width)), axis=1) + 120
        grey[:, width // 2 :] += 20 * seed
        grey = np.clip(grey, 0, 255)

    np.testing.assert_array_equal(visible_edges(grey), _reference_visible_edges(grey))


def test_a_visible_step_marks_the_pixels_on_both_sides():
    grey = np.full((WINDOW_SIZE, WINDOW_SIZE), 100)
    grey[:, 4:] = 150  # Weber contrast 2 * C(122) = 0.36 > 5 %

    edges = visible_edges(grey)

    expected = np.zeros_like(edges)
    expected[:, 3:5] = True
    np.testing.assert_array_equal(edges, expected)


def test_a_step_below_five_percent_contrast_is_not_visible():
    grey = np.full((WINDOW_SIZE, WINDOW_SIZE), 100)
    grey[:, 4:] = 102  # 2 * C(101) = 2 / 102 < 5 %

    assert not visible_edges(grey).any()


def test_identity_restoration_scores_no_change():
    hazy = _pattern_rgb(48, 64) * 0.4 + 0.5

    scores = compute_hautiere(hazy, hazy)

    assert scores == {"hautiere_e": 0.0, "hautiere_r": 1.0, "hautiere_sigma": 0.0}


def test_restoring_contrast_adds_visible_edges_and_gradient():
    clear = _pattern_rgb(48, 64)
    hazy = clear * 0.4 + 0.5

    scores = compute_hautiere(clear, hazy)

    assert scores["hautiere_e"] > 0.0
    assert scores["hautiere_r"] > 1.0
    assert scores["hautiere_sigma"] == 0.0


def test_sigma_counts_only_newly_black_or_white_pixels_in_percent():
    degraded = np.full((10, 10), 128)
    degraded[0, :5] = 255  # already saturated: not counted again
    restored = degraded.copy()
    restored[0, :] = 255  # 5 newly white pixels
    restored[1, :3] = 0  # 3 newly black pixels

    scores = compute_hautiere(_grey_rgb(restored), _grey_rgb(degraded))

    assert scores["hautiere_sigma"] == pytest.approx(8.0)


def test_undefined_rates_are_nan():
    flat = _grey_rgb(np.full((16, 16), 128))
    textured = _pattern_rgb(16, 16)

    no_input_edges = compute_hautiere(textured, flat)
    no_restored_edges = compute_hautiere(flat, flat)

    assert np.isnan(no_input_edges["hautiere_e"])
    assert np.isnan(no_restored_edges["hautiere_e"])
    assert np.isnan(no_restored_edges["hautiere_r"])


def test_images_must_share_a_grid_and_cover_a_window():
    with pytest.raises(ValueError, match="same grid"):
        compute_hautiere(_pattern_rgb(16, 16), _pattern_rgb(16, 17))
    with pytest.raises(ValueError, match=f"at least {WINDOW_SIZE}x{WINDOW_SIZE}"):
        compute_hautiere(_pattern_rgb(6, 16), _pattern_rgb(6, 16))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        compute_hautiere(_pattern_rgb(16, 16) * 2.0, _pattern_rgb(16, 16))


def test_small_window_chunks_do_not_change_the_result(monkeypatch):
    clear = _pattern_rgb(40, 52)
    hazy = clear * 0.4 + 0.5
    expected = compute_hautiere(clear, hazy)

    monkeypatch.setattr(hautiere_mod, "_DEFAULT_WINDOW_CHUNK", 7)

    assert compute_hautiere(clear, hazy) == expected


def test_scorer_is_bound_to_its_device():
    scorer = load_hautiere("cpu")

    assert isinstance(scorer, HautiereScorer)
    assert scorer.device == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_and_cpu_results_are_identical():
    rng = np.random.default_rng(0)
    restored = np.clip(_pattern_rgb(96, 128) + rng.normal(0, 0.05, (96, 128, 3)), 0, 1)
    degraded = restored * 0.4 + 0.5

    assert compute_hautiere(restored, degraded, device="cuda") == compute_hautiere(
        restored, degraded, device="cpu"
    )
