"""Hautière's blind contrast-restoration descriptors e, r̄ and σ.

Hautière, Tarel, Aubert and Dumont, "Blind contrast enhancement assessment
by gradient ratioing at visible edges", Image Analysis & Stereology 27:87-95
(2008), score a restored image against the degraded input it was computed
from, without a clear reference:

- ``e``: rate of new visible edges, ``(n_r - n_o) / n_o`` (Eq. 24).
- ``r̄``: geometric mean of the gradient ratios ``|∇I_r| / |∇I_o|`` at the
  visible edges of the restored image (Eqs. 6 and 25).
- ``σ``: share of pixels that are black or white after restoration but were
  not before (Eq. 26), reported in percent like the paper's tables.

Visible edges are found as in Hautière, Aubert and Jourlin, "Mesure du
contraste local dans les images", Traitement du Signal 23:145-158 (2006),
which the 2008 paper refers to for the implementation: Köhler's threshold
selection is run in odd sliding windows that advance by half a window, on
the pixel couples of the half 4-neighbourhood, with Weber contrast. A window
whose best border has contrast ``2 C(s0) > 5 %`` marks the pixels of that
border ``F(s0)`` as visible edges.

Neither paper fixes every detail, so this implementation chooses:

- 7 × 7 windows (the papers name 7 × 7 or 9 × 9) advancing 3 pixels, plus a
  final window flush with the right and bottom borders so every pixel is
  covered;
- both pixels of a couple in ``F(s0)`` count as visible-edge pixels;
- ``∇`` is the Sobel gradient magnitude, and r̄ uses the restored image's
  visible-edge pixels where both gradients are non-zero;
- images are the evaluator's ``[0, 1]`` RGB, rounded to 8 bits and converted
  to grey with MATLAB's ``rgb2gray`` weights; black and white are grey 0 and
  255.

Values are therefore comparable between evaluations made with this module,
not necessarily with numbers from other implementations.

The computation runs in torch on the evaluator's device. Köhler's curves
are accumulated in exact 64-bit integer (fixed-point) arithmetic, so results
are identical on CPU and CUDA and across runs, whatever order CUDA's atomic
additions take.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from .niqe import _rgb_to_gray_255, _to_uint8_rgb

WINDOW_SIZE = 7
WINDOW_STEP = WINDOW_SIZE // 2
VISIBILITY_THRESHOLD = 0.05

_LEVELS = 256
# Windows whose Köhler curves are computed at once; bounds peak memory to
# roughly 8 arrays of ``chunk x 256`` 64-bit values.
_WINDOW_CHUNK = {"cuda": 1 << 15}
_DEFAULT_WINDOW_CHUNK = 1 << 12
# ``1 / high`` in fixed point. A window's sum of at most 84 such terms stays
# far below 2**63, and integer sums do not depend on the order of addition.
_FIXED_POINT_BITS = 52

Device = Union[str, torch.device]


def _grey_levels(image: np.ndarray, device: torch.device) -> torch.Tensor:
    grey = _rgb_to_gray_255(_to_uint8_rgb(image)).astype(np.int64)
    return torch.from_numpy(grey).to(device)


def _window_starts(length: int, device: torch.device) -> torch.Tensor:
    starts = list(range(0, length - WINDOW_SIZE + 1, WINDOW_STEP))
    if starts[-1] != length - WINDOW_SIZE:
        starts.append(length - WINDOW_SIZE)
    return torch.tensor(starts, dtype=torch.int64, device=device)


def _couple_offsets(width: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Flat offsets, from a window's top-left pixel, of its V*4 couples."""
    rows = torch.arange(WINDOW_SIZE, device=device)
    columns = torch.arange(WINDOW_SIZE - 1, device=device)
    horizontal = (rows[:, None] * width + columns[None, :]).reshape(-1)
    vertical = (columns[:, None] * width + rows[None, :]).reshape(-1)
    first = torch.cat((horizontal, vertical))
    second = torch.cat((horizontal + 1, vertical + width))
    return first, second


def _interval_sums(
    starts: torch.Tensor,
    stops: torch.Tensor,
    weights: torch.Tensor,
    windows: int,
) -> torch.Tensor:
    """Per window and level, the sum of ``weights`` whose interval holds it.

    Interval ``k`` covers levels ``starts[k] <= s < stops[k]``; both are flat
    ``window * 256 + level`` indices. Integer arithmetic keeps it exact.
    """
    delta = torch.zeros(windows * _LEVELS, dtype=torch.int64, device=weights.device)
    delta.index_add_(0, starts, weights)
    delta.index_add_(0, stops, -weights)
    return torch.cumsum(delta.view(windows, _LEVELS), dim=1)


def _best_borders(
    low: torch.Tensor, high: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Köhler's best threshold ``s0`` and its contrast ``2 C(s0)`` per window.

    ``low`` and ``high`` hold, per window and couple, the smaller and larger
    grey level. A couple belongs to the border ``F(s)`` of every threshold
    ``low <= s < high`` and adds the Weber contrast
    ``min(1 - low / s, 1 - s / high)`` to it (0 at ``s = 0``). The first
    term is the smaller one exactly while ``s**2 <= low * high``, so with
    ``q = isqrt(low * high)`` a window's summed contrast is
    ``N(s) - B(s) / s - E(s) * s``: ``N`` counts its couples in ``F(s)``,
    ``B`` sums ``low`` over those with ``low <= s <= q`` and ``E`` sums
    ``1 / high`` over those with ``q < s < high``. Each is a sum of interval
    indicators, built here from difference arrays.
    """
    windows, couples = low.shape
    device = low.device
    # Couples with equal grey levels lie on no border. They are kept, with
    # zero weight, so the arrays keep their shape: no data-dependent indexing.
    spans = (high > low).to(torch.int64)

    product = low * high
    split = torch.sqrt(product.to(torch.float64)).floor().to(torch.int64)
    split -= (split * split > product).to(torch.int64)
    split += ((split + 1) * (split + 1) <= product).to(torch.int64)

    offset = (torch.arange(windows, device=device) * _LEVELS)[:, None]
    starts = (offset + low).reshape(-1)
    middles = (offset + split + 1).clamp(max=offset + _LEVELS - 1).reshape(-1)
    ends = (offset + high).reshape(-1)
    spans = spans.reshape(-1)
    low = low.reshape(-1)
    high = high.reshape(-1)

    inverse_high = torch.round(
        torch.tensor(2.0**_FIXED_POINT_BITS, dtype=torch.float64, device=device)
        / high.clamp(min=1).to(torch.float64)
    ).to(torch.int64)
    counts = _interval_sums(starts, ends, spans, windows)
    lower_low = _interval_sums(starts, middles, low * spans, windows)
    upper_inverse = _interval_sums(middles, ends, inverse_high * spans, windows)

    levels = torch.arange(_LEVELS, dtype=torch.float64, device=device)
    counts = counts.to(torch.float64)
    sums = (
        counts
        - lower_low.to(torch.float64) / levels.clamp(min=1.0)
        - upper_inverse.to(torch.float64) * (levels / 2.0**_FIXED_POINT_BITS)
    )
    sums[:, 0] = 0.0
    mean = torch.where(
        counts > 0.5,
        sums / counts.clamp(min=1.0),
        torch.tensor(float("-inf"), dtype=torch.float64, device=device),
    )
    # Exact ties pick the lowest threshold whatever the rounding of each sum.
    mean = torch.round(mean * 1e12) / 1e12

    best_contrast, best = torch.max(mean, dim=1)
    # torch.max does not promise the first maximum on ties; take it explicitly.
    best = torch.argmax((mean == best_contrast[:, None]).to(torch.uint8), dim=1)
    best_contrast = 2.0 * best_contrast
    best_contrast = torch.where(
        torch.isfinite(best_contrast), best_contrast, torch.zeros_like(best_contrast)
    )
    return best, best_contrast


def _visible_edges(grey: torch.Tensor) -> torch.Tensor:
    height, width = grey.shape
    if height < WINDOW_SIZE or width < WINDOW_SIZE:
        raise ValueError(
            f"Hautière descriptors require an image of at least "
            f"{WINDOW_SIZE}x{WINDOW_SIZE} pixels"
        )
    device = grey.device
    rows = _window_starts(height, device)
    columns = _window_starts(width, device)
    origins = (rows[:, None] * width + columns[None, :]).reshape(-1)
    first_offsets, second_offsets = _couple_offsets(width, device)
    flat = grey.reshape(-1)
    chunk_size = _WINDOW_CHUNK.get(device.type, _DEFAULT_WINDOW_CHUNK)

    marks = torch.zeros(flat.numel(), dtype=torch.int32, device=device)
    for chunk in torch.split(origins, chunk_size):
        first_index = (chunk[:, None] + first_offsets[None, :]).reshape(-1)
        second_index = (chunk[:, None] + second_offsets[None, :]).reshape(-1)
        first = flat[first_index].view(chunk.numel(), -1)
        second = flat[second_index].view(chunk.numel(), -1)
        low = torch.minimum(first, second)
        high = torch.maximum(first, second)
        best, contrast = _best_borders(low, high)
        threshold = best[:, None]
        on_border = (
            (contrast > VISIBILITY_THRESHOLD)[:, None]
            & (low <= threshold)
            & (threshold < high)
        ).reshape(-1).to(torch.int32)
        marks.index_add_(0, first_index, on_border)
        marks.index_add_(0, second_index, on_border)
    return (marks > 0).view(height, width)


def visible_edges(grey: np.ndarray, device: Device = "cpu") -> np.ndarray:
    """Return the boolean map of visible edges of an 8-bit grey image."""
    levels = torch.as_tensor(np.asarray(grey), dtype=torch.int64, device=device)
    return _visible_edges(levels).cpu().numpy()


@torch.no_grad()
def _sobel_magnitude(grey: torch.Tensor) -> torch.Tensor:
    # Replicating the edge row matches scipy.ndimage.sobel's default
    # "reflect" mode for its one-pixel support.
    image = grey.to(torch.float64)[None, None]
    image = F.pad(image, (1, 1, 1, 1), mode="replicate")
    derivative = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=torch.float64,
        device=grey.device,
    )
    kernels = torch.stack((derivative, derivative.T))[:, None]
    gradients = F.conv2d(image, kernels)[0]
    return torch.hypot(gradients[0], gradients[1])


class HautiereScorer:
    """Score restored images against their degraded inputs on one device."""

    def __init__(self, device: Device = "cpu") -> None:
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(self, restored: np.ndarray, degraded: np.ndarray) -> dict[str, float]:
        """Compute Hautière's e, r̄ and σ.

        Both images are ``(H, W, 3)`` RGB in ``[0, 1]`` on the same pixel grid.

        Returns:
            ``hautiere_e`` (higher is better; ``nan`` when the input has no
            visible edge), ``hautiere_r`` (higher is better; ``nan`` without a
            scorable visible edge in the restored image) and
            ``hautiere_sigma``, the percentage of newly saturated pixels
            (lower is better).
        """
        restored_grey = _grey_levels(restored, self.device)
        degraded_grey = _grey_levels(degraded, self.device)
        if restored_grey.shape != degraded_grey.shape:
            raise ValueError(
                "Hautière descriptors need the restored image and its input on "
                f"the same grid, got {tuple(restored_grey.shape)} and "
                f"{tuple(degraded_grey.shape)}"
            )

        restored_edges = _visible_edges(restored_grey)
        degraded_edges = _visible_edges(degraded_grey)
        n_restored = int(restored_edges.sum())
        n_degraded = int(degraded_edges.sum())
        rate = (n_restored - n_degraded) / n_degraded if n_degraded else float("nan")

        restored_gradient = _sobel_magnitude(restored_grey)[restored_edges]
        degraded_gradient = _sobel_magnitude(degraded_grey)[restored_edges]
        scorable = (restored_gradient > 0) & (degraded_gradient > 0)
        if bool(scorable.any()):
            ratio = float(
                torch.exp(
                    torch.log(
                        restored_gradient[scorable] / degraded_gradient[scorable]
                    ).mean()
                )
            )
        else:
            ratio = float("nan")

        restored_saturated = (restored_grey == 0) | (restored_grey == 255)
        degraded_saturated = (degraded_grey == 0) | (degraded_grey == 255)
        newly_saturated = int((restored_saturated & ~degraded_saturated).sum())
        saturation = 100.0 * newly_saturated / restored_grey.numel()

        return {
            "hautiere_e": float(rate),
            "hautiere_r": ratio,
            "hautiere_sigma": float(saturation),
        }


def load_hautiere(device: Device = "cpu") -> HautiereScorer:
    """Return a scorer computing Hautière's e, r̄ and σ on ``device``."""
    return HautiereScorer(device)


def compute_hautiere(
    restored: np.ndarray, degraded: np.ndarray, device: Device = "cpu"
) -> dict[str, float]:
    """Compute Hautière's e, r̄ and σ for a restored image and its input."""
    return HautiereScorer(device)(restored, degraded)
