"""Batched GPU PSNR/SSIM for RGB and depth pairs.

The per-sample ``compute_rgb_psnr`` / ``compute_rgb_ssim`` (and their
depth counterparts) are scipy/numpy implementations that run on the CPU
and dispatch a fresh gaussian filter per sample. For large evaluations
on a GPU-equipped machine this leaves the accelerator idle. This module
groups compatible-shape pairs into batches and runs the computation in
a single ``torchmetrics.functional`` call per batch, keeping the per-
sample return contract unchanged (one PSNR + one SSIM value per input).

The batcher gracefully falls back to the CPU ``compute_*`` functions if
torchmetrics is unavailable or if the target device rejects the tensor
(e.g. unsupported dtype). Metadata fields (``max_val_used``,
``depth_range`` etc.) are computed upfront on the caller side because
they summarize the inputs, not the metric value.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

try:
    from torchmetrics.functional import peak_signal_noise_ratio as _tm_psnr
    from torchmetrics.functional.image import (
        structural_similarity_index_measure as _tm_ssim,
    )

    _TORCHMETRICS_AVAILABLE = True
except ImportError:
    _TORCHMETRICS_AVAILABLE = False


_SSIM_WINDOW = 11
_SSIM_SIGMA = _SSIM_WINDOW / 6.0


def _rgb_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert (H, W, 3) float image to (3, H, W) float32 tensor."""
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float()


def _depth_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert (H, W) float depth to (1, H, W) float32 tensor."""
    return torch.from_numpy(np.ascontiguousarray(arr))[None].float()


class GPUImageMetricsBatcher:
    """Accumulates RGB or depth pairs and batches PSNR/SSIM on GPU.

    Callers enqueue ``(pred, gt)`` pairs together with a callback that
    receives ``(psnr_value, ssim_value)`` once the batch flushes. A flush
    fires automatically when the pending buffer reaches ``batch_size``;
    :meth:`finalize` drains any remainder at end of loop.

    For depth, ``data_range`` must be provided per pair because each
    sample's normalization is input-dependent (uses GT max). For RGB we
    use a fixed ``data_range=1.0``.
    """

    def __init__(
        self,
        device: str,
        batch_size: int = 16,
        modality: str = "rgb",
    ):
        if modality not in {"rgb", "depth"}:
            raise ValueError(f"Unsupported modality: {modality}")
        self.device = torch.device(device)
        self.batch_size = max(1, int(batch_size))
        self.modality = modality
        self._pending: list = []

    @staticmethod
    def is_available(device: str) -> bool:
        """Return True when GPU batching is usable on ``device``."""
        if not _TORCHMETRICS_AVAILABLE:
            return False
        dev = torch.device(device)
        if dev.type == "cuda":
            return torch.cuda.is_available()
        if dev.type == "mps":
            return torch.backends.mps.is_available()
        return False

    def enqueue(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        callback: Callable[[float, float], None],
        *,
        data_range: Optional[float] = None,
    ) -> None:
        self._pending.append((pred, gt, callback, data_range))
        if len(self._pending) >= self.batch_size:
            self._flush()

    def finalize(self) -> None:
        self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return

        # Group by spatial shape (H, W). Channel layout is fixed per modality.
        shape_groups: dict = {}
        for idx, (pred, gt, cb, dr) in enumerate(self._pending):
            key = (pred.shape[0], pred.shape[1])
            shape_groups.setdefault(key, []).append((idx, pred, gt, cb, dr))

        for shape, items in shape_groups.items():
            try:
                self._compute_group(items)
            except Exception:
                # Fall back to per-sample CPU compute using the existing
                # ``compute_*`` implementations. Importing here avoids a
                # circular dep and ensures tests that monkeypatch the
                # compute functions continue to work.
                from . import psnr as _psnr_mod
                from . import rgb_psnr_ssim as _rgb_mod
                from . import ssim as _ssim_mod

                for _, pred, gt, cb, dr in items:
                    if self.modality == "rgb":
                        p = float(_rgb_mod.compute_rgb_psnr(pred, gt))
                        s = float(_rgb_mod.compute_rgb_ssim(pred, gt))
                    else:
                        p = float(_psnr_mod.compute_psnr(pred, gt))
                        s = float(_ssim_mod.compute_ssim(pred, gt))
                    cb(p, s)

        self._pending.clear()

    def _compute_group(self, items: list) -> None:
        """Batch-compute PSNR+SSIM for items of identical spatial shape."""
        to_tensor = _rgb_to_tensor if self.modality == "rgb" else _depth_to_tensor
        # Host→device transfer is blocking: on MPS, successive non_blocking
        # copies can alias and the second overwrites the first.
        preds_t = torch.stack([to_tensor(p) for _, p, _, _, _ in items]).to(
            self.device
        )
        gts_t = torch.stack([to_tensor(g) for _, _, g, _, _ in items]).to(
            self.device
        )

        data_ranges = [dr for _, _, _, _, dr in items]
        if self.modality == "rgb":
            psnr_vals = self._compute_psnr(preds_t, gts_t, data_range=1.0)
            ssim_vals = self._compute_ssim(preds_t, gts_t, data_range=1.0)
        else:
            # Depth uses per-sample data_range. Compute each sample's PSNR/
            # SSIM one at a time but all on-GPU in a tight loop to avoid CPU
            # round-trip between ops.
            psnr_vals = []
            ssim_vals = []
            for i, dr in enumerate(data_ranges):
                dr_eff = float(dr) if dr is not None and dr > 0 else 1.0
                psnr_vals.append(
                    self._compute_psnr(
                        preds_t[i : i + 1], gts_t[i : i + 1], data_range=dr_eff
                    )[0]
                )
                ssim_vals.append(
                    self._compute_ssim(
                        preds_t[i : i + 1], gts_t[i : i + 1], data_range=dr_eff
                    )[0]
                )

        for (_, _, _, cb, _), p, s in zip(items, psnr_vals, ssim_vals):
            cb(float(p), float(s))

    @staticmethod
    def _compute_psnr(
        preds: torch.Tensor, targets: torch.Tensor, data_range: float
    ) -> list:
        """Per-sample PSNR over a (B, C, H, W) batch."""
        vals = _tm_psnr(
            preds,
            targets,
            data_range=data_range,
            reduction="none",
            dim=(1, 2, 3),
        )
        return vals.detach().cpu().tolist()

    @staticmethod
    def _compute_ssim(
        preds: torch.Tensor, targets: torch.Tensor, data_range: float
    ) -> list:
        """Per-sample SSIM over a (B, C, H, W) batch."""
        vals = _tm_ssim(
            preds,
            targets,
            gaussian_kernel=True,
            sigma=_SSIM_SIGMA,
            kernel_size=_SSIM_WINDOW,
            reduction="none",
            data_range=data_range,
        )
        if vals.ndim > 1:
            vals = vals.mean(dim=tuple(range(1, vals.ndim)))
        return vals.detach().cpu().tolist()
