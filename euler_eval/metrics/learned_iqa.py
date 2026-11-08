"""Learned no-reference image quality metrics: MUSIQ and CLIP-IQA.

Both metrics score a single RGB image with a pretrained network and are
computed with ``pyiqa`` (IQA-PyTorch), which holds the reference
implementations and downloads their weights on first use:

- MUSIQ (Ke et al., ICCV 2021): multi-scale image quality transformer, pyiqa
  metric ``musiq`` with KonIQ-10k weights. Scores are roughly 0-100.
- CLIP-IQA (Wang et al., AAAI 2023): CLIP ResNet-50 similarity to antonym
  quality prompts, pyiqa metric ``clipiqa``. Scores lie in ``[0, 1]``.

Higher is better for both. ``pyiqa`` is an optional dependency, installed with
``pip install "euler-eval[dehazing]"``; it is licensed for non-commercial use
only (PolyForm Noncommercial 1.0.0).
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

from .niqe import _validate_unit_rgb

PYIQA_INSTALL_HINT = 'pip install "euler-eval[dehazing]"'


class PyIQAScorer:
    """Score RGB images with one pyiqa no-reference metric.

    The network is built once, on ``device``, when the scorer is created;
    calling the scorer then scores one ``(H, W, 3)`` image with values in
    ``[0, 1]`` at its native resolution.
    """

    def __init__(
        self, metric_name: str, device: Union[str, torch.device] = "cpu"
    ) -> None:
        try:
            import pyiqa
        except ImportError as exc:
            raise ImportError(
                f"The {metric_name!r} metric requires the 'pyiqa' package. "
                f"Install it with `{PYIQA_INSTALL_HINT}`."
            ) from exc

        self.metric_name = metric_name
        self.device = torch.device(device)
        self._metric = pyiqa.create_metric(metric_name, device=self.device)
        self.lower_better = bool(self._metric.lower_better)

    def __call__(self, image: np.ndarray) -> float:
        array = _validate_unit_rgb(image, subject="pyiqa metrics")
        tensor = torch.from_numpy(
            np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.float32)
        ).unsqueeze(0)
        with torch.inference_mode():
            score = self._metric(tensor.to(self.device))
        return float(score.item())


def load_musiq(device: Union[str, torch.device] = "cpu") -> PyIQAScorer:
    """Load MUSIQ (KonIQ-10k weights) on ``device``; higher is better."""
    return PyIQAScorer("musiq", device)


def load_clipiqa(device: Union[str, torch.device] = "cpu") -> PyIQAScorer:
    """Load CLIP-IQA (ResNet-50 backbone) on ``device``; higher is better."""
    return PyIQAScorer("clipiqa", device)
