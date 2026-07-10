"""
@file monocular.py
@brief Monocular neural depth (opt-in; requires PyTorch).

Estimates a relative depth map per image and scales it to metres. Because
monocular depth is only known up to an unknown affine transform, we anchor it
using whatever metric information is available, in order of preference:

  1. ``meta.depth_m``  - the ASV altitude, used so the *mean* range equals the
     altitude (best when the CSV reports a valid depth).
  2. ``fixed_range``   - a user-supplied ``(near_m, far_m)`` pair.
  3. otherwise the raw normalised [near, far] fallback of ``(2, 10)`` metres.

Two backends are supported via ``torch.hub`` (downloaded on first use):
  * ``"depth_anything_v2"`` (default, best quality) - needs internet + the
    Depth-Anything-V2 repo; produces relative *depth* (larger = farther) after
    inversion of its disparity output.
  * ``"midas"`` (Intel MiDaS ``DPT_Large``) - widely available fallback.

This module is imported lazily by the pipeline so the rest of the library works
without torch installed.
"""
from __future__ import annotations

import numpy as np

from .base import DepthSource, ImageMeta


class MonocularDepthSource(DepthSource):
    """@brief Neural monocular relative-depth estimate, scaled to metres; see
    module docstring for the scaling/anchoring strategy and backend options."""

    def __init__(self, backend="midas", model_type="DPT_Large",
                 fixed_range=None, device=None):
        """
        @param backend ``"midas"`` or ``"depth_anything_v2"``.
        @param model_type torch.hub model id for the MiDaS backend (e.g. ``"DPT_Large"``).
        @param fixed_range Optional ``(near_m, far_m)`` fallback used when
            ``meta.depth_m`` is unavailable.
        @param device Torch device string; defaults to CUDA if available, else CPU.
        @throws ImportError if PyTorch is not installed.
        """
        try:
            import torch  # noqa: F401
        except ImportError as err:  # pragma: no cover - env dependent
            raise ImportError(
                "MonocularDepthSource needs PyTorch. Install torch + torchvision "
                "in an env with Python <= 3.12, or use FileDepthSource / "
                "PlaneDepthSource instead."
            ) from err
        import torch

        self.backend = backend
        self.model_type = model_type
        self.fixed_range = fixed_range
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._transform = None

    # -- model loading ----------------------------------------------------- #
    def _ensure_model(self):
        """@brief Lazily download/load the selected backend's torch.hub model.
        @throws ValueError if ``self.backend`` is not recognised."""
        if self._model is not None:
            return
        import torch
        if self.backend == "midas":
            self._model = torch.hub.load("intel-isl/MiDaS", self.model_type)
            self._model.to(self.device).eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self._transform = (transforms.dpt_transform
                               if "DPT" in self.model_type
                               else transforms.default_transform)
        elif self.backend == "depth_anything_v2":
            # Uses the community torch.hub entrypoint; falls back to MiDaS-style
            # usage. Relative disparity is returned and inverted below.
            self._model = torch.hub.load("LiheYoung/Depth-Anything",
                                         "DepthAnything", pretrained=True)
            self._model.to(self.device).eval()
            self._transform = None
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # -- inference --------------------------------------------------------- #
    def _infer_relative(self, img):
        """@brief Run the backend model and invert its disparity output.
        @param img (H, W, 3) float image in [0, 1].
        @return (H, W) float relative *depth* map (larger = farther), unit-less."""
        import torch
        self._ensure_model()
        rgb8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        with torch.no_grad():
            if self.backend == "midas":
                batch = self._transform(rgb8).to(self.device)
                pred = self._model(batch)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=img.shape[:2],
                    mode="bicubic", align_corners=False).squeeze().cpu().numpy()
                # MiDaS returns inverse depth (disparity): invert to depth.
                disp = pred - pred.min() + 1e-6
                rel = 1.0 / disp
            else:
                import torch.nn.functional as F
                t = torch.from_numpy(rgb8).float().permute(2, 0, 1)[None] / 255.0
                t = t.to(self.device)
                disp = self._model(t)
                disp = F.interpolate(disp[:, None], size=img.shape[:2],
                                     mode="bicubic", align_corners=False)
                disp = disp.squeeze().cpu().numpy()
                rel = 1.0 / (disp - disp.min() + 1e-6)
        return rel

    def _scale_to_metres(self, rel, meta: ImageMeta):
        """@brief Affine-scale a relative depth map into metres (see module docstring).
        @param rel (H, W) float relative depth from _infer_relative.
        @param meta ImageMeta; used to anchor scale to a known altitude if present.
        @return (H, W) float range map in metres."""
        rel = rel - rel.min()
        span = rel.max() + 1e-8
        if meta and meta.depth_m and meta.depth_m > 0:
            # Anchor so the mean range equals the reported altitude.
            rel_mean = rel.mean() / span
            return (rel / span) * (meta.depth_m / max(rel_mean, 1e-3))
        near, far = self.fixed_range or (2.0, 10.0)
        return near + (rel / span) * (far - near)

    def get_depth(self, img, meta: ImageMeta):
        """@brief @copydoc DepthSource.get_depth"""
        rel = self._infer_relative(img)
        return self._scale_to_metres(rel, meta).astype(np.float32)
