"""
@file io_images.py
@brief Image loading / saving for Sea-thru.

Handles 8-bit JPEGs (the GoPro path) now, with a hook for linear RAW later.
Sea-thru is a *physical* model, so images are converted to a roughly linear
[0, 1] float representation before processing. For 8-bit sRGB JPEGs we apply an
inverse-sRGB (gamma) curve; the result is re-encoded to sRGB on save.

@author Michael Venz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _srgb_to_linear(x):
    """@brief Inverse sRGB transfer function (gamma decode).
    @param x Array in [0, 1], sRGB-encoded.
    @return Array in [0, 1], approximately linear radiance."""
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)


def _linear_to_srgb(x):
    """@brief Forward sRGB transfer function (gamma encode).
    @param x Array in [0, 1], approximately linear radiance.
    @return Array in [0, 1], sRGB-encoded."""
    a = 0.055
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * x ** (1 / 2.4) - a)


def load_image(path, max_size=1024, linearize=True):
    """
    @brief Load an image, downscale so the long edge <= ``max_size``, return
    float RGB in [0, 1].

    When ``linearize`` is True the sRGB gamma is removed so the data is
    (approximately) linear radiance, which is what the Sea-thru model assumes.

    @param path Path to the source image (any format Pillow can open).
    @param max_size Maximum long-edge size in pixels; pass ``None``/``0`` to
        keep native resolution (used for the ``--full-res`` output pass).
    @param linearize Apply the inverse-sRGB gamma curve (see _srgb_to_linear).
    @return Tuple ``(img, original_size)``: ``img`` is an ``(H, W, 3)`` float64
        array in [0, 1]; ``original_size`` is the source ``(W, H)`` in pixels.
    """
    img = Image.open(path).convert("RGB")
    original_size = img.size
    if max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float64) / 255.0
    if linearize:
        arr = _srgb_to_linear(arr)
    return arr, original_size


def save_image(path, img, delinearize=True):
    """
    @brief Save a float RGB [0, 1] image, re-applying sRGB gamma if requested.
    @param path Destination file path; format is inferred from the extension.
    @param img (H, W, 3) float array, ideally in [0, 1].
    @param delinearize Re-apply the forward sRGB gamma curve before saving
        (should match the ``linearize`` flag used when the image was loaded).
    """
    arr = _linear_to_srgb(img) if delinearize else np.clip(img, 0, 1)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(str(path))


def save_debug_panel(path, result, img):
    """
    @brief Save a 2x3 montage of the intermediate maps for QA.
    @param path Destination image file path.
    @param result A seathru.core.SeathruResult produced with ``return_debug=True``.
    @param img The working-resolution input image the result was computed from.
    @pre ``result.backscatter`` / ``.illuminant`` / ``.beta_D`` /
        ``.neighborhood_map`` are populated (i.e. ``return_debug=True`` was set).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .core import _scale01

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    panels = [
        ("Input (linear)", img),
        ("Backscatter B", _scale01(result.backscatter)),
        ("Illuminant", _scale01(result.illuminant)),
        ("Neighborhoods", result.neighborhood_map),
        ("beta_D", _scale01(result.beta_D)),
        ("Recovered", result.recovered),
    ]
    for ax, (title, data) in zip(axes.ravel(), panels):
        ax.imshow(data)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(path), dpi=90)
    plt.close(fig)
