"""
@file __init__.py
@brief seathru: physically based removal of water from underwater images.

A from-scratch Python reimplementation of Derya Akkaynak & Tali Treibitz,
"Sea-thru: A Method for Removing Water From Underwater Images" (CVPR 2019):
https://openaccess.thecvf.com/content_CVPR_2019/papers/Akkaynak_Sea-Thru_A_Method_for_Removing_Water_From_Underwater_Images_CVPR_2019_paper.pdf

This package implements the paper's equations and revised underwater image
formation model from scratch in NumPy/SciPy/scikit-image; it is not a copy of
the authors' code. See seathru.core for the algorithm, seathru.depth for
pluggable range-map sources, and seathru.survey / seathru.pipeline for
batch/"survey-locked" processing of large photo datasets.

Quick start
-----------
>>> from seathru import run_seathru, SeathruParams
>>> from seathru.io_images import load_image, save_image
>>> img, _ = load_image("frame.jpg")
>>> depths = ...  # (H, W) range map in metres, e.g. from seathru.depth
>>> result = run_seathru(img, depths)
>>> save_image("frame_seathru.png", result.recovered)

@author Michael Venz
"""
from .core import (
    SeathruParams,
    SeathruResult,
    SurveyStats,
    run_seathru,
    recover_image,
    estimate_backscatter,
    estimate_illumination,
    refine_attenuation,
)

## Package version (kept in sync with pyproject.toml).
__version__ = "0.1.0"
__all__ = [
    "SeathruParams", "SeathruResult", "SurveyStats", "run_seathru",
    "recover_image", "estimate_backscatter", "estimate_illumination",
    "refine_attenuation",
]
