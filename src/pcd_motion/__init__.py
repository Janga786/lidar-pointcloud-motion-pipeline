"""
pcd_motion
==========

A small, dependency-light Python pipeline for analysing motion in
LiDAR / depth-camera point-cloud sequences.

The pipeline stages — each implemented as one module here — are:

- :mod:`pcd_motion.preprocess`   — load a ``.pcd`` file, drop NaNs / Infs.
- :mod:`pcd_motion.dimensions`   — axis-aligned bounding-box extents.
- :mod:`pcd_motion.orientation`  — principal-axis direction via PCA.
- :mod:`pcd_motion.motion`       — frame-pair displacement test.
- :mod:`pcd_motion.speed`        — ICP-based rotation rate + axis.

Public API
----------

>>> from pcd_motion import (
...     load_and_preprocess_pcd, calculate_dimensions,
...     determine_orientation, is_moving, calculate_speed_rate,
... )
"""

from __future__ import annotations

from .dimensions import calculate_dimensions
from .motion import is_moving
from .orientation import determine_orientation
from .preprocess import load_and_preprocess_pcd
from .speed import (
    calculate_speed_rate,
    compute_rotation_angle,
    compute_rotation_axis_angle,
)

__all__ = [
    "load_and_preprocess_pcd",
    "calculate_dimensions",
    "determine_orientation",
    "is_moving",
    "calculate_speed_rate",
    "compute_rotation_angle",
    "compute_rotation_axis_angle",
]

__version__ = "1.0.0"
__author__ = "Bliss Janga"
__license__ = "MIT"
