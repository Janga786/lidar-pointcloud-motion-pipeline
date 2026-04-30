"""
Axis-aligned bounding-box dimensions.

The width / height / depth of an object inside a point cloud is the
extent of its axis-aligned bounding box (AABB) — the smallest box,
with edges parallel to the coordinate axes, that fully contains every
point.

This is the simplest possible "size" estimator and is robust to
typical LiDAR sparsity (a single returning beam still contributes a
sample to each axis), but it is sensitive to the orientation of the
object: a rotated rectangle will read larger than its physical
dimensions because the AABB has to grow to enclose the rotated
extents.  For orientation-invariant sizing, follow with the
:mod:`pcd_motion.orientation` PCA-based method to align the cloud
to its principal axes first.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from numpy.typing import ArrayLike, NDArray


def calculate_dimensions(
    pcd: o3d.geometry.PointCloud,
    scaling_factors: ArrayLike | None = None,
) -> NDArray[np.float64] | None:
    """
    Return the (width, height, depth) extent of the point cloud's AABB.

    Parameters
    ----------
    pcd
        A non-empty Open3D point cloud.
    scaling_factors
        Optional per-axis multipliers, applied element-wise to the
        AABB extent.  Useful when the LiDAR's metric scaling is known
        to need correction (e.g. a vendor-specific unit-of-measurement
        offset).

    Returns
    -------
    extents
        Length-3 ``ndarray`` of axis-aligned dimensions in metres
        (or whatever unit the source cloud uses), or ``None`` if the
        input is empty.
    """
    if not pcd.has_points():
        print("warning: empty point cloud — no dimensions to report.")
        return None

    aabb = pcd.get_axis_aligned_bounding_box()
    extents = aabb.get_extent()
    if scaling_factors is not None:
        extents = extents * np.asarray(scaling_factors, dtype=float)
    return extents
