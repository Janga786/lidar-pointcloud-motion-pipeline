"""
Frame-pair motion detector.

Compares consecutive point clouds in a sequence to flag whether the
imaged object has moved enough between frames to warrant running the
full ICP-based speed estimator.  Uses a cheap per-point distance
average; if the mean displacement across all paired points exceeds
a threshold (default 1 cm), the scene is "moving".

This is intentionally a coarse boolean: we want to *gate* the
expensive ICP computation in :mod:`pcd_motion.speed`, not estimate
absolute motion.  False positives are tolerable because they only
cost an extra ICP evaluation; false negatives are bad because they
miss movement.

Limitation: the simple version below assumes both clouds have the
same number of points and that the *k*-th point in cloud A
corresponds to the *k*-th point in cloud B.  That holds when the
LiDAR runs in a fixed-pattern sweep mode (every captured frame
contains the same beam-angles in the same order); for unstructured
captures, run a one-step ICP first to produce a correspondence.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import open3d as o3d


def is_moving(
    pcd_sequence: Sequence[o3d.geometry.PointCloud],
    threshold: float = 0.01,
) -> bool:
    """
    Return ``True`` if any consecutive frame pair shows mean
    displacement above ``threshold`` (metres).

    Parameters
    ----------
    pcd_sequence
        At least two point clouds in temporal order.
    threshold
        Per-point displacement (metres) above which a frame pair is
        considered to be moving.  Default 0.01 m = 1 cm.

    Returns
    -------
    moving
        ``True`` if movement was detected at any point in the
        sequence; ``False`` if all frames are within threshold of
        each other.
    """
    if len(pcd_sequence) < 2:
        return False

    for prev, curr in zip(pcd_sequence, pcd_sequence[1:], strict=False):
        p1 = np.asarray(prev.points)
        p2 = np.asarray(curr.points)

        if p1.shape != p2.shape:
            # Point counts differ ⇒ scene topology changed ⇒ moving.
            return True

        per_point_displacement = np.linalg.norm(p1 - p2, axis=1)
        if float(np.mean(per_point_displacement)) > threshold:
            return True

    return False
