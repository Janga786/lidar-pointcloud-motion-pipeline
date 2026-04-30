"""
ICP-based rotation-rate estimation.

Given a temporal sequence of LiDAR captures of a rotating object,
this module estimates:

- The instantaneous **rotation axis** between each consecutive
  frame pair, recovered from the SO(3) transform that aligns the
  pair via Iterative Closest Point (ICP).
- The instantaneous **rotation angle**, from the same transform.
- The **average rotation rate** (degrees / second) over the whole
  sequence, given timestamps for each frame.
- The **average rotation axis** as the unit-normalised mean of all
  per-frame axes (this is a reasonable estimate when the rotation
  axis is approximately stationary; for tumbling motion you'd want a
  Karcher mean on SO(3) instead).

The maths
---------

For a rotation matrix :math:`R \\in SO(3)`,

.. math::

    \\theta = \\arccos\\!\\left( \\tfrac{\\operatorname{tr} R - 1}{2} \\right)

is the rotation angle, and the corresponding axis is

.. math::

    \\hat n = \\frac{1}{2 \\sin \\theta}
              \\bigl( R_{32} - R_{23},\\;
                     R_{13} - R_{31},\\;
                     R_{21} - R_{12} \\bigr).

Both formulas are degenerate at :math:`\\theta = 0` (where the axis
is undefined) and :math:`\\theta = \\pi` (where the off-diagonal
formula loses one bit of sign information).  We guard with a small
tolerance.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import open3d as o3d
from numpy.typing import ArrayLike, NDArray


_AXIS_DEGENERATE_TOL = 1e-6


def compute_rotation_axis_angle(
    transformation: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    """
    Decompose a 4×4 homogeneous transform into a (degrees, axis) pair.

    The rotation block is read from the 3×3 upper-left of
    ``transformation`` and projected to the angle/axis representation
    using the standard trace + skew-symmetric formulas.
    """
    R = transformation[:3, :3]
    cos_arg = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_arg))

    sin_a = float(np.sin(angle))
    if abs(sin_a) < _AXIS_DEGENERATE_TOL:
        return float(np.degrees(angle)), np.zeros(3, dtype=float)

    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]
    ) / (2.0 * sin_a)
    axis /= np.linalg.norm(axis)
    return float(np.degrees(angle)), axis


def compute_rotation_angle(
    pcd_a: o3d.geometry.PointCloud,
    pcd_b: o3d.geometry.PointCloud,
    threshold: float = 0.02,
) -> tuple[float, NDArray[np.float64]]:
    """
    Run point-to-point ICP between ``pcd_a`` and ``pcd_b`` and return
    the (degrees, axis) of the recovered rotation.

    Parameters
    ----------
    pcd_a, pcd_b
        Source and target point clouds.
    threshold
        ICP distance threshold (metres).  Lower for higher-resolution
        captures; default 2 cm matches typical indoor LiDAR.
    """
    init = np.identity(4)
    reg = o3d.pipelines.registration.registration_icp(
        pcd_a, pcd_b, threshold, init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    return compute_rotation_axis_angle(reg.transformation)


def calculate_speed_rate(
    pcd_sequence: Sequence[o3d.geometry.PointCloud],
    time_intervals: ArrayLike,
) -> tuple[float, NDArray[np.float64] | None]:
    """
    Estimate average rotation rate (deg/s) and axis over a sequence.

    Parameters
    ----------
    pcd_sequence
        N ≥ 2 point clouds in temporal order.
    time_intervals
        Length-N array of monotonically-increasing timestamps,
        in seconds.  The total elapsed time is the difference
        between the last and the first.

    Returns
    -------
    avg_speed_dps
        Average rotation rate in **degrees per second**, computed as
        the total accumulated angular displacement divided by the
        total elapsed time.
    avg_axis
        Mean rotation axis over all consecutive frame pairs, or
        ``None`` if the sequence has fewer than two frames.
    """
    if len(pcd_sequence) < 2:
        return 0.0, None

    total_angle = 0.0
    axes: list[NDArray[np.float64]] = []

    for prev, curr in zip(pcd_sequence, pcd_sequence[1:], strict=False):
        angle, axis = compute_rotation_angle(prev, curr)
        total_angle += angle
        if np.linalg.norm(axis) > 0:
            axes.append(axis)

    times = np.asarray(time_intervals, dtype=float)
    elapsed = float(times[-1] - times[0])
    if elapsed <= 0.0:
        return 0.0, None

    avg_speed_dps = total_angle / elapsed
    if not axes:
        return avg_speed_dps, None

    mean_axis = np.mean(np.stack(axes, axis=0), axis=0)
    return avg_speed_dps, mean_axis / np.linalg.norm(mean_axis)
