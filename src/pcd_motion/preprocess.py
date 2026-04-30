"""
Point-cloud loader with invalid-point filtering.

LiDAR captures regularly contain spurious returns at infinity (no echo
within the sensor's range) and NaN values where the time-of-flight
electronics couldn't lock onto a phase.  Both manifest in Open3D
point clouds as numerical garbage that wrecks downstream algorithms
— PCA gets dominated by a single Inf, ICP fails its Cholesky
decomposition, bounding boxes blow up to infinite extent.

This loader strips those points before any downstream stage ever
sees them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


def load_and_preprocess_pcd(file_path: str | Path) -> o3d.geometry.PointCloud | None:
    """
    Load a ``.pcd`` file and return a clean point cloud.

    Parameters
    ----------
    file_path
        Path to a binary or ASCII PCD file readable by Open3D.

    Returns
    -------
    pcd
        An ``open3d.geometry.PointCloud`` with finite points only,
        or ``None`` if the file was empty / unreadable.
    """
    pcd = o3d.io.read_point_cloud(str(file_path))
    if pcd.is_empty():
        print(f"warning: empty point cloud at {file_path}")
        return None

    points = np.asarray(pcd.points)

    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        valid = np.where(np.isfinite(points).all(axis=1))[0]
        n_dropped = len(points) - len(valid)
        if n_dropped:
            print(f"  dropped {n_dropped} NaN/Inf points from {Path(file_path).name}")
        pcd = pcd.select_by_index(valid)

    return pcd
