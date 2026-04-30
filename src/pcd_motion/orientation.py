"""
Principal-axis orientation via PCA.

For an object lying along an unknown direction in 3-D space, the
**principal axis** is the eigenvector of the point cloud's covariance
matrix associated with the **largest** eigenvalue — equivalently, the
direction along which the cloud's variance is maximised.

This is the textbook PCA recipe applied to a 3 × *N* coordinate
matrix:

1. Subtract the centroid from every point.
2. Form the 3 × 3 covariance matrix.
3. Symmetric eigendecomposition (``np.linalg.eigh`` is faster and
   more numerically stable than the general ``np.linalg.eig`` here
   because the covariance matrix is guaranteed symmetric).
4. The eigenvector corresponding to the largest eigenvalue is the
   primary axis of the object.

The returned axis is **unsigned** — PCA cannot tell apart "long axis
points +x" from "long axis points -x".  If you need a signed
direction, pair this with a domain-specific tie-breaker (e.g. align
to camera forward).
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from numpy.typing import NDArray


def determine_orientation(
    pcd: o3d.geometry.PointCloud,
) -> NDArray[np.float64] | None:
    """
    Return the unit vector along the cloud's principal axis.

    Parameters
    ----------
    pcd
        A point cloud.  Must contain at least three points; otherwise
        the covariance matrix is rank-deficient and the principal
        axis is ill-defined.

    Returns
    -------
    axis
        Unit-length length-3 ``ndarray`` pointing along the largest
        principal direction, or ``None`` if the input is too small.
    """
    points = np.asarray(pcd.points)
    if points.shape[0] < 3:
        print("warning: at least three points required for PCA orientation.")
        return None

    centred = points - np.mean(points, axis=0)
    cov = np.cov(centred, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # eigh returns ascending eigenvalues; we want the largest.
    primary = eigvecs[:, np.argmax(eigvals)]
    return primary / np.linalg.norm(primary)
