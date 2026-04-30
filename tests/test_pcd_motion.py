"""Tests for the pcd_motion pipeline."""

from __future__ import annotations

import math

import numpy as np
import open3d as o3d
import pytest

from pcd_motion import (
    calculate_dimensions,
    calculate_speed_rate,
    compute_rotation_axis_angle,
    compute_rotation_angle,
    determine_orientation,
    is_moving,
)


# ---------------------------------------------------------------------------
# Synthetic-cloud helpers
# ---------------------------------------------------------------------------

def make_box_cloud(width=2.0, height=1.0, depth=0.5, density=20, seed=0):
    """Return a uniformly-sampled axis-aligned box of the given extent."""
    rng = np.random.default_rng(seed)
    n = density**3
    pts = rng.uniform(
        low=[-width / 2, -height / 2, -depth / 2],
        high=[width / 2, height / 2, depth / 2],
        size=(n, 3),
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def rotate_cloud(pcd, axis, angle_deg):
    """Return a rotated copy of ``pcd``."""
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(
        np.asarray(axis) * math.radians(angle_deg)
    )
    rotated = o3d.geometry.PointCloud(pcd)
    rotated.rotate(R, center=(0, 0, 0))
    return rotated


# ---------------------------------------------------------------------------
# dimensions
# ---------------------------------------------------------------------------

def test_dimensions_match_box_extents():
    box = make_box_cloud(width=2.0, height=1.0, depth=0.5, density=15)
    extents = calculate_dimensions(box)
    assert extents is not None
    np.testing.assert_allclose(extents, [2.0, 1.0, 0.5], atol=0.05)


def test_dimensions_returns_none_on_empty_cloud():
    empty = o3d.geometry.PointCloud()
    assert calculate_dimensions(empty) is None


def test_dimensions_apply_scaling():
    box = make_box_cloud(width=1.0, height=1.0, depth=1.0, density=10)
    out = calculate_dimensions(box, scaling_factors=[2.0, 3.0, 4.0])
    assert out is not None
    np.testing.assert_allclose(out, [2.0, 3.0, 4.0], atol=0.1)


# ---------------------------------------------------------------------------
# orientation
# ---------------------------------------------------------------------------

def test_orientation_recovers_long_axis():
    """A 5×0.1×0.1 stick should have its principal axis along x."""
    stick = make_box_cloud(width=5.0, height=0.1, depth=0.1, density=15)
    axis = determine_orientation(stick)
    assert axis is not None
    assert abs(abs(axis[0]) - 1.0) < 0.05
    assert abs(axis[1]) < 0.05
    assert abs(axis[2]) < 0.05


def test_orientation_returns_none_on_too_few_points():
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0]])
    assert determine_orientation(pcd) is None


# ---------------------------------------------------------------------------
# motion
# ---------------------------------------------------------------------------

def test_motion_detects_translation():
    a = make_box_cloud(seed=1)
    pts = np.asarray(a.points) + np.array([0.5, 0, 0])
    b = o3d.geometry.PointCloud()
    b.points = o3d.utility.Vector3dVector(pts)
    assert is_moving([a, b]) is True


def test_motion_returns_false_for_static_pair():
    a = make_box_cloud(seed=2)
    b = o3d.geometry.PointCloud(a)
    assert is_moving([a, b]) is False


def test_motion_returns_false_for_short_sequence():
    a = make_box_cloud(seed=3)
    assert is_moving([a]) is False


# ---------------------------------------------------------------------------
# speed (axis/angle decomposition + ICP)
# ---------------------------------------------------------------------------

def test_axis_angle_recovery_round_trip():
    """A known rotation matrix should round-trip through axis/angle."""
    axis = np.array([0.0, 0.0, 1.0])
    angle_deg = 30.0
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(
        axis * math.radians(angle_deg)
    )
    T = np.eye(4)
    T[:3, :3] = R
    a, n = compute_rotation_axis_angle(T)
    assert abs(a - angle_deg) < 1e-6
    np.testing.assert_allclose(n, axis, atol=1e-6)


def test_axis_angle_handles_identity():
    """Identity rotation must not raise — axis is degenerate but angle = 0."""
    a, n = compute_rotation_axis_angle(np.eye(4))
    assert a < 1e-6
    np.testing.assert_allclose(n, np.zeros(3))


def test_icp_recovers_small_z_rotation():
    """A 5° rotation about z should be recovered to within 1°."""
    a = make_box_cloud(width=2.0, height=1.0, depth=0.5, density=15, seed=10)
    b = rotate_cloud(a, [0, 0, 1], 5.0)
    angle, axis = compute_rotation_angle(a, b, threshold=0.5)
    assert abs(angle - 5.0) < 1.0
    np.testing.assert_allclose(np.abs(axis), [0, 0, 1], atol=0.1)


def test_speed_rate_returns_zero_for_short_sequence():
    a = make_box_cloud(seed=4)
    speed, axis = calculate_speed_rate([a], [0.0])
    assert speed == 0.0
    assert axis is None
