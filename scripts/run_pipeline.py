"""
Run the full pcd_motion pipeline on a directory of `.pcd` files.

For a single-frame directory, the script reports the object's
axis-aligned dimensions and PCA-derived principal axis.

For a multi-frame directory, the script additionally:

- Detects whether the object moves between frames.
- Estimates the average rotation rate (deg/s and RPM).
- Reports the mean rotation axis (unit vector).

Usage::

    python scripts/run_pipeline.py [DIRECTORY]

Defaults to ``data/sample/`` if no path is supplied.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from pcd_motion import (
    calculate_dimensions,
    calculate_speed_rate,
    determine_orientation,
    is_moving,
    load_and_preprocess_pcd,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        default=str(REPO_ROOT / "data" / "sample"),
        help="directory containing .pcd files (default: data/sample/)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="seconds between consecutive frames (default 1.0; used to "
             "convert ICP rotation per pair into a rotation rate)",
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    pcd_paths = sorted(directory.glob("*.pcd"))
    if not pcd_paths:
        print(f"no .pcd files in {directory}")
        return 1

    print(f"loading {len(pcd_paths)} frames from {directory} …")
    sequence = []
    for path in pcd_paths:
        cloud = load_and_preprocess_pcd(path)
        if cloud is not None:
            sequence.append(cloud)

    if not sequence:
        print("no usable point clouds — aborting")
        return 1

    if len(sequence) == 1:
        single_frame_report(sequence[0])
        return 0

    multi_frame_report(sequence, args.dt)
    return 0


def single_frame_report(pcd) -> None:
    dims = calculate_dimensions(pcd)
    if dims is not None:
        print(f"  AABB extents (W, H, D): {np.round(dims, 3)}")

    axis = determine_orientation(pcd)
    if axis is not None:
        print(f"  PCA principal axis    : {np.round(axis, 3)}")

    print("  motion: object is stationary (single frame)")


def multi_frame_report(sequence, dt_seconds: float) -> None:
    moving = is_moving(sequence)
    print(f"  motion detector       : object is {'MOVING' if moving else 'stationary'}")

    if not moving:
        return

    timestamps = np.arange(len(sequence)) * dt_seconds
    speed_dps, axis = calculate_speed_rate(sequence, timestamps)
    rpm = speed_dps / 360.0 * 60.0
    print(f"  average rotation rate : {speed_dps:.2f} deg/s  ({rpm:.2f} RPM)")
    if axis is not None:
        print(f"  average rotation axis : {np.round(axis, 3)}")

    for k, pcd in enumerate(sequence):
        dims = calculate_dimensions(pcd)
        if dims is None:
            continue
        print(f"  frame {k:>2d} extents     : {np.round(dims, 3)}")


if __name__ == "__main__":
    sys.exit(main())
