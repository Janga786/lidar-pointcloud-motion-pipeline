"""
Convert proprietary `.xbin` point-cloud captures to standard `.pcd`.

The vendor LiDAR records frames as a tightly-packed binary stream of
``float32`` triplets (``x, y, z`` per point, no header, no alignment
padding).  Open3D and Point Cloud Library both expect the standard
PCL `.pcd` ASCII or binary format, so a one-shot conversion is the
cleanest way to bring vendor captures into the rest of the pipeline.

Usage::

    python scripts/xbin_to_pcd.py INPUT.xbin [-o OUTPUT.pcd]

If ``-o`` is omitted, the output filename is derived by replacing
``.xbin`` with ``.pcd`` next to the source.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def read_xbin(path: Path) -> np.ndarray:
    """
    Parse an ``.xbin`` file into an (N, 3) ``float32`` ndarray.

    Each point is exactly 12 bytes: three little-endian IEEE-754
    single-precision floats.  Files whose total length is not a
    multiple of 12 are flagged but parsed up to the last whole
    triplet.
    """
    size = path.stat().st_size
    if size % 12 != 0:
        print(f"warning: {path.name} length is not a multiple of 12 — possible header.")

    points = []
    with path.open("rb") as f:
        while True:
            chunk = f.read(12)
            if len(chunk) != 12:
                break
            x, y, z = struct.unpack("<fff", chunk)
            points.append((x, y, z))
    return np.asarray(points, dtype=np.float32)


def write_pcd(points: np.ndarray, path: Path) -> None:
    """Write an (N, 3) array to a binary PCD file (PCL v0.7 format)."""
    n = points.shape[0]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(points.astype(np.float32).tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="path to .xbin source")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="output .pcd path (default: alongside input with .pcd suffix)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"error: {args.input} not found", file=sys.stderr)
        return 1

    output = args.output or args.input.with_suffix(".pcd")

    print(f"reading  {args.input.name} ...")
    points = read_xbin(args.input)
    print(f"  {len(points)} points")
    print(f"writing  {output.name} ...")
    write_pcd(points, output)
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
