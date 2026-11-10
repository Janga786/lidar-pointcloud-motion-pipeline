# LiDAR Point-Cloud Motion Pipeline

A small Python package (Open3D + NumPy) that estimates the size, orientation, and rotation rate of an object from a sequence of time-of-flight point clouds.

It grew out of my algorithm work on a junior-design team project (CE 315, Fort Lewis College, Fall 2024): point-cloud dimension and rotation-axis estimation, RPM estimation, and conversion of the vendor `.xbin` format.

<p align="center">
<img src="docs/example_pointcloud.png" alt="Point cloud with bounding box" width="44%"/>
&nbsp;
<img src="docs/dimensions_visualization.png" alt="Dimensions sketch" width="44%"/>
</p>

## What it does

Given `.pcd` frames captured at known timestamps, the pipeline reports:

1. **Size**: axis-aligned bounding-box extents (`pcd_motion.dimensions`).
2. **Orientation**: unsigned principal axis from PCA (`pcd_motion.orientation`).
3. **Motion**: a mean-displacement test between frames that gates the ICP step (`pcd_motion.motion`).
4. **Rotation rate and axis**: point-to-point ICP between frame pairs, decomposed into axis/angle, averaged to deg/s and RPM (`pcd_motion.speed`).

Ten sample frames (320 x 240 time-of-flight captures of a rotating rectangle) are in [`data/sample/`](data/sample/), so the pipeline runs without external data.

**Scope.** The 12 unit tests use synthetic point clouds. The pipeline has not been checked against independent ground truth on real captures, and no accuracy figures are claimed here.

## Quick start

```bash
git clone https://github.com/Janga786/lidar-pointcloud-motion-pipeline.git
cd lidar-pointcloud-motion-pipeline

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Sample run on the included rotating-rectangle dataset
python scripts/run_pipeline.py
```

The driver prints the motion-detector result, the average rotation rate (deg/s and RPM), the mean rotation axis, and per-frame bounding-box extents.

---

## The pipeline, stage by stage

### 1. `preprocess.py` — load and clean

LiDAR captures contain spurious returns at infinity (no echo within
the sensor's range) and NaN values where the time-of-flight
electronics couldn't lock onto a phase.  Both wreck downstream
algorithms — PCA gets dominated by a single Inf, ICP fails its
Cholesky decomposition, bounding boxes blow up to infinite extent.
The loader strips these before any later stage sees them.

### 2. `dimensions.py` — bounding-box extents

The simplest possible "size" estimator: the axis-aligned bounding box
of the cloud.  Robust to typical LiDAR sparsity (a single returning
beam still contributes a sample to each axis) but sensitive to object
orientation (a rotated rectangle reads larger than its physical size
because the AABB has to grow to enclose the rotated extents).
Pair with `orientation.py` for orientation-invariant sizing.

### 3. `orientation.py` — PCA principal axis

The textbook PCA recipe applied to a 3 × *N* coordinate matrix:
subtract the centroid, form the 3 × 3 covariance matrix, run a
symmetric eigendecomposition (`np.linalg.eigh` is faster and more
numerically stable than `np.linalg.eig` because the covariance
matrix is guaranteed symmetric), and return the eigenvector with the
largest eigenvalue.  The result is the **unsigned** direction of
maximum variance — PCA cannot tell apart "long axis points +x" from
"long axis points −x".

### 4. `motion.py` — coarse motion gate

A cheap mean-displacement test between consecutive frames.  Used to
*gate* the expensive ICP computation in stage 5: if no motion is
detected, the pipeline reports the object as stationary and skips
the rotation analysis entirely.  False positives are tolerable (an
extra ICP evaluation), false negatives are bad (missed motion), so
the threshold is intentionally low (1 cm by default).

### 5. `speed.py` — ICP-based rotation rate

The headline algorithm.  Given two consecutive point clouds, point-to-
point ICP recovers the SO(3) transform that aligns the second to the
first.  We decompose that transform into an axis/angle pair using

$$
\theta = \arccos\!\left(\frac{\operatorname{tr} R - 1}{2}\right),
\qquad
\hat n = \frac{1}{2 \sin \theta}
\bigl( R_{32} - R_{23},\; R_{13} - R_{31},\; R_{21} - R_{12} \bigr)
$$

with safeguards at the degenerate angles (θ = 0 and θ = π).
Summed over the whole sequence, this produces the average rotation
rate in deg/s and a mean rotation axis.

For tumbling motion (axis itself rotating in the body frame), the
arithmetic mean of axes is no longer the right answer; the natural
generalisation is a Karcher mean on SO(3), which is left as an
extension exercise.

---

## Repository layout

```
lidar-pointcloud-motion-pipeline/
├── src/pcd_motion/             # the library (importable Python package)
│   ├── __init__.py             # public API
│   ├── preprocess.py           # 1. load + drop NaN/Inf
│   ├── dimensions.py           # 2. AABB extents
│   ├── orientation.py          # 3. PCA principal axis
│   ├── motion.py               # 4. frame-pair displacement test
│   └── speed.py                # 5. ICP rotation-rate estimation
│
├── scripts/
│   ├── run_pipeline.py         # end-to-end driver
│   └── xbin_to_pcd.py          # convert vendor `.xbin` → standard `.pcd`
│
├── data/sample/                # 10-frame rotating-rectangle dataset
│
├── tests/
│   └── test_pcd_motion.py      # synthetic-cloud unit tests
│
├── docs/                       # figures and test-plan table
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

## Testing

```bash
pytest -q                # 12 synthetic-cloud unit tests
```

The tests use procedurally generated point clouds (no LiDAR
hardware required) to verify:

- AABB extents recover the synthesised box dimensions to within 5 %.
- PCA correctly identifies the long axis of a 5 × 0.1 × 0.1 stick.
- The motion detector flags translated clouds and clears static ones.
- The axis/angle decomposition round-trips through a known SO(3)
  rotation to floating-point precision.
- ICP recovers a 5° rotation about z to within 1°.

## Test plan

The sample frames come from a rotating-target rig built for the course project. The team's parametric test plan (object shape, orientation, and 0, 1, and 5 RPM) is below; only the 10 frames in `data/sample/` are included in this repository.

<p align="center">
<img src="docs/study_setup.png" alt="Parametric test plan table" width="60%"/>
</p>

## License

[MIT](LICENSE)
