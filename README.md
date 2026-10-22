# FastD4RT: Training-Free Acceleration of Query-Based 4D Reconstruction

FastD4RT accelerates D4RT at test time without changing its weights or its inference protocol.

* **Query reduction.** 3D tracks are decoded only at sparse temporal anchors and interpolated; depth maps and
  point clouds are decoded on a coarse grid and upsampled.
* **Efficient execution path.** bf16 autocast, fused attention that reuses the original projection weights,
  decoder keys/values computed once per segment, and a compiled encoder.
* **Anchored windows (optional, long videos).** Shared anchor frames in every window, one Sim(3) per window to the
  first window and a single track hand-over, instead of chained segments.

This repository also contains our implementation of the D4RT inference and evaluation protocol that is needed to
compare against D4RT: segment-based long-video inference (Appendix B of D4RT) with a track hand-over for queries
from any source frame, and the evaluation of 3D tracking, point clouds, video depth, camera pose and tracking
throughput.

## Setup

```bash
pip install -r requirements.txt
```

FastD4RT does not include model code or weights. It loads the D4RT model definition and checkpoints of the
community OpenD4RT release; install it separately and set

```bash
export OPEND4RT_ROOT=/path/to/Open-d4rt
CKPT=$OPEND4RT_ROOT/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG   # 48-frame model (model.yaml + opend4rt.ckpt)
```

Data: TAPVid-3D (`.npz` release with `queries_xyt`), MPI Sintel (training images + depth/camera data) and the Bonn
RGB-D dynamic dataset.

## Usage

```python
from fastd4rt.model import load_model
from fastd4rt.engine import Engine
from fastd4rt.data import load_tapvid3d
from fastd4rt.pipeline import track_segments

model = load_model(CKPT)
d = load_tapvid3d("adt_mini/Apartment_release_clean_seq133_0.npz")
pred, seconds, _ = track_segments(Engine(model, "bf16c"), d, tasks=("camera", "world"), r=3)
```

`Engine(model, "fp32")` is the reference path; `track_anchored(...)` uses anchored windows.

## Reproducing the experiments

| Experiment | Command |
|---|---|
| 3D tracking (Tab. 4) | `python scripts/eval_tracking.py --ckpt $CKPT --data <tapvid3d>/adt_mini --subset adt --out out/adt` (also `drivetrack`, `pstudio`) |
| Point cloud / video depth (Tab. 5) | `python scripts/eval_dense.py --ckpt $CKPT --dataset sintel --root <sintel> --out out/sintel` (also `--dataset bonn`) |
| Camera pose (Tab. 6, Sintel) | `python scripts/eval_pose.py --ckpt $CKPT --root <sintel> --out out/pose` |
| Tracking throughput (Tab. 3) | `python scripts/throughput.py --ckpt $CKPT --clip <adt npz> --out out/throughput` |
| Runtime breakdown | `python scripts/breakdown.py --ckpt $CKPT --adt <adt dir> --bonn <bonn> --out out/breakdown` |
| Long videos | `python scripts/eval_tracking.py ... --max-frames 128 [--anchored]` |
| Ablations | `--configs d4rt,bf16,bf16c,bf16c_r2,ours,bf16c_g64,ours_fp8` (see `fastd4rt/configs.py`) |

`d4rt` is the FP32 reference under the same protocol and `ours` is FastD4RT (BF16 + compile, r = 3, G = 128).
The ADT videos used in the paper are listed in `splits/adt_39.json`; DriveTrack and PStudio use all videos of the
official TAPVid-3D minival split. Timing: single NVIDIA H20, PyTorch 2.6, CUDA 12.4.

## Repository structure

```
fastd4rt/
  engine.py     execution paths (fp32 reference, bf16, bf16 + compile, FP8 ablation)
  pipeline.py   segments, Sim(3) chaining, track hand-over, anchored windows, dense outputs, camera pose
  query.py      temporal anchors, interpolation, coarse grids, upsampling
  data.py       TAPVid-3D, Sintel and Bonn loaders
  metrics/      tracking (camera / world), depth (CUT3R protocol), point cloud, camera pose
scripts/        one script per experiment
splits/         ADT video list
```

`fastd4rt/metrics/tapvid3d.py` is the official TAPVid-3D metric from google-deepmind/tapnet (Apache-2.0),
included unmodified.

## License

Apache-2.0.
