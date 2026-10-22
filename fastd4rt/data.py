"""Dataset loaders: TAPVid-3D (official npz release), Sintel and Bonn."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

MODEL_HW = (256, 256)


def resize_video(video_rgb: np.ndarray, hw=MODEL_HW) -> np.ndarray:
    h, w = hw
    return np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA if f.shape[0] >= h else cv2.INTER_LINEAR)
                     for f in video_rgb])


def to_model_video(video_rgb: np.ndarray) -> torch.Tensor:
    """uint8 [T, H, W, 3] -> float [1, T, 3, 256, 256] on CUDA in [0, 1]."""
    return torch.from_numpy(resize_video(video_rgb)).cuda().float().permute(0, 3, 1, 2)[None] / 255.


def _decode_jpeg(b: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("failed to decode JPEG frame")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_tapvid3d(path: Path, max_frames: int | None = None) -> dict:
    """TAPVid-3D sequence with official queries.

    Returns video (model input), query frames ``tq`` and normalized query uv, camera-frame GT tracks [N, T, 3],
    visibility [N, T], world-frame GT tracks expressed in the camera of frame 0 (if extrinsics are available),
    and intrinsics rescaled to a 256 shortest side (as used by the official metric).
    """
    with np.load(path, allow_pickle=True) as z:
        jpgs = np.asarray(z["images_jpeg_bytes"])
        xyz = np.asarray(z["tracks_XYZ"], np.float64)
        vis = np.asarray(z["visibility"], bool)
        intr = np.asarray(z["fx_fy_cx_cy"], np.float64)
        qxyt = np.asarray(z["queries_xyt"], np.float64)
        w2c = np.asarray(z["extrinsics_w2c"], np.float64) if "extrinsics_w2c" in z.files else None
    T = min(len(jpgs), len(xyz), len(vis))
    if max_frames:
        T = min(T, max_frames)
    rgb = np.stack([_decode_jpeg(b) for b in jpgs[:T]])
    xyz, vis = xyz[:T], vis[:T]
    if w2c is not None:
        w2c = w2c[:T] @ np.linalg.inv(w2c[0])          # world = camera of frame 0
        c2w = np.linalg.inv(w2c)
        world = np.einsum("tij,tnj->tni", c2w[:, :3, :3], xyz) + c2w[:, None, :3, 3]
    else:
        world = xyz.copy()
    h, w = rgb.shape[1:3]
    keep = qxyt[:, 2] < T
    q = qxyt[keep]
    uvq = np.stack([q[:, 0] / max(w - 1, 1), q[:, 1] / max(h - 1, 1)], 1).clip(0, 1).astype(np.float32)
    return dict(T=T, tq=q[:, 2].astype(np.int64), uvq=uvq, video=to_model_video(rgb),
                aspect=torch.tensor([[w / max(h, 1)]], device="cuda"),
                gt_cam=xyz[:, keep].transpose(1, 0, 2), vis=vis[:, keep].T,
                gt_world=world[:, keep].transpose(1, 0, 2), intr=intr * (256 / min(h, w)))


def truncate(d: dict, Tp: int) -> dict:
    """First Tp frames of a loaded TAPVid-3D sequence (queries from later frames are dropped)."""
    if Tp >= d["T"]:
        return d
    keep = d["tq"] < Tp
    e = dict(d)
    e.update(T=Tp, tq=d["tq"][keep], uvq=d["uvq"][keep], video=d["video"][:, :Tp].clone(),
             gt_cam=d["gt_cam"][keep, :Tp], vis=d["vis"][keep, :Tp], gt_world=d["gt_world"][keep, :Tp])
    return e


# ---------------------------------------------------------------- Sintel / Bonn
SINTEL14 = ["alley_2", "ambush_4", "ambush_5", "ambush_6", "cave_2", "cave_4", "market_2", "market_5",
            "market_6", "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3"]
BONN5 = ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]
_TAG = 202021.25


def _read_dpt(p):
    with open(p, "rb") as f:
        assert abs(np.fromfile(f, np.float32, 1)[0] - _TAG) < 1e-3
        w, h = np.fromfile(f, np.int32, 2)
        return np.fromfile(f, np.float32, w * h).reshape(h, w)


def _read_cam(p):
    with open(p, "rb") as f:
        assert abs(np.fromfile(f, np.float32, 1)[0] - _TAG) < 1e-3
        K = np.fromfile(f, np.float64, 9).reshape(3, 3)
        N = np.fromfile(f, np.float64, 12).reshape(3, 4)
    return K, np.vstack([N, [0, 0, 0, 1]])


def load_sintel(root: Path, seq: str) -> dict:
    """Sintel training sequence (final pass): rgb, depth, intrinsics K, world-to-camera w2c."""
    root = Path(root)
    frames = sorted((root / "training/final" / seq).glob("frame_*.png"))
    rgb = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in frames])
    depth = np.stack([_read_dpt(root / "training/depth" / seq / (f.stem + ".dpt")) for f in frames])
    cams = [_read_cam(root / "training/camdata_left" / seq / (f.stem + ".cam")) for f in frames]
    return dict(rgb=rgb, depth=depth, K=np.stack([c[0] for c in cams]), w2c=np.stack([c[1] for c in cams]))


def load_bonn(root: Path, seq: str, start: int = 30, end: int = 140) -> dict:
    """Bonn RGB-D split used by MonST3R / CUT3R: sorted rgb and depth lists, frames [30:140], paired by index."""
    d = Path(root) / f"rgbd_bonn_dataset/rgbd_bonn_{seq}"
    f_rgb = sorted((d / "rgb").glob("*.png"))[start:end]
    f_d = sorted((d / "depth").glob("*.png"))[start:end]
    rgb = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in f_rgb])
    depth = np.stack([np.asarray(Image.open(f)).astype(np.float32) / 5000. for f in f_d])
    return dict(rgb=rgb, depth=depth)
