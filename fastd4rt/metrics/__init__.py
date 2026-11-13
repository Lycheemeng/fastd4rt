from __future__ import annotations

import numpy as np
import torch

from ..align import umeyama_np
from . import tapvid3d

WORLD_THRESHOLDS = (0.1, 0.3, 0.5, 1.0)


def tracking_camera(gt, vis, pred, intr):
    raw = tapvid3d.compute_tapvid3d_metrics(
        gt_occluded=~vis, gt_tracks=gt, pred_occluded=~(pred[..., 3] >= .5), pred_tracks=pred[..., :3],
        intrinsics_params=intr, scaling="median", use_fixed_metric_threshold=False, order="n t")
    r = {k: float(np.asarray(v).reshape(-1)[0]) for k, v in raw.items()}
    return dict(AJ=r["average_jaccard"], APD3D=r["average_pts_within_thresh"], OA=r["occlusion_accuracy"])


def tracking_world(gt, pred):
    g, p = gt.reshape(-1, 3), pred[..., :3].reshape(-1, 3)
    ok = np.isfinite(g).all(1) & np.isfinite(p).all(1)
    s = np.median(np.linalg.norm(g[ok], axis=1)) / max(np.median(np.linalg.norm(p[ok], axis=1)), 1e-12)
    d = np.linalg.norm(p[ok] * s - g[ok], axis=1)
    return dict(APD=float(np.mean([(d <= th).mean() for th in WORLD_THRESHOLDS])),
                L1=float(np.abs(p[ok] * s - g[ok]).sum(1).mean()))


def _scale_l1(p, g, iters=10, eps=1e-6):
    s = (p * g).sum() / (p * p).sum()
    for _ in range(iters):
        w = 1.0 / (s * p - g).abs().clamp(min=eps)
        s = (w * p * g).sum() / (w * p * p).sum()
    return s


def _scale_shift_l1(p, g, lr=1e-4, iters=1000, tol=1e-6):
    s = torch.tensor((g.median() / p.median()).item(), dtype=p.dtype, device=p.device, requires_grad=True)
    t = torch.tensor(0.0, dtype=p.dtype, device=p.device, requires_grad=True)
    opt = torch.optim.Adam([s, t], lr=lr)
    prev = None
    for _ in range(iters):
        opt.zero_grad()
        loss = (s * p + t - g).abs().sum()
        loss.backward()
        opt.step()
        if prev is not None and abs(prev - loss.item()) < tol:
            break
        prev = loss.item()
    return s.detach(), t.detach()


def depth(pred, gt, max_depth=70.0, post_clip_max=None):
    with torch.inference_mode(False), torch.enable_grad():
        pred, gt = pred.clone().double(), gt.clone().double()
        mask = (gt > 0) & (gt < max_depth) & torch.isfinite(gt) & torch.isfinite(pred)
        p, g = pred[mask], gt[mask]
        out = {"valid_pixels": int(mask.sum())}
        for name, q in (("S", _scale_l1(p, g) * p), ("SS", None)):
            if q is None:
                s, t = _scale_shift_l1(p, g)
                q = s * p + t
            if post_clip_max is not None:
                q = q.clamp(max=post_clip_max)
            out[f"AbsRel_{name}"] = float(((q - g).abs() / g).mean())
            out[f"delta1_{name}"] = float((torch.maximum(q / g, g / q) < 1.25).double().mean())
    return out


def pointcloud_l1(pred_xyz, gt_depth, K, w2c, max_depth=70.0):
    T, H, W = gt_depth.shape
    v, u = torch.meshgrid(torch.arange(H, device="cuda", dtype=torch.float64),
                          torch.arange(W, device="cuda", dtype=torch.float64), indexing="ij")
    w2c0 = torch.as_tensor(w2c[0], device="cuda")
    gts, prs = [], []
    for t in range(T):
        z = gt_depth[t].double()
        m = (z > 0) & (z < max_depth) & torch.isfinite(z)
        k = torch.as_tensor(K[t], device="cuda")
        cam = torch.stack([(u[m] - k[0, 2]) / k[0, 0] * z[m], (v[m] - k[1, 2]) / k[1, 1] * z[m], z[m],
                           torch.ones_like(z[m])], 1)
        gts.append((w2c0 @ torch.linalg.inv(torch.as_tensor(w2c[t], device="cuda")) @ cam.T).T[:, :3])
        prs.append(pred_xyz[t][m].double())
    g, p = torch.cat(gts), torch.cat(prs)
    g, p = g - g.mean(0), p - p.mean(0)
    g, p = g / g.norm(dim=1).mean(), p / p.norm(dim=1).mean()
    return float((p - g).abs().sum(1).mean())


def camera_pose(est, gt):
    s, R, t = umeyama_np(est[:, :3, 3], gt[:, :3, 3])
    a = est.copy()
    a[:, :3, :3] = R @ est[:, :3, :3]
    a[:, :3, 3] = (s * (R @ est[:, :3, 3].T)).T + t
    ate = float(np.sqrt((np.linalg.norm(a[:, :3, 3] - gt[:, :3, 3], axis=1) ** 2).mean()))
    te, re = [], []
    for i in range(len(gt) - 1):
        E = np.linalg.inv(np.linalg.inv(gt[i]) @ gt[i + 1]) @ (np.linalg.inv(a[i]) @ a[i + 1])
        te.append(np.linalg.norm(E[:3, 3]))
        re.append(np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1))))
    rmse = lambda x: float(np.sqrt(np.mean(np.square(x))))
    return dict(ATE=ate, RPE_t=rmse(te), RPE_r=rmse(re))
