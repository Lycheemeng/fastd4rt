"""Long-video inference for tracking, dense outputs and camera pose.

Segment-based pipeline (Appendix B of D4RT): overlapping 48-frame segments with stride 24, each encoded once;
adjacent segments registered by Sim(3) (Umeyama on the 85% most confident points of a dense overlap grid) and
chained to the first segment. Tracks are handed over between segments through the decoded 2D position at the
overlap frame with the highest predicted visibility (D4RT does not describe this step).

Anchored windows (FastD4RT extension for long videos): m shared anchor frames (frame 0 included) in every window,
non-overlapping blocks of consecutive frames, one Sim(3) per window directly to window 0, one hand-over per track.

All functions return predictions and wall-clock seconds (CUDA-synchronised) per task.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .align import compose, umeyama, umeyama_np
from .engine import CLIP_LEN
from .query import anchor_subset, grid, interp_fill

STRIDE = 24
ALIGN_GRID = 32


def sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def segments(T: int, L: int = CLIP_LEN, stride: int = STRIDE) -> list[int]:
    if T <= L:
        return [0]
    starts = list(range(0, T - L + 1, stride))
    if starts[-1] != T - L:
        starts.append(T - L)
    return starts


def _expand(u, v, ts, tt):
    """Cartesian product of per-query (u, v, t_src) with a list of target indices tt."""
    nq, nf = len(u), len(tt)
    tt = torch.as_tensor(tt, device=u.device)
    return (u[:, None].expand(nq, nf).reshape(-1), v[:, None].expand(nq, nf).reshape(-1),
            ts[:, None].expand(nq, nf).reshape(-1), tt[None].expand(nq, nf).reshape(-1))


def _grid_sim3(eng, seg_a, seg_b, frames_a, frames_b, keep_ratio=0.85):
    """Sim(3) mapping segment b to segment a from a grid decoded at corresponding frames (t_cam = segment start)."""
    gu, gv = grid(ALIGN_GRID)
    pts = []
    for seg, frames in ((seg_a, frames_a), (seg_b, frames_b)):
        nf = len(frames)
        tt = torch.as_tensor(frames, device="cuda")[:, None].expand(nf, gu.numel()).reshape(-1)
        pts.append(eng.decode(seg, gu[None].expand(nf, -1).reshape(-1), gv[None].expand(nf, -1).reshape(-1),
                              tt, tt, torch.zeros_like(tt)))
    c = torch.minimum(pts[0]["conf"], pts[1]["conf"])
    keep = c >= torch.quantile(c, 1 - keep_ratio)
    return umeyama(pts[1]["xyz"][keep].double(), pts[0]["xyz"][keep].double())


def chained_sim3(eng, segs, starts, seg_end):
    """Sim(3) of every segment to segment 0, chained through the overlaps (every 4th overlap frame)."""
    ident = (torch.ones((), dtype=torch.float64, device="cuda"), torch.eye(3, dtype=torch.float64, device="cuda"),
             torch.zeros(3, dtype=torch.float64, device="cuda"))
    sim3 = {0: ident}
    for k in range(1, len(starts)):
        ov = list(range(starts[k], seg_end[k - 1] + 1))[::4]
        rel = _grid_sim3(eng, segs[k - 1], segs[k], [f - starts[k - 1] for f in ov], [f - starts[k] for f in ov])
        sim3[k] = compose(sim3[k - 1], rel)
    return sim3


def _apply(sim, xyz):
    s, R, t = sim
    return (s * xyz.double() @ R.T + t).float()


# ======================================================================= tracking (segments)
def track_segments(eng, d, tasks=("camera", "world"), r=1):
    """Track all official queries of sequence ``d`` with chained segments; r > 1 enables temporal anchors."""
    T, n = d["T"], len(d["tq"])
    starts = segments(T)
    K = len(starts)
    secs = {t: 0. for t in tasks}
    t0 = sync_time()
    segs = [eng.encode(d["video"][:, a:a + CLIP_LEN], d["aspect"]) for a in starts]
    enc = sync_time() - t0
    for t in tasks:
        secs[t] += enc
    seg_end = [min(a + CLIP_LEN, T) - 1 for a in starts]
    centre = torch.tensor([(a + e) / 2 for a, e in zip(starts, seg_end)], device="cuda")
    tq = torch.as_tensor(d["tq"], device="cuda")
    inside = (tq[:, None] >= torch.tensor(starts, device="cuda")[None]) & \
             (tq[:, None] <= torch.tensor(seg_end, device="cuda")[None])
    home = torch.where(inside, (tq[:, None] - centre[None]).abs(), torch.full((n, K), 1e9, device="cuda")).argmin(1)
    uvq = torch.as_tensor(d["uvq"], device="cuda")
    preds = {t: torch.zeros(n, T, 4, device="cuda") for t in tasks}
    sim3 = {}
    if "world" in tasks:
        t0 = sync_time()
        sim3 = chained_sim3(eng, segs, starts, seg_end)
        secs["world"] += sync_time() - t0

    def decode_group(task, k, qi, src_local, uv_src, frames):
        a = starts[k]
        dec = anchor_subset(frames, r)
        u, v, ts, tt = _expand(uv_src[:, 0], uv_src[:, 1], src_local, [f - a for f in dec])
        tc = tt.clone() if task == "camera" else torch.zeros_like(tt)
        o = eng.decode(segs[k], u, v, ts, tt, tc)
        nq, nf = len(qi), len(dec)
        xyz = o["xyz"].reshape(nq, nf, 3)
        if task == "world":
            xyz = _apply(sim3[k], xyz)
        block = torch.zeros(nq, T, 4, device="cuda")
        block[:, dec, :3] = xyz
        block[:, dec, 3] = o["vis"].reshape(nq, nf)
        if len(dec) < len(frames):
            interp_fill(block, dec, T)
        preds[task][qi[:, None], torch.as_tensor(frames, device="cuda")[None]] = block[:, frames]
        return dict(dec=dec, uv=o["uv"].reshape(nq, nf, 2), vis=o["vis"].reshape(nq, nf))

    def hand_over(out, qi, overlap_lo, overlap_hi):
        ov = [f for f in out["dec"] if overlap_lo <= f <= overlap_hi]
        pos = torch.as_tensor([out["dec"].index(f) for f in ov], device="cuda")
        best = out["vis"][:, pos].argmax(1)
        h = torch.as_tensor(ov, device="cuda")[best]
        return h, out["uv"][torch.arange(len(qi), device="cuda"), pos[best]]

    for task in tasks:
        t0 = sync_time()
        for k in range(K):
            qi = torch.nonzero(home == k).squeeze(1)
            if len(qi) == 0:
                continue
            out = decode_group(task, k, qi, tq[qi] - starts[k], uvq[qi], list(range(starts[k], seg_end[k] + 1)))
            cur, ck = out, k                                     # forward hand-over
            while ck + 1 < K:
                h, uvh = hand_over(cur, qi, starts[ck + 1], seg_end[ck])
                cur = decode_group(task, ck + 1, qi, h - starts[ck + 1], uvh, list(range(seg_end[ck] + 1, seg_end[ck + 1] + 1)))
                ck += 1
            cur, ck = out, k                                     # backward hand-over
            while ck - 1 >= 0:
                h, uvh = hand_over(cur, qi, starts[ck], seg_end[ck - 1])
                cur = decode_group(task, ck - 1, qi, h - starts[ck - 1], uvh, list(range(starts[ck - 1], starts[ck])))
                ck -= 1
        secs[task] += sync_time() - t0
    return {t: p.cpu().numpy().astype(np.float64) for t, p in preds.items()}, secs, dict(windows=K)


# ======================================================================= tracking (anchored windows)
def anchored_windows(T: int, m: int = 8, L: int = CLIP_LEN):
    """Windows (sorted frame list, owned block) with m shared anchor frames; blocks do not overlap."""
    if T <= L:
        return [(list(range(T)), list(range(T)))], [0]
    G = sorted({int(round(i * T / m)) for i in range(m)})
    if 0 not in G:
        G = [0] + G[1:]
    blocks, block = [], []
    for t in range(T):
        block.append(t)
        if len(set(G) | set(block)) == L:
            blocks.append(block)
            block = []
    wins = [(sorted(set(G) | set(b)), b) for b in blocks]
    if block:   # last block: fill the window with earlier frames so it still has L unique frames
        win = set(G) | set(block)
        for f in range(block[0] - 1, -1, -1):
            if len(win) >= L:
                break
            if f not in G:
                win.add(f)
        wins.append((sorted(win), block))
    return wins, G


def track_anchored(eng, d, tasks=("camera", "world"), m=8, r=1):
    """Track all official queries with anchored windows (one hand-over, Sim(3) of each window to window 0)."""
    T, n = d["T"], len(d["tq"])
    wins, G = anchored_windows(T, m)
    K = len(wins)
    secs = {t: 0. for t in tasks}
    t0 = sync_time()
    segs = [eng.encode(d["video"][:, w], d["aspect"]) for w, _ in wins]
    enc = sync_time() - t0
    for t in tasks:
        secs[t] += enc
    owner = {f: k for k, (_, owned) in enumerate(wins) for f in owned}
    home = np.array([owner[int(s)] for s in d["tq"]])
    sims = {}
    if "world" in tasks and K > 1:
        t0 = sync_time()
        for k in range(1, K):
            sims[k] = _grid_sim3(eng, segs[0], segs[k], [wins[0][0].index(a) for a in G], [wins[k][0].index(a) for a in G])
        secs["world"] += sync_time() - t0
    uvq = torch.as_tensor(d["uvq"], device="cuda")
    preds = {t: torch.zeros(n, T, 4, device="cuda") for t in tasks}

    def dec(task, k, qi, src_local, uv_src, frames, interp):
        w = wins[k][0]
        fr = anchor_subset(frames, r) if interp else frames
        u, v, ts, tt = _expand(uv_src[:, 0], uv_src[:, 1], src_local, [w.index(f) for f in fr])
        tc = tt.clone() if task == "camera" else torch.full_like(tt, w.index(0))
        o = eng.decode(segs[k], u, v, ts, tt, tc)
        nq, nf = len(qi), len(fr)
        xyz = o["xyz"].reshape(nq, nf, 3)
        if task == "world" and k in sims:
            xyz = _apply(sims[k], xyz)
        vis, uv = o["vis"].reshape(nq, nf), o["uv"].reshape(nq, nf, 2)
        if interp and len(fr) < len(frames):
            block = torch.zeros(nq, T, 4, device="cuda")
            block[:, fr, :3] = xyz
            block[:, fr, 3] = vis
            interp_fill(block, fr, T)
            return block[:, frames, :3], block[:, frames, 3], uv
        return xyz, vis, uv

    for task in tasks:
        t0 = sync_time()
        for k in range(K):
            qi_np = np.nonzero(home == k)[0]
            if len(qi_np) == 0:
                continue
            qi = torch.as_tensor(qi_np, device="cuda")
            w, owned = wins[k]
            src = torch.as_tensor([w.index(int(d["tq"][i])) for i in qi_np], device="cuda")
            xyz, vis, _ = dec(task, k, qi, src, uvq[qi], owned, True)
            oi = torch.as_tensor(owned, device="cuda")[None]
            preds[task][qi[:, None], oi, :3] = xyz
            preds[task][qi[:, None], oi, 3] = vis
            if K == 1:
                continue
            _, gv, guv = dec(task, k, qi, src, uvq[qi], G, False)      # exact decode at the anchor frames
            best = gv.argmax(1)
            g_star = torch.as_tensor(G, device="cuda")[best]
            uv_g = guv[torch.arange(len(qi), device="cuda"), best]
            for j in range(K):
                if j == k:
                    continue
                wj, owned_j = wins[j]
                src_j = torch.as_tensor([wj.index(int(g)) for g in g_star.tolist()], device="cuda")
                xyz_j, vis_j, _ = dec(task, j, qi, src_j, uv_g, owned_j, True)
                oj = torch.as_tensor(owned_j, device="cuda")[None]
                preds[task][qi[:, None], oj, :3] = xyz_j
                preds[task][qi[:, None], oj, 3] = vis_j
        secs[task] += sync_time() - t0
    return {t: p.cpu().numpy().astype(np.float64) for t, p in preds.items()}, secs, dict(windows=K)


# ======================================================================= dense outputs and camera pose
def _owner_segments(T):
    starts = segments(T)
    seg_end = [min(a + CLIP_LEN, T) - 1 for a in starts]
    centre = [(a + e) / 2 for a, e in zip(starts, seg_end)]
    owner = [min((k for k in range(len(starts)) if starts[k] <= f <= seg_end[k]), key=lambda k: abs(f - centre[k]))
             for f in range(T)]
    return starts, seg_end, owner


def dense(eng, video, aspect, T, G=256, tasks=("depth", "pointcloud")):
    """Depth (u,v,t,t,t) and point cloud (u,v,t,t,segment start -> frame 0) on a G x G grid of every frame.

    Returns {task: [T, G, G, 3]}; depth predictions of a segment are multiplied by its chained scale.
    """
    starts, seg_end, owner = _owner_segments(T)
    secs = {t: 0. for t in tasks}
    t0 = sync_time()
    segs = [eng.encode(video[:, a:a + CLIP_LEN], aspect) for a in starts]
    sim3 = chained_sim3(eng, segs, starts, seg_end)
    common = sync_time() - t0
    for t in tasks:
        secs[t] += common
    u, v = grid(G)
    out = {t: torch.empty(T, G * G, 3, device="cuda") for t in tasks}
    for task in tasks:
        t0 = sync_time()
        for k in range(len(starts)):
            frames = [f for f in range(T) if owner[f] == k]
            if not frames:
                continue
            nf = len(frames)
            tt = torch.as_tensor([f - starts[k] for f in frames], device="cuda")[:, None].expand(nf, G * G).reshape(-1)
            tc = tt.clone() if task == "depth" else torch.zeros_like(tt)
            xyz = eng.decode(segs[k], u[None].expand(nf, -1).reshape(-1), v[None].expand(nf, -1).reshape(-1),
                             tt, tt, tc)["xyz"].reshape(nf, G * G, 3)
            out[task][frames] = xyz * sim3[k][0].float() if task == "depth" else _apply(sim3[k], xyz)
        secs[task] += sync_time() - t0
    return {t: x.reshape(T, G, G, 3) for t, x in out.items()}, secs, dict(segments=len(starts))


def camera_poses(eng, video, aspect, T, G=32):
    """Camera-to-world poses [T, 4, 4] (world = camera of frame 0), following D4RT Sec. 3:
    rigid Umeyama between (u,v,t,t,t) and (u,v,t,t,ref) on a G x G grid; segments mapped by the chained Sim(3)."""
    starts, seg_end, owner = _owner_segments(T)
    t0 = sync_time()
    segs = [eng.encode(video[:, a:a + CLIP_LEN], aspect) for a in starts]
    sim3 = {k: tuple(x.cpu().numpy() for x in s) for k, s in chained_sim3(eng, segs, starts, seg_end).items()}
    u, v = grid(G)
    n = G * G
    poses = np.tile(np.eye(4), (T, 1, 1))
    for k in range(len(starts)):
        frames = [f for f in range(T) if owner[f] == k]
        if not frames:
            continue
        nf = len(frames)
        tt = torch.as_tensor([f - starts[k] for f in frames], device="cuda")[:, None].expand(nf, n).reshape(-1)
        uu, vv = u[None].expand(nf, -1).reshape(-1), v[None].expand(nf, -1).reshape(-1)
        cam = eng.decode(segs[k], uu, vv, tt, tt, tt)["xyz"].reshape(nf, n, 3).double().cpu().numpy()
        ref = eng.decode(segs[k], uu, vv, tt, tt, torch.zeros_like(tt))["xyz"].reshape(nf, n, 3).double().cpu().numpy()
        sk, Rk, tk = sim3[k]
        for i, f in enumerate(frames):
            _, R, t = umeyama_np(cam[i], ref[i], with_scale=False)
            poses[f, :3, :3] = Rk @ R
            poses[f, :3, 3] = float(sk) * (Rk @ t) + tk
    return poses, sync_time() - t0, dict(segments=len(starts))
