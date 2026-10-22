"""Tracking throughput (paper Tab. 3): maximum number of full-video 3D tracks from a 48-frame clip at a target FPS.

D4RT does not specify the clip length or whether encoding is counted; we use one 48-frame clip and include encoding
in the time budget 48 / f. Query points are uniform in [0,1]^2 (fixed seed) and start at the first frame. Runtime is
affine in the number of tracks M, so we fit t(M) = a + b M from M in {0, 2k, 8k, 32k} (median of 3), predict M_max,
and verify it by direct measurement, decreasing it by 3% until the target is met. At most 200k tracks are sampled.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from _common import parser, setup
from fastd4rt.data import load_tapvid3d
from fastd4rt.query import anchor_subset

ap = parser(__doc__)
ap.add_argument("--clip", type=Path, required=True, help="a TAPVid-3D .npz (first 48 frames are used)")
args = ap.parse_args()
model, cfgs, engines = setup(args)
d = load_tapvid3d(args.clip, 48)
video, aspect, T = d["video"], d["aspect"], d["T"]
uv = torch.rand(200000, 2, generator=torch.Generator().manual_seed(0)).cuda()


def make_call(eng, r):
    frames = anchor_subset(list(range(T)), r)

    def call(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        seg = eng.encode(video, aspect)
        if n > 0:
            k = len(frames)
            tt = torch.as_tensor(frames, device="cuda")[None].expand(n, k).reshape(-1)
            o = eng.decode(seg, uv[:n, 0, None].expand(n, k).reshape(-1), uv[:n, 1, None].expand(n, k).reshape(-1),
                           torch.zeros_like(tt), tt, tt)
            if k < T:
                x = o["xyz"].reshape(n, k, 3)
                h = torch.as_tensor(frames, device="cuda")
                t = torch.arange(T, device="cuda")
                hi = torch.searchsorted(h, t).clamp(1, k - 1)
                lo = hi - 1
                w = ((t - h[lo]).float() / (h[hi] - h[lo]).float())[None, :, None]
                _ = x[:, lo] * (1 - w) + x[:, hi] * w
        torch.cuda.synchronize()
        return time.perf_counter() - t0
    return call


res = {}
with torch.inference_mode():
    for cname, (eng_name, r, _) in cfgs.items():
        call = make_call(engines[eng_name], r)
        for _ in range(3):
            call(1000)
        med = lambda n: float(np.median([call(n) for _ in range(3)]))
        probes = [0, 2000, 8000, 32000]
        b, a = np.polyfit(probes, [med(n) for n in probes], 1)
        row = dict(encode_s=float(a), per_track_s=float(b))
        for f in (60, 24, 10, 1):
            budget = T / f
            n = 0 if budget <= a else min(int((budget - a) / b), len(uv))
            while n > 0 and med(n) > budget:
                n = int(n * 0.97)
            row[f"{f}fps"] = n
        res[cname] = row
        print(cname, row, flush=True)
(args.out / "throughput.json").write_text(json.dumps(res, indent=1))
