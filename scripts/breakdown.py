"""Runtime breakdown (encoder / decoder / other) for tracking (TAPVid-3D) and dense depth (Bonn).
One warm-up, then the median of three runs per video."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from _common import SPLITS, parser, setup
from fastd4rt.data import load_bonn, load_tapvid3d, to_model_video
from fastd4rt.engine import Engine
from fastd4rt.pipeline import dense, track_segments

ap = parser(__doc__)
ap.add_argument("--adt", type=Path, required=True, help="ADT .npz directory")
ap.add_argument("--bonn", type=Path, required=True, help="Bonn root")
ap.add_argument("--videos", type=int, default=3)
args = ap.parse_args()
model, cfgs, _ = setup(args)


class Timed(Engine):
    def reset(self):
        self.t_enc = self.t_dec = 0.

    def encode(self, *a):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = super().encode(*a)
        torch.cuda.synchronize(); self.t_enc += time.perf_counter() - t0
        return out

    def decode(self, *a):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = super().decode(*a)
        torch.cuda.synchronize(); self.t_dec += time.perf_counter() - t0
        return out


def measure(fn, eng, reps=3):
    eng.reset()
    fn()
    rows = []
    for _ in range(reps):
        eng.reset()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(); tot = time.perf_counter() - t0
        rows.append((eng.t_enc, eng.t_dec, tot - eng.t_enc - eng.t_dec, tot))
    m = np.median(np.array(rows), 0)
    return dict(encoder=float(m[0]), decoder=float(m[1]), other=float(m[2]), total=float(m[3]))


engines = {c: Timed(model, cfgs[c][0]) for c in cfgs}
out = {}
with torch.inference_mode():
    names = json.loads((SPLITS / "adt_39.json").read_text())["videos"][: args.videos]
    for task in ("camera", "world"):
        for c, (_, r, _) in cfgs.items():
            per = [measure(lambda: track_segments(engines[c], load, [task], r), engines[c])
                   for load in (load_tapvid3d(args.adt / f"{n}.npz") for n in names)]
            out[f"adt_{task}_{c}"] = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
            print(f"adt {task} {c}", out[f"adt_{task}_{c}"], flush=True)
    d = load_bonn(args.bonn, "balloon2")
    T, H, W = d["depth"].shape
    video, aspect = to_model_video(d["rgb"]), torch.tensor([[W / H]], device="cuda")
    for c, (_, _, G) in cfgs.items():
        out[f"bonn_depth_{c}"] = measure(lambda: dense(engines[c], video, aspect, T, G, ["depth"]), engines[c])
        print(f"bonn depth {c}", out[f"bonn_depth_{c}"], flush=True)
(args.out / "breakdown.json").write_text(json.dumps(out, indent=1))
