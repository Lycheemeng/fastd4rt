"""Camera pose on Sintel (paper Tab. 6, Sintel columns): ATE / RPE-T / RPE-R. Both methods use a 32x32 grid."""
from pathlib import Path

import numpy as np
import torch
from _common import parser, save, setup
from fastd4rt import metrics
from fastd4rt.data import SINTEL14, load_sintel, to_model_video
from fastd4rt.pipeline import camera_poses

ap = parser(__doc__)
ap.add_argument("--root", type=Path, required=True, help="Sintel root")
ap.add_argument("--grid", type=int, default=32)
ap.add_argument("--repeats", type=int, default=3)
args = ap.parse_args()
model, cfgs, engines = setup(args)
rows, warmed = [], set()
with torch.inference_mode():
    for seq in SINTEL14:
        d = load_sintel(args.root, seq)
        T, H, W = d["depth"].shape
        video, aspect = to_model_video(d["rgb"]), torch.tensor([[W / H]], device="cuda")
        gt = np.linalg.inv(d["w2c"])
        row = dict(seq=seq, T=T, configs={})
        for cname, (eng_name, _, _) in cfgs.items():
            if eng_name not in warmed:
                camera_poses(engines[eng_name], video, aspect, T, args.grid)
                warmed.add(eng_name)
            runs = [camera_poses(engines[eng_name], video, aspect, T, args.grid) for _ in range(args.repeats)]
            row["configs"][cname] = dict(pose=dict(seconds=float(np.median([r[1] for r in runs])), **metrics.camera_pose(runs[-1][0], gt)))
            print(seq, T, cname, {k: round(v, 4) for k, v in row["configs"][cname]["pose"].items()}, flush=True)
        rows.append(row)
        save(rows, args.out)
from _common import summarize  # noqa: E402
summarize(rows, {"pose": ["ATE", "RPE_t", "RPE_r"]})
