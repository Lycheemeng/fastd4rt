"""Point cloud (Sintel) and video depth (Sintel, Bonn), paper Tab. 5."""
from pathlib import Path

import numpy as np
import torch
from _common import parser, save, setup
from fastd4rt import metrics
from fastd4rt.data import BONN5, SINTEL14, load_bonn, load_sintel, to_model_video
from fastd4rt.pipeline import dense
from fastd4rt.query import upsample

ap = parser(__doc__)
ap.add_argument("--dataset", choices=["sintel", "bonn"], required=True)
ap.add_argument("--root", type=Path, required=True, help="Sintel root (training/...) or Bonn root (rgbd_bonn_dataset/...)")
ap.add_argument("--seqs", default="")
args = ap.parse_args()
model, cfgs, engines = setup(args)
seqs = args.seqs.split(",") if args.seqs else (SINTEL14 if args.dataset == "sintel" else BONN5)
tasks = ["depth", "pointcloud"] if args.dataset == "sintel" else ["depth"]
clip = 70.0 if args.dataset == "sintel" else None
rows, warmed = [], set()
with torch.inference_mode():
    for seq in seqs:
        d = load_sintel(args.root, seq) if args.dataset == "sintel" else load_bonn(args.root, seq)
        T, H, W = d["depth"].shape
        video, aspect = to_model_video(d["rgb"]), torch.tensor([[W / H]], device="cuda")
        gt = torch.from_numpy(d["depth"]).cuda()
        row = dict(seq=seq, T=T, configs={})
        for cname, (eng_name, _, G) in cfgs.items():
            if eng_name not in warmed:
                dense(engines[eng_name], video, aspect, T, 32, tasks)
                warmed.add(eng_name)
            pred, secs, info = dense(engines[eng_name], video, aspect, T, G, tasks)
            res = dict(info, grid=G)
            if "depth" in pred:
                z = upsample(pred["depth"][..., 2:3], (H, W))[..., 0]
                res["depth"] = dict(seconds=secs["depth"], **metrics.depth(z, gt, 70.0, clip))
            if "pointcloud" in pred:
                res["pointcloud"] = dict(seconds=secs["pointcloud"],
                                         L1=metrics.pointcloud_l1(upsample(pred["pointcloud"], (H, W)), gt, d["K"], d["w2c"]))
            row["configs"][cname] = res
            print(seq, T, cname, {t: {k: round(v, 4) for k, v in x.items()} for t, x in res.items() if isinstance(x, dict)}, flush=True)
        rows.append(row)
        save(rows, args.out)
# CUT3R aggregates depth metrics weighted by the number of valid pixels
for c in rows[0]["configs"]:
    w = np.array([r["configs"][c]["depth"]["valid_pixels"] for r in rows], float)
    s = {k: np.average([r["configs"][c]["depth"][k] for r in rows], weights=w) for k in ("AbsRel_S", "AbsRel_SS")}
    sec = np.mean([r["configs"][c]["depth"]["seconds"] for r in rows])
    line = f"{c:10s} AbsRel S {s['AbsRel_S']:.4f} SS {s['AbsRel_SS']:.4f}  {sec:.2f}s"
    if "pointcloud" in rows[0]["configs"][c]:
        line += f"  point-cloud L1 {np.mean([r['configs'][c]['pointcloud']['L1'] for r in rows]):.4f}"
    print(line)
