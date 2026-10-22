"""3D point tracking on TAPVid-3D (paper Tab. 4): camera AJ/APD3D/OA and world APD3D/L1, full-length videos."""
from pathlib import Path
import json

from _common import SPLITS, parser, save, setup, summarize
import torch
from fastd4rt import metrics
from fastd4rt.data import load_tapvid3d
from fastd4rt.pipeline import track_anchored, track_segments

WORLD = ("adt", "drivetrack")

ap = parser(__doc__)
ap.add_argument("--data", type=Path, required=True, help="directory of TAPVid-3D .npz files of one subset")
ap.add_argument("--subset", choices=["adt", "drivetrack", "pstudio"], required=True)
ap.add_argument("--anchored", action="store_true", help="use anchored windows (m=8) instead of chained segments")
ap.add_argument("--max-frames", type=int, default=0, help="truncate videos (long-video study)")
ap.add_argument("--offset", type=int, default=0)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
model, cfgs, engines = setup(args)
names = json.loads((SPLITS / "adt_39.json").read_text())["videos"] if args.subset == "adt" else \
    sorted(p.stem for p in args.data.glob("*.npz"))
names = names[args.offset:][: args.limit or None]
tasks = ["camera", "world"] if args.subset in WORLD else ["camera"]
rows, warmed = [], set()
with torch.inference_mode():
    for i, name in enumerate(names):
        d = load_tapvid3d(args.data / f"{name}.npz", args.max_frames or None)
        row = dict(name=name, T=d["T"], queries=len(d["tq"]), configs={})
        for cname, (eng_name, r, _) in cfgs.items():
            run = (lambda: track_anchored(engines[eng_name], d, tasks, 8, r)) if args.anchored else \
                  (lambda: track_segments(engines[eng_name], d, tasks, r))
            if eng_name not in warmed:          # torch.compile warm-up, not timed
                run()
                warmed.add(eng_name)
            pred, secs, info = run()
            res = dict(info, camera=dict(seconds=secs["camera"], **metrics.tracking_camera(d["gt_cam"], d["vis"], pred["camera"], d["intr"])))
            if "world" in pred:
                res["world"] = dict(seconds=secs["world"], **metrics.tracking_world(d["gt_world"], pred["world"]))
            row["configs"][cname] = res
            print(f"[{i+1}/{len(names)}] {name[:32]:32s} T={d['T']} {cname:8s} cam {secs['camera']:.2f}s "
                  + " ".join(f"{k} {v:.4f}" for k, v in res["camera"].items() if k != "seconds"), flush=True)
        rows.append(row)
        save(rows, args.out)
summarize(rows, {"camera": ["AJ", "APD3D", "OA"], "world": ["APD", "L1"]})
