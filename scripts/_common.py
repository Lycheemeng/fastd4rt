import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastd4rt.configs import CONFIGS  # noqa: E402
from fastd4rt.engine import Engine  # noqa: E402
from fastd4rt.model import load_model  # noqa: E402

SPLITS = Path(__file__).resolve().parents[1] / "splits"


def parser(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--ckpt", required=True, help="OpenD4RT checkpoint dir (model.yaml + opend4rt.ckpt), 48-frame model")
    ap.add_argument("--configs", default="d4rt,ours", help="comma-separated names from fastd4rt/configs.py")
    ap.add_argument("--out", type=Path, required=True)
    return ap


def setup(args):
    torch.backends.cuda.matmul.allow_tf32 = False   # the reference runs in plain FP32, as the community code
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(args.ckpt)
    cfgs = {c: CONFIGS[c] for c in args.configs.split(",")}
    engines = {e: Engine(model, e) for e in {c[0] for c in cfgs.values()}}
    args.out.mkdir(parents=True, exist_ok=True)
    return model, cfgs, engines


def save(rows, out):
    (Path(out) / "per_video.json").write_text(json.dumps(rows, indent=1))


def summarize(rows, task_keys, ref="d4rt"):
    """Video-mean metrics; speed-up = mean reference time / mean time."""
    print(f"\n{len(rows)} videos")
    for task, keys in task_keys.items():
        if task not in rows[0]["configs"][next(iter(rows[0]["configs"]))]:
            continue
        base = np.mean([r["configs"][ref][task]["seconds"] for r in rows]) if ref in rows[0]["configs"] else None
        print(f"[{task}]")
        for c in rows[0]["configs"]:
            m = {k: np.mean([r["configs"][c][task][k] for r in rows]) for k in keys + ["seconds"]}
            sp = f"{base / m['seconds']:6.2f}x" if base else ""
            print(f"  {c:10s} " + " ".join(f"{k} {m[k]:.4f}" for k in keys) + f"  {m['seconds']:.2f}s {sp}")
