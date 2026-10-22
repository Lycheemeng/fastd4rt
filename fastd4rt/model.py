"""Loading a D4RT checkpoint.

FastD4RT does not ship model code or weights. It uses the model definition and checkpoints of the community
OpenD4RT release, which must be installed separately. Point ``OPEND4RT_ROOT`` to its repository root.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


def _opend4rt_root() -> Path:
    root = os.environ.get("OPEND4RT_ROOT")
    if not root:
        raise RuntimeError("Set OPEND4RT_ROOT to the root of the OpenD4RT repository.")
    root = Path(root).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _state_dict(payload):
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "module", "network", "net"):
            if isinstance(payload.get(key), dict):
                payload = payload[key]
                break
    out = {}
    for k, v in payload.items():
        for prefix in ("module.", "model."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        out[k] = v
    return out


def load_model(ckpt_dir: str | Path, device: str = "cuda") -> torch.nn.Module:
    """Build the D4RT model from ``ckpt_dir/model.yaml`` and load ``ckpt_dir/opend4rt.ckpt`` (frozen, eval mode)."""
    _opend4rt_root()
    from src.core import load_checkpoint, load_yaml_config, seed_everything
    from src.model import build_model

    ckpt_dir = Path(ckpt_dir)
    cfg = load_yaml_config(ckpt_dir / "model.yaml")
    seed_everything(int(cfg.get_path("experiment.seed", 42)), deterministic=True)
    model = build_model(cfg["model"]).eval()
    model.load_state_dict(_state_dict(load_checkpoint(ckpt_dir / "opend4rt.ckpt", map_location="cpu")), strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def sinusoidal_position_embedding(length: int, dim: int, device) -> torch.Tensor:
    """Position embedding used by the D4RT encoder (imported from the model code)."""
    _opend4rt_root()
    from src.model.utils import sinusoidal_position_embedding as spe
    return spe(length, dim, device)
