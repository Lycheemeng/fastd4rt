from __future__ import annotations

import torch
import torch.nn.functional as F


def anchor_subset(frames: list[int], r: int) -> list[int]:
    if r <= 1 or len(frames) <= 2:
        return list(frames)
    keep = list(frames[::r])
    if keep[-1] != frames[-1]:
        keep.append(frames[-1])
    return keep


def interp_fill(arr: torch.Tensor, have: list[int], T: int) -> None:
    h = torch.as_tensor(have, device=arr.device)
    t = torch.arange(h[0], h[-1] + 1, device=arr.device)
    hi = torch.searchsorted(h, t).clamp(1, len(h) - 1)
    lo = hi - 1
    w = ((t - h[lo]).float() / (h[hi] - h[lo]).float())[None, :, None]
    arr[:, t] = arr[:, h[lo]] * (1 - w) + arr[:, h[hi]] * w


def grid(G: int, device="cuda"):
    c = (torch.arange(G, device=device, dtype=torch.float32) + .5) / G
    v, u = torch.meshgrid(c, c, indexing="ij")
    return u.reshape(-1), v.reshape(-1)


def upsample(x: torch.Tensor, hw) -> torch.Tensor:
    return F.interpolate(x.permute(0, 3, 1, 2), size=hw, mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
