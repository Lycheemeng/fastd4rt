"""Umeyama alignment (Sim(3) or rigid) for torch and numpy."""
from __future__ import annotations

import numpy as np
import torch


def umeyama(src: torch.Tensor, dst: torch.Tensor, with_scale: bool = True):
    """(s, R, t) minimising ||s R src + t - dst||^2 for point sets [n, 3] (Umeyama, 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = torch.linalg.svd(cov)
    D = torch.eye(3, dtype=src.dtype, device=src.device)
    if torch.det(U) * torch.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = (S * D.diagonal()).sum() / (xs ** 2).sum(1).mean() if with_scale else torch.ones((), dtype=src.dtype, device=src.device)
    return s, R, mu_d - s * R @ mu_s


def umeyama_np(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (xs ** 2).sum(1).mean() if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def compose(a, b):
    """Sim(3) composition a o b, each given as (s, R, t)."""
    sa, Ra, ta = a
    sb, Rb, tb = b
    return sa * sb, Ra @ Rb, sa * (Ra @ tb) + ta
