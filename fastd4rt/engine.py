"""Execution engines for a frozen D4RT model.

``fp32``  : the reference path (original modules, FP32, keys/values recomputed per query batch).
``bf16``  : efficient execution path: bf16 autocast, fused SDPA attention reusing the original projection weights,
            decoder keys/values computed once per segment.
``bf16c`` : ``bf16`` + ``torch.compile`` of the encoder (the FastD4RT default).
``fp8c``  : ``bf16c`` with W8A8 FP8 linear layers in the encoder (ablation only).
None of the engines changes the network or its weights.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import sinusoidal_position_embedding

CLIP_LEN = 48


# ---------------------------------------------------------------- attention
def mha_sdpa(attn: torch.nn.MultiheadAttention, q_in, kv_in=None, k=None, v=None):
    """nn.MultiheadAttention (batch_first, eval) computed with scaled_dot_product_attention."""
    d, h = attn.embed_dim, attn.num_heads
    w, b = attn.in_proj_weight, attn.in_proj_bias
    bsz, nq, _ = q_in.shape
    q = F.linear(q_in, w[:d], b[:d]).view(bsz, nq, h, d // h).transpose(1, 2)
    if k is None:
        nk = kv_in.shape[1]
        kv = F.linear(kv_in, w[d:], b[d:]).view(bsz, nk, 2, h, d // h)
        k, v = kv.unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v)
    return attn.out_proj(out.transpose(1, 2).reshape(bsz, nq, d))


def _block(blk, x):
    q = blk.norm_attn(x)
    x = x + mha_sdpa(blk.attn, q, q)
    return x + blk.ff(blk.norm_ff(x))


def encode_fast(model, video, aspect, blocks=None):
    """Same function as ``model.encode_video`` with fused attention; optional replacement blocks (FP8)."""
    enc = model.encoder
    extra = model._project_aspect_ratio_token(video=video, aspect_ratio=aspect)
    x = enc._token_cap(enc.patch_embed(video.permute(0, 2, 1, 3, 4)))
    b, c, tp, hp, wp = x.shape
    tokens = x.flatten(2).transpose(1, 2)
    n = tokens.shape[1]
    tokens = tokens + sinusoidal_position_embedding(n, enc.hidden_dim, tokens.device).unsqueeze(0)
    s = hp * wp
    blocks = blocks or [None] * len(enc.blocks)
    for mode, blk, rep in zip(enc.block_modes, enc.blocks, blocks):
        run = rep if rep is not None else (lambda z, blk=blk: _block(blk, z))
        if mode == "local":
            tokens = run(tokens.reshape(b * tp, s, c)).reshape(b, tp * s, c)
        else:
            h = run(torch.cat([tokens, extra.to(tokens.dtype)], 1))
            tokens, extra = h[:, :n], h[:, n:]
    return model.memory_proj(enc.final_norm(torch.cat([tokens, extra], 1)))


def decoder_kv(model, memory):
    """Keys and values of the scene representation for every decoder block, computed once."""
    kvs = []
    for blk in model.decoder.blocks:
        attn = blk.cross_attn
        d, h = attn.embed_dim, attn.num_heads
        kv = F.linear(blk.norm_kv(memory), attn.in_proj_weight[d:], attn.in_proj_bias[d:])
        kv = kv.view(memory.shape[0], -1, 2, h, d // h)
        k, v = kv.unbind(2)
        kvs.append((k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()))
    return kvs


# ---------------------------------------------------------------- FP8 (ablation)
E4M3_MAX = 448.0


def _quant_rows(x):
    amax = x.abs().amax(-1, keepdim=True).float().clamp(min=1e-12)
    scale = amax / E4M3_MAX
    return (x.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), scale


class FP8Linear(torch.nn.Module):
    """W8A8 FP8 matmul with per-output-channel weight scales and per-token activation scales."""

    def __init__(self, weight, bias):
        super().__init__()
        w8, ws = _quant_rows(weight.detach())
        self.register_buffer("w8t", w8.t())
        self.register_buffer("ws", ws.t().contiguous())
        self.bias = None if bias is None else bias.detach().to(torch.bfloat16)
        self.out_features = weight.shape[0]

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        m = x2.shape[0]
        pad = (-m) % 16
        if pad:
            x2 = F.pad(x2, (0, 0, 0, pad))
        x8, xs = _quant_rows(x2)
        y = torch._scaled_mm(x8, self.w8t, scale_a=xs, scale_b=self.ws, bias=self.bias, out_dtype=torch.bfloat16)
        return y[:m].reshape(*shp[:-1], self.out_features)


class FP8Block(torch.nn.Module):
    def __init__(self, blk):
        super().__init__()
        a = blk.attn
        self.blk, self.d, self.h = blk, a.embed_dim, a.num_heads
        self.qkv = FP8Linear(a.in_proj_weight, a.in_proj_bias)
        self.proj = FP8Linear(a.out_proj.weight, a.out_proj.bias)
        self.fc1 = FP8Linear(blk.ff[0].weight, blk.ff[0].bias)
        self.fc2 = FP8Linear(blk.ff[3].weight, blk.ff[3].bias)

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = self.qkv(self.blk.norm_attn(x)).view(b, n, 3, self.h, self.d // self.h).permute(2, 0, 3, 1, 4).unbind(0)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, self.d))
        return x + self.fc2(F.gelu(self.fc1(self.blk.norm_ff(x))))


# ---------------------------------------------------------------- engine
class Engine:
    """Encode clips and decode queries with one of the execution paths above."""

    def __init__(self, model, kind: str = "bf16c"):
        assert kind in ("fp32", "bf16", "bf16c", "fp8c"), kind
        self.model, self.kind = model, kind
        if kind == "fp8c":
            fb = [FP8Block(b) for b in model.encoder.blocks]
            self._enc = torch.compile(lambda v, a: encode_fast(model, v, a, fb), mode="max-autotune-no-cudagraphs")
        elif kind == "bf16c":
            self._enc = torch.compile(lambda v, a: encode_fast(model, v, a), mode="max-autotune-no-cudagraphs")
        elif kind == "bf16":
            self._enc = lambda v, a: encode_fast(model, v, a)

    def autocast(self):
        if self.kind == "fp32":
            return torch.autocast("cuda", enabled=False)
        return torch.autocast("cuda", dtype=torch.bfloat16)

    def encode(self, clip, aspect):
        """Encode a clip [1, T, 3, H, W] (end-padded to the clip length). Returns an opaque segment state."""
        if clip.shape[1] < CLIP_LEN:
            clip = torch.cat([clip, clip[:, -1:].expand(-1, CLIP_LEN - clip.shape[1], -1, -1, -1)], 1)
        clip = clip.clone(memory_format=torch.contiguous_format)   # fixed strides: no recompilation
        with self.autocast():
            if self.kind == "fp32":
                return clip, self.model.encode_video(video=clip, aspect_ratio=aspect), None
            mem = self._enc(clip, aspect)
            return clip, mem, decoder_kv(self.model, mem)

    def decode(self, seg, u, v, t_src, t_tgt, t_cam):
        """Decode flat query vectors (torch, equal length). Returns float32 xyz [n,3], uv [n,2], vis [n], conf [n]."""
        clip, mem, kvs = seg
        m = self.model
        q = dict(u=u[None], v=v[None], t_src=t_src[None], t_tgt=t_tgt[None], t_cam=t_cam[None])
        step = 4096 if kvs is None else 16384   # FP32: query batch of the community code
        outs = []
        with self.autocast():
            for s0 in range(0, len(u), step):
                sub = {k: x[:, s0:s0 + step] for k, x in q.items()}
                if kvs is None:
                    o = m.decode_queries(video=clip, query=sub, memory=mem)
                else:
                    x = m.query_embedder(video=clip, u=sub["u"].to(clip.dtype), v=sub["v"].to(clip.dtype),
                                         t_src=sub["t_src"], t_tgt=sub["t_tgt"], t_cam=sub["t_cam"])
                    for blk, (kk, vv) in zip(m.decoder.blocks, kvs):
                        x = x + mha_sdpa(blk.cross_attn, blk.norm_q(x), k=kk, v=vv)
                        x = x + blk.ff(blk.norm_ff(x))
                    o = m.heads(m.decoder.out_norm(x))
                outs.append(dict(xyz=o["xyz_3d"][0].float(), uv=o["uv_2d"][0].float(),
                                 vis=o["visibility"][0].float().sigmoid(), conf=o["confidence"][0].float().sigmoid()))
        return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}
