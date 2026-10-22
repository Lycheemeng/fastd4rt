"""Named configurations used in the paper.

Each config is (engine, temporal anchor stride r for tracks, grid size G for dense outputs).
``d4rt`` is the FP32 reference; ``ours`` is the FastD4RT default (BF16 + compile, r = 3, G = 128).
Camera pose always uses a 32 x 32 grid for both methods, as in the community configuration.
"""
CONFIGS = {
    "d4rt":     ("fp32", 1, 256),
    "ours":     ("bf16c", 3, 128),
    # ablations
    "bf16":     ("bf16", 1, 256),
    "bf16c":    ("bf16c", 1, 256),
    "bf16c_r2": ("bf16c", 2, 256),
    "bf16c_g64": ("bf16c", 1, 64),
    "ours_fp8": ("fp8c", 3, 128),
}
