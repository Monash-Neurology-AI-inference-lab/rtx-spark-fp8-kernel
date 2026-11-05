"""FP8 per-channel matmul on Windows: current path vs two cheaper variants.
Random tensors only; reads config.json for the layer shapes.
  cw      vLLM ChannelWiseTorch: fp32 GEMM output, then separate scaling
  cw+cmp  the same code under torch.compile (what a compile run would do)
  bf16    GEMM straight to bf16, then scale in place (no fp32 copy)
"""
import json
import sys
from pathlib import Path

import torch

cfg = json.loads((Path(sys.argv[1]) / "config.json").read_text())
cfg = cfg.get("text_config", cfg)
H, I = cfg["hidden_size"], cfg["intermediate_size"]
shapes = {"gate_up": (H, 2 * I), "down": (I, H), "o_proj": (H, H)}
dev = "cuda"
ONE = torch.ones(1, dtype=torch.float32, device=dev)
print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability())


def quant(x, dim):
    s = x.abs().amax(dim=dim, keepdim=True).float().clamp(min=1e-12) / 448.0
    return (x / s).to(torch.float8_e4m3fn), s


def cw(A, B, As, Bs):
    out = torch._scaled_mm(A, B, scale_a=ONE, scale_b=ONE, out_dtype=torch.float32)
    return (out * As * Bs.t()).to(torch.bfloat16)


cw_cmp = torch.compile(cw, dynamic=False)


def bf16(A, B, As, Bs):
    out = torch._scaled_mm(A, B, scale_a=ONE, scale_b=ONE, out_dtype=torch.bfloat16)
    return out.mul_(As).mul_(Bs.t())


def timed(fn, *args, iters=20):
    for _ in range(3):
        fn(*args)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


torch.manual_seed(0)
for name, (k, n) in shapes.items():
    W = torch.randn(n, k, device=dev, dtype=torch.bfloat16)
    Wq, Bs = quant(W, dim=1)
    B = Wq.t()
    for m in (512, 8192, 32768):
        X = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        A, As = quant(X, dim=1)
        ref = (A.float() * As) @ (B.float() * Bs.t())
        line = f"{name:8s} m={m:6d}"
        for label, fn in (("cw", cw), ("cw+cmp", cw_cmp), ("bf16", bf16)):
            try:
                o = fn(A, B, As, Bs)
                e = ((o.float() - ref).abs().max() / ref.abs().max()).item()
                line += f"  {label} {timed(fn, A, B, As, Bs):7.2f} ms ({e:.1e})"
            except Exception as ex:
                line += f"  {label} FAILED {type(ex).__name__}: {str(ex)[:80]}"
        print(line, flush=True)
        del X, A, ref
        torch.cuda.empty_cache()
