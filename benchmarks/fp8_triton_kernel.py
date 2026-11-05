"""FP8 matmul with per-token x per-channel scales fused into the epilogue.

C[m, n] = (A_q[m, :] @ B_q[:, n]) * scale_a[m] * scale_b[n], written as bf16.
A_q is (M, K) row-major FP8; B_q is (K, N) column-major FP8 (vLLM's layout).
Grouped tile order keeps weight tiles in L2 across neighbouring row blocks.

Usage: python fp8_triton_kernel.py D:\\models\\gemma-4-12B-it-FP8-dynamic
Random tensors only; reads config.json for the layer shapes.
"""
import json
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

CONFIGS = [
    triton.Config({"BM": bm, "BN": bn, "BK": bk, "GROUP_M": 8}, num_warps=w, num_stages=s)
    for bm, bn, bk, w, s in [
        (128, 128, 128, 8, 3), (128, 128, 64, 4, 4), (128, 128, 64, 8, 4),
        (128, 256, 64, 8, 3), (256, 128, 64, 8, 3), (64, 128, 128, 4, 4),
        (64, 64, 128, 4, 4), (128, 64, 128, 4, 4),
    ]
]


@triton.autotune(configs=CONFIGS, key=["M_BUCKET", "N", "K"])
@triton.jit
def _fp8_scaled_mm(a_ptr, b_ptr, sa_ptr, sb_ptr, c_ptr, M, N, K, M_BUCKET,
                   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group = GROUP_M * num_n
    first_m = (pid // group) * GROUP_M
    size_m = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % size_m
    pid_n = (pid % group) // size_m

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = a_ptr + rm[:, None].to(tl.int64) * stride_am + rk[None, :] * stride_ak
    b = b_ptr + rk[:, None] * stride_bk + rn[None, :].to(tl.int64) * stride_bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        k_left = K - k * BK
        a_t = tl.load(a, mask=(rm[:, None] < M) & (rk[None, :] < k_left), other=0.0)
        b_t = tl.load(b, mask=(rk[:, None] < k_left) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a_t, b_t, acc)
        a += BK * stride_ak
        b += BK * stride_bk

    sa = tl.load(sa_ptr + rm, mask=rm < M, other=0.0)
    sb = tl.load(sb_ptr + rn, mask=rn < N, other=0.0)
    out = (acc * sa[:, None] * sb[None, :]).to(c_ptr.dtype.element_ty)
    c = c_ptr + rm[:, None].to(tl.int64) * stride_cm + rn[None, :] * stride_cn
    tl.store(c, out, mask=(rm[:, None] < M) & (rn[None, :] < N))


def fp8_scaled_mm(A, B, As, Bs, out_dtype=torch.bfloat16):
    """A (M,K) fp8, B (K,N) fp8, As (M,1) or (M,), Bs (N,1) or (N,) float32."""
    M, K = A.shape
    N = B.shape[1]
    C = torch.empty((M, N), dtype=out_dtype, device=A.device)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)
    _fp8_scaled_mm[grid](A, B, As.reshape(-1), Bs.reshape(-1), C, M, N, K,
                         triton.next_power_of_2(M),
                         A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                         C.stride(0), C.stride(1))
    return C


if __name__ == "__main__":
    cfg = json.loads((Path(sys.argv[1]) / "config.json").read_text())
    cfg = cfg.get("text_config", cfg)
    H, I = cfg["hidden_size"], cfg["intermediate_size"]
    shapes = {"gate_up": (H, 2 * I), "down": (I, H), "o_proj": (H, H)}
    dev = "cuda"
    ONE = torch.ones(1, dtype=torch.float32, device=dev)
    print(torch.__version__, triton.__version__, torch.cuda.get_device_name())

    def quant(x):
        s = x.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12) / 448.0
        return (x / s).to(torch.float8_e4m3fn), s

    def cublas_per_tensor(A, B, As, Bs):  # speed ceiling: same GEMM, no per-row scales
        return torch._scaled_mm(A, B, scale_a=ONE, scale_b=ONE, out_dtype=torch.bfloat16)

    def timed(fn, *args, iters=20):
        for _ in range(3):
            fn(*args)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        for _ in range(iters):
            fn(*args)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    torch.manual_seed(0)
    for name, (k, n) in shapes.items():
        Wq, Bs = quant(torch.randn(n, k, device=dev, dtype=torch.bfloat16))
        B = Wq.t()
        for m in (512, 8192, 32768):
            A, As = quant(torch.randn(m, k, device=dev, dtype=torch.bfloat16))
            ref = (A.float() * As) @ (B.float() * Bs.t())
            out = fp8_scaled_mm(A, B, As, Bs)
            err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
            t_new = timed(fp8_scaled_mm, A, B, As, Bs)
            t_cap = timed(cublas_per_tensor, A, B, As, Bs)
            print(f"{name:8s} m={m:6d}  triton-fused {t_new:7.2f} ms ({err:.1e})  "
                  f"cuBLAS ceiling {t_cap:7.2f} ms  best={_fp8_scaled_mm.best_config}", flush=True)
            del A, ref, out
            torch.cuda.empty_cache()
