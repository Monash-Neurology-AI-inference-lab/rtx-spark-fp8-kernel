"""How close does vLLM's Triton attention get to the GPU's memory bandwidth?

Times vLLM's own unified_attention at Gemma 4 12B shapes with an FP8 KV cache
laid out as the Triton backend lays it out ([blocks, kv_heads, block, 2*head]),
for ~1,000 sequences generating one token each, plus a mixed batch like the
profiled step (994 generating + 6 prompts of 88 tokens). Reports the KV bytes
each call must read and the bandwidth achieved, against a plain device copy.

Usage: C:\\rtx-spark\\.venv\\Scripts\\python.exe attn_bench.py
Random tensors only.
"""
import os
from itertools import accumulate

os.environ.setdefault("MBX_DEPTH_LIB", r"C:\rtx-spark\mbx-depth-disabled")
os.environ.setdefault("VLLM_DISABLED_KERNELS",
                      "B12xTensorFP8ScaledMMLinearKernel,B12xFp8BlockScaledMMKernel")

import torch  # noqa: E402

from vllm.v1.attention.ops.triton_unified_attention import unified_attention  # noqa: E402
from vllm.v1.kv_cache_interface import KVQuantMode  # noqa: E402

dev = "cuda"
FP8 = torch.float8_e4m3fn
# (name, q_heads, kv_heads, head_size, sliding window or None, layers in the model)
LAYERS = [("sliding", 16, 8, 256, 1024, 40), ("full", 16, 1, 512, None, 8)]
N_DECODE = 994


def timed(fn, iters=10):
    fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def copy_bandwidth():
    x = torch.empty(2 * 1024**3, dtype=torch.uint8, device=dev)
    y = torch.empty_like(x)
    ms = timed(lambda: y.copy_(x))
    del x, y
    return 2 * 2 * 1024**3 / (ms / 1000) / 1e9  # read + write


def run(q_heads, kv_heads, head, window, ctx, block, prefill=0, prefill_len=88):
    """ctx: context length of every generating sequence (including its new token)."""
    n_seqs = N_DECODE + prefill
    q_lens = [1] * N_DECODE + [prefill_len] * prefill
    k_lens = [ctx] * N_DECODE + [ctx] * prefill
    blocks_per_seq = (ctx + block - 1) // block
    n_blocks = n_seqs * blocks_per_seq
    kv = torch.empty(n_blocks, kv_heads, block, 2 * head, dtype=torch.uint8, device=dev)
    kv.view(torch.int8).random_(-100, 100)  # finite FP8 values
    kv = kv.view(FP8).transpose(1, 2)
    key_cache, value_cache = kv.split(head, dim=-1)
    block_table = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(n_seqs, blocks_per_seq)
    n_tok = sum(q_lens)
    q = torch.randn(n_tok, q_heads, head, dtype=torch.bfloat16, device=dev)
    out = torch.empty_like(q)
    cu = torch.tensor([0, *accumulate(q_lens)], dtype=torch.int32, device=dev)
    seqused = torch.tensor(k_lens, dtype=torch.int32, device=dev)
    one = torch.ones(1, dtype=torch.float32, device=dev)
    descale = one.expand(n_seqs, kv_heads)
    win = (window - 1, 0) if window else (-1, -1)

    def call():
        unified_attention(q=q, k=key_cache, v=value_cache, out=out, cu_seqlens_q=cu,
                          max_seqlen_q=max(q_lens), seqused_k=seqused, max_seqlen_k=ctx,
                          softmax_scale=head ** -0.5, causal=True, window_size=win,
                          block_table=block_table, softcap=0, q_descale=None,
                          k_descale=descale, v_descale=descale,
                          kv_quant_mode=KVQuantMode.FP8_PER_TENSOR)

    ms = timed(call)
    span = min(ctx, window) if window else ctx
    kv_bytes = n_seqs * span * kv_heads * head * 2  # K and V, 1 byte each
    del kv, key_cache, value_cache, q, out
    torch.cuda.empty_cache()
    return ms, kv_bytes / 1e9


if __name__ == "__main__":
    print(torch.__version__, torch.cuda.get_device_name())
    peak = copy_bandwidth()
    print(f"device copy bandwidth: {peak:.0f} GB/s\n")
    for prefill in (0, 6):
        print("generating only" if not prefill else "mixed: 994 generating + 6 prompts x 88 tokens")
        for name, qh, kvh, hd, win, n_layers in LAYERS:
            for ctx in (512, 1024, 2048):
                for block in ((16, 64) if win else (64,)):
                    ms, gb = run(qh, kvh, hd, win, ctx, block, prefill)
                    bw = gb / (ms / 1000)
                    print(f"  {name:7s} ctx {ctx:5d} block {block:3d}: {ms:7.2f} ms/layer  "
                          f"x{n_layers} = {ms * n_layers:7.1f} ms  KV {gb:5.2f} GB  "
                          f"{bw:6.0f} GB/s ({bw / peak * 100:3.0f}% of copy)", flush=True)
        print()
