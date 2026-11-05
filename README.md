# Fused FP8 matmul for vLLM on NVIDIA RTX Spark (GB10, sm_121): native Windows

**vLLM on Windows · RTX Spark / GB10 · sm_121 · FP8 · Triton · torch.compile · Gemma 4**

A Triton FP8 matrix-multiply kernel, and a small patch that installs it into vLLM, for the
[vllm-for-rtx-spark](https://github.com/devjeonghwan/vllm-for-rtx-spark) native Windows ARM64
build of vLLM by **devjeonghwan**. That project is what makes vLLM run natively on Windows on
the RTX Spark; this repository only adds a faster FP8 matmul on top of it.

With this kernel and torch.compile switched back on, a batch job with Gemma 4 12B
(`RedHatAI/gemma-4-12B-it-FP8-dynamic`, 1,000 requests at once) ran **about 13% faster than the
same setup under WSL2**, and about 58% faster than the build as shipped.

Full write-up, results and settings:
[GB10-class infrastructure for batch research inference](https://www.drmvalente.com.au/blog-rtx-spark-dgx-spark-batch-inference.html#native-windows-vllm)
(Dr Michael Valente).

## Who this is for

- You run vLLM **natively on Windows** on an **NVIDIA RTX Spark** laptop (GB10-class N1X, compute
  capability **sm_121**) with vllm-for-rtx-spark, and serve an **FP8** model
  (for example `RedHatAI/gemma-4-12B-it-FP8-dynamic`).
- Your logs show vLLM using `ChannelWiseTorchFP8ScaledMMLinearKernel`, or
  `cutlass_scaled_mm_supports_fp8(121)` returns `False`, or row-wise `torch._scaled_mm` fails with
  "Rowwise scaling is not currently supported on your device".
- You serve many requests at once (batch inference), where the FP8 matmuls are a large share of
  GPU time.

**DGX Spark and WSL2/Linux:** the slow fallback was found in the native Windows build. Standard
Linux vLLM builds (including under WSL2) use a fused CUTLASS kernel instead, so the gain there is
likely small. Untested.

## The problem it fixes

On the RTX Spark's GPU (sm_121), the build's CUTLASS FP8 kernel reports no support, so vLLM
falls back to `ChannelWiseTorchFP8ScaledMMLinearKernel`. That path writes the matmul result in
fp32 and then applies the per-token and per-channel scales in separate passes. On this
memory-bandwidth-limited GPU the extra passes cost more than the matmul. PyTorch's row-wise
`_scaled_mm` is also unsupported on sm_121.

`rtx_fp8_fused.py` computes `C = (A_q @ B_q) * scale_a[m] * scale_b[n]` with the scales applied
in the GEMM epilogue, written straight to bf16. It uses grouped tile ordering (as in the
[Triton matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html))
with tile sizes autotuned on the RTX Spark N1X and fixed, so it can be captured in CUDA graphs.
It is registered as `torch.ops.rtxfp8.scaled_mm` so torch.compile treats it as one op.

Milliseconds per call at Gemma 4 12B shapes, 32,768 tokens per step:

| Layer | Stock (unfused) | + torch.compile | **Fused kernel** | cuBLAS per-tensor (ceiling) |
|---|---|---|---|---|
| gate_up (3840→30720) | 133 | 71 | **47.7** | 42.8 |
| down (15360→3840) | 31.7 | 24.3 | **24.3** | 20.7 |
| o_proj (3840→3840) | 17.0 | – | **6.0** | 5.4 |

Maximum relative error is the same as the stock path (2.7e-03): the same maths, rounded once to bf16.

## Install

Requires a working [vllm-for-rtx-spark](https://github.com/devjeonghwan/vllm-for-rtx-spark)
install (tested with v0.1.2: vLLM 0.32.0, PyTorch 2.14.0+cu134, triton-windows 3.8).

```powershell
# from this folder, with the build's Python
C:\rtx-spark\.venv\Scripts\python.exe patch_vllm_fp8_fused.py          # install + self-test
C:\rtx-spark\.venv\Scripts\python.exe patch_vllm_fp8_fused.py --undo   # restore the original
```

The patch copies the kernel into the vLLM package and edits
`ChannelWiseTorchFP8ScaledMMLinearKernel.apply_scaled_mm` to call it **only when
`VLLM_RTX_FP8_FUSED=1`**; otherwise vLLM behaves as before. The original file is kept as
`pytorch.py.bak`. A self-test on random tensors prints `SELF-TEST OK`.

## Run

`windows/start-server.ps1` sets the environment and starts the server with
`windows/gemma4-12b-fp8dyn.yaml` (edit the two paths at the bottom first). The variables that matter:

| Variable | Why |
|---|---|
| `VLLM_USE_BREAKABLE_CUDAGRAPH=0` | The build sets it to 1, which makes vLLM turn torch.compile off. 0 turns compile back on. |
| `VLLM_RTX_FP8_FUSED=1` | Uses the fused kernel. |
| `VLLM_DISABLE_COMPILE_CACHE=1` | The patched file changes the traced code. |
| `MBX_DEPTH_LIB=<a path that does not exist>` | Avoids `KeyError: 'MBX_DEPTH_LIB'` with models other than the build's Qwen recipe. |
| `VLLM_DISABLED_KERNELS=B12xTensorFP8ScaledMMLinearKernel,B12xFp8BlockScaledMMKernel` | Avoids `ModuleNotFoundError: No module named 'cutlass'`. |
| `PYTHONUTF8=1` | Avoids `UnicodeEncodeError` when logging to a file. |

On Windows, also fix the paging file at roughly the size of GPU memory (CUDA allocations count
against system commit), and set GPU memory to its maximum in the NVIDIA app.

## Benchmarks

`benchmarks/` holds the scripts behind the numbers above. All use random tensors and read only
the model's `config.json` for layer shapes.

- `fp8_triton_kernel.py`: the autotuning version of the kernel, against the cuBLAS per-tensor ceiling.
- `fp8_bench2.py`: the stock ChannelWise path, under torch.compile, and a bf16-output variant.
- `attn_bench.py`: vLLM's Triton attention at Gemma 4 12B shapes against memory bandwidth
  (attention is now the largest share of GPU time at 1,000 requests).

## Notes

- Tested only on the RTX Spark N1X (ASUS ProArt P14) with Gemma 4 12B shapes. Other GPUs or
  shapes may want different tiles; re-run `benchmarks/fp8_triton_kernel.py` to autotune.
- Not affiliated with NVIDIA, the vLLM project or vllm-for-rtx-spark. Use at your own risk.

## Related

- [vllm-for-rtx-spark](https://github.com/devjeonghwan/vllm-for-rtx-spark): the native Windows vLLM build this patches.
- [GB10-class infrastructure for batch research inference](https://www.drmvalente.com.au/blog-rtx-spark-dgx-spark-batch-inference.html):
  full benchmarks, native Windows vs WSL2, ASUS ProArt P14 power-profile tuning, prefix-cache
  hit rate and recommended vLLM settings for the RTX Spark and DGX Spark.

## Licence

Apache License 2.0, the same licence as vLLM. See `LICENSE`.
Copyright 2026 Michael Valente.
