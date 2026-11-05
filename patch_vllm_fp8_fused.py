"""Install the fused FP8 matmul (rtx_fp8_fused.py) into the native Windows vLLM.

    C:\\rtx-spark\\.venv\\Scripts\\python.exe patch_vllm_fp8_fused.py          # install + self-test
    C:\\rtx-spark\\.venv\\Scripts\\python.exe patch_vllm_fp8_fused.py --undo   # restore the original

Copies rtx_fp8_fused.py into the vllm package as vllm/_rtx_fp8_fused.py and edits
ChannelWiseTorchFP8ScaledMMLinearKernel.apply_scaled_mm (scaled_mm/pytorch.py) to
call it when VLLM_RTX_FP8_FUSED=1. Without that variable vLLM behaves as before.
The original pytorch.py is kept as pytorch.py.bak.
"""
import importlib.util
import shutil
import sys
from pathlib import Path

MARK = "RTX_FP8_FUSED patch"
HEADER = f"""import torch
import os as _rtx_os
_RTX_FP8_FUSED = _rtx_os.environ.get("VLLM_RTX_FP8_FUSED") == "1"  # {MARK}
if _RTX_FP8_FUSED:
    import vllm._rtx_fp8_fused  # noqa: F401  registers torch.ops.rtxfp8.scaled_mm
"""
ANCHOR = "        # Use unfused DQ due to limitations with scaled_mm\n"
FUSED = f"""        if _RTX_FP8_FUSED:  # {MARK}: scales applied inside the GEMM
            out = torch.ops.rtxfp8.scaled_mm(A, B, As, Bs, out_dtype)
            out = torch.narrow(out, 0, 0, _get_num_tokens(output_shape))
            if bias is not None:
                out = out + bias
            return out.view(*output_shape)

"""

vllm_dir = Path(importlib.util.find_spec("vllm").submodule_search_locations[0])
target = vllm_dir / "model_executor" / "kernels" / "linear" / "scaled_mm" / "pytorch.py"
backup = target.with_name("pytorch.py.bak")
module = vllm_dir / "_rtx_fp8_fused.py"

if "--undo" in sys.argv:
    if backup.exists():
        shutil.copy2(backup, target)
        print(f"restored {target}")
    module.unlink(missing_ok=True)
    print("fused FP8 patch removed")
    sys.exit(0)

text = target.read_text(encoding="utf-8")
if MARK in text:
    print(f"already patched: {target}")
else:
    if text.count(ANCHOR) != 1 or "import torch\n" not in text or "_get_num_tokens" not in text:
        sys.exit(f"unexpected contents in {target}; not patched")
    if not backup.exists():
        shutil.copy2(target, backup)
    text = text.replace("import torch\n", HEADER, 1).replace(ANCHOR, FUSED + ANCHOR, 1)
    target.write_text(text, encoding="utf-8")
    print(f"patched {target} (original kept as {backup.name})")
shutil.copy2(Path(__file__).with_name("rtx_fp8_fused.py"), module)
print(f"installed {module}")

# Self-test on random tensors, loading the module file directly (no vLLM start-up).
import torch  # noqa: E402

spec = importlib.util.spec_from_file_location("rtx_fp8_fused_test", module)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
torch.manual_seed(0)
worst = 0.0
for m, k, n in ((7, 3840, 30720), (100, 3840, 4096), (4096, 15360, 3840)):
    A = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    B = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn).t()
    As = torch.rand(m, 1, device="cuda") + 0.5
    Bs = torch.rand(n, 1, device="cuda") + 0.5
    ref = (A.float() @ B.float()) * As * Bs.t()
    out = torch.ops.rtxfp8.scaled_mm(A, B, As, Bs, torch.bfloat16)
    err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
    worst = max(worst, err)
    print(f"  m={m:5d} k={k:5d} n={n:5d}  rel.err {err:.1e}")
print("SELF-TEST OK" if worst < 1e-2 else "SELF-TEST FAILED: run with --undo")
