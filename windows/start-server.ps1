# Start vLLM (vllm-for-rtx-spark build) with torch.compile on and the fused FP8 kernel.
# Run patch_vllm_fp8_fused.py once first.
$env:MBX_DEPTH_LIB = "C:\rtx-spark\mbx-depth-disabled"
$env:VLLM_DISABLED_KERNELS = "B12xTensorFP8ScaledMMLinearKernel,B12xFp8BlockScaledMMKernel"
$env:VLLM_USE_BREAKABLE_CUDAGRAPH = "0"
$env:VLLM_RTX_FP8_FUSED = "1"
$env:VLLM_DISABLE_COMPILE_CACHE = "1"
$env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:HF_HUB_OFFLINE = "1"; $env:VLLM_NO_USAGE_STATS = "1"; $env:DO_NOT_TRACK = "1"
# Edit these two paths for your install and model folder.
$Vllm  = "C:\rtx-spark\.venv\Scripts\vllm.exe"
$Model = "D:\models\gemma-4-12B-it-FP8-dynamic"
& $Vllm serve $Model --config (Join-Path $PSScriptRoot "gemma4-12b-fp8dyn.yaml")
