# GPU development

The development machine has an NVIDIA RTX 3050 Laptop GPU with 6 GB VRAM
(driver 610.62, CUDA driver 13.3).

## Verified state

| Component | Version | Notes |
| --- | --- | --- |
| `torch` | `2.14.0+cu126` | `torch.cuda.is_available() == True` |
| `onnxruntime-gpu` | `1.26.0` | built against CUDA 12.8 |
| `nvidia-cudnn-cu12` | `9.26.0.51` | cuDNN 9, required by ORT 1.21+ |
| `nvidia-cublas-cu12` | `12.9.2.10` | |
| `nvidia-cuda-nvrtc-cu12` | `12.9.86` | pulled in as a cuBLAS dependency |

Both the PyTorch and the ONNX Runtime paths are GPU-capable. Note that the
earlier assumption that PyTorch was a `+cpu` build was wrong; the CUDA DLLs that
`preload_dlls()` loads are the ones bundled under `torch/lib`.

## ONNX Runtime path

```powershell
# Resumable parallel download + offline install + verification.
# Plain `pip install` of these wheels is unusably slow from this machine
# (~50 KB/s single stream, ~1.6 GB total).
python scripts/setup_onnx_cuda.py
```

The script pins `onnxruntime-gpu==1.26.0` on purpose. From ONNX Runtime 1.27
onward the PyPI GPU wheels are built with **CUDA 13**, but NVIDIA ships no
`nvidia-cublas-cu13` / `nvidia-cuda-runtime-cu13` wheels for Windows, so a
CUDA 13 build cannot be satisfied here. ORT 1.21.x-1.26.x are the **CUDA 12.8
+ cuDNN 9** build line and pair correctly with the `nvidia-*-cu12` wheels.

To verify an existing environment by hand:

```powershell
python scripts/onnx_gpu_smoke.py --image data/inspection/frame_5000.jpg
```

The smoke command must print and exit `0`:

```text
available_providers: [... 'CUDAExecutionProvider' ...]
session_providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
{'status': 'PASS', 'device': 'CUDAExecutionProvider', 'device_id': 0, ...}
```

If `session_providers` starts with `CPUExecutionProvider`, stop and fix the
DLL path before processing any full video. The script exits non-zero in that
case so it cannot be mistaken for a successful run.

### DLL loading

`src/perception/onnx_detector.py::_configure_cuda_dlls` does two things before
any session is created:

1. Adds every `site-packages/nvidia/*/bin` directory to the DLL search path.
   This is what silences `Could not locate nvrtc64_120_0.dll`; cuBLAS probes for
   an optional JIT path that is not otherwise on the loader path.
2. Calls `onnxruntime.preload_dlls()`, which resolves the CUDA/cuDNN/MSVC
   runtime dependencies (available since ORT 1.21).

Both must run before `InferenceSession`. Note that
`site.getsitepackages()[0]` is the interpreter prefix, *not*
`Lib/site-packages`, so every entry has to be searched.

## PyTorch path

For Ultralytics training/inference, verify with:

```powershell
python scripts/check_runtime.py
python scripts/gpu_smoke.py
```

Both paths must report CUDA before a full-video development run.
