# Install a CUDA-enabled PyTorch build for the development GPU.
# Run from PowerShell in the project directory.
$ErrorActionPreference = "Stop"
# Verified on this machine (RTX 3050 6GB Laptop, sm_86, driver 610.62):
# torch 2.14.0 and torchvision 0.29.0 are the versions already present as
# +cpu wheels, so matching them keeps the rest of the environment unchanged.
# 2.14.0 is only published for cu126; the newer driver is backward compatible.
$indexUrl = "https://download.pytorch.org/whl/cu126"
$torchVersion = "2.14.0+cu126"
$torchvisionVersion = "0.29.0+cu126"
python -m pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12
python -m pip install --upgrade --force-reinstall "torch==$torchVersion" "torchvision==$torchvisionVersion" --index-url $indexUrl
python -c "import torch; print(torch.__version__); print('cuda=', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
# torchvision must be the +cu126 build too: a leftover +cpu torchvision imports
# fine but has no CUDA kernels, and Ultralytics NMS then dies at predict time
# with "Could not run 'torchvision::nms'" instead of failing here.
python -c "import torch, torchvision; from torchvision.ops import nms; b=torch.rand(8,4,device='cuda'); b[:,2:]+=b[:,:2]; s=torch.rand(8,device='cuda'); nms(b,s,0.5); print('torchvision', torchvision.__version__, 'cuda nms OK')"
