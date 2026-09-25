# Install a CUDA-enabled PyTorch build for the development GPU.
# Run from PowerShell in the project directory.
$ErrorActionPreference = "Stop"
$indexUrl = "https://download.pytorch.org/whl/cu128"
python -m pip install --upgrade --force-reinstall torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url $indexUrl
python -c "import torch; print(torch.__version__); print('cuda=', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
