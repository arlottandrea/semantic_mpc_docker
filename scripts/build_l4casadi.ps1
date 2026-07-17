$ErrorActionPreference = "Stop"

$cudaRoot = (Resolve-Path ".pixi/envs/default/Library").Path.Replace("\", "/")
$env:CUDA_PATH = $cudaRoot
$env:CUDAToolkit_ROOT = $cudaRoot
$env:CUDACXX = "$cudaRoot/bin/nvcc.exe"
$env:CMAKE_CUDA_ARCHITECTURES = "86"

python -m pip install ./l4casadi --no-build-isolation --no-deps --force-reinstall
