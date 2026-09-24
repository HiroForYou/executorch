# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Builds the Windows CUDA wheel for one CUDA train, installs it, and runs its smoke test
# against a program a Linux job exported for that train. This is the route a Windows user
# has: lowering for CUDA cannot run on Windows, so the program comes from Linux and only
# the wheel's C++ SDK runs it.
param(
    [Parameter(Mandatory = $true)][string]$Train,
    [Parameter(Mandatory = $true)][string]$Artifacts
)
$ErrorActionPreference = "Stop"

# The runner's powershell.exe is Windows PowerShell 5.1, where a native command that
# fails does not stop the script ($PSNativeCommandUseErrorActionPreference is 7.3+ only).
# Without this a failing test printed its traceback and the job still passed.
function Invoke-Native {
    param([Parameter(Mandatory = $true)][scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "exit code ${LASTEXITCODE}: $Command"
    }
}

if (-not (Test-Path (Join-Path $Artifacts "model.pte"))) {
    throw "no model.pte in $Artifacts; the Linux export job did not hand over a program"
}

# The GPU runners come without a driver; the same step PyTorch's Windows CUDA tests run.
& "$PSScriptRoot\..\install_nvidia_driver_windows.ps1"

# The runner image carries some toolkits; any other train is assembled from NVIDIA's
# redistributable archives so each row compiles with its own nvcc.
$installed = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$Train"
if (Test-Path (Join-Path $installed "bin\nvcc.exe")) {
    $cudaHome = $installed
} else {
    $cudaHome = Join-Path $env:RUNNER_TEMP "cuda-$Train"
    Invoke-Native { python .ci/scripts/wheel/install_cuda_redist.py --train $Train --platform windows-x86_64 --dest $cudaHome }
}
$env:CUDA_HOME = $cudaHome
$env:CUDA_PATH = $cudaHome
$env:PATH = "$cudaHome\bin\x64;$cudaHome\bin;$env:PATH"
Remove-Item Env:CUDACXX -ErrorAction SilentlyContinue
Invoke-Native { nvcc --version }

Invoke-Native { conda create --yes --quiet -n et python=3.12 }
conda activate et
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64

# The Python side of this wheel carries no CUDA, and install_requirements installs CPU
# torch on Windows, which is what the wheel build and its import checks need. It also
# installs the torchao nightly the wheel declares, which PyPI does not carry, and with
# --example the matching torchvision the smoke test's model run needs, as the release
# build's pre-build hook does.
Invoke-Native { python install_requirements.py --example }
Invoke-Native { pip install ninja }

# The release row's identity: its +cuXYZ tag and its GPU list, from executorch_cuda_arch_list,
# called the way the release build's pre-build hook calls it. The train goes on the command
# line and the script is read relative to the directory bash starts in, without carriage
# returns, which works for Git's bash and for WSL's launcher, on any checkout's line endings.
$cu = "cu" + $Train.Replace(".", "")
$env:BUILD_VERSION = "$(Get-Content version.txt)+$cu"
Push-Location .ci/scripts/wheel
# Git's bash where installed: System32's bash.exe is WSL's launcher, which can come first on
# PATH and fails on a machine with no WSL distribution.
$gitBash = Join-Path $env:ProgramFiles "Git\bin\bash.exe"
$bash = if (Test-Path $gitBash) { $gitBash } else { "bash" }
$archList = & $bash -c "CU_VERSION=$cu; source <(tr -d '\r' < ./cuda_arch_list.sh) && executorch_cuda_arch_list"
$archExit = $LASTEXITCODE
Pop-Location
if ($archExit -ne 0 -or -not "$archList".Trim()) { throw "cuda_arch_list.sh gives no GPU list for $cu" }
$env:TORCH_CUDA_ARCH_LIST = "$archList".Trim()
$env:CMAKE_ARGS = "-DEXECUTORCH_BUILD_CUDA=ON -DEXECUTORCH_BUILD_VULKAN=OFF"
$env:DISTUTILS_USE_SDK = "1"
Invoke-Native { python setup.py bdist_wheel }
$wheel = Get-ChildItem dist\*.whl | Select-Object -First 1
# With its declared dependencies, as a user installs it: the smoke test's clean-install
# check fails on any module the wheel needs and does not declare.
Invoke-Native { pip install $wheel.FullName }

$env:EXECUTORCH_CUDA_WINDOWS_ARTIFACTS = (Resolve-Path $Artifacts).Path
$env:PYTHONIOENCODING = "utf-8"
# From outside the checkout, so the checks inspect the installed wheel, not the sources.
$scripts = (Resolve-Path .ci/scripts/wheel).Path
Push-Location $env:RUNNER_TEMP
try {
    Invoke-Native { python (Join-Path $scripts "test_cuda_windows.py") }
} finally {
    Pop-Location
}
