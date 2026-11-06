[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BuildDirectory,
    [Parameter(Mandatory)][string]$Destination,
    [Parameter(Mandatory)][string]$AudioPrefix,
    [Parameter(Mandatory)][string]$CudaPath,
    [Parameter(Mandatory)][string]$ExpectedArchitectures,
    [string]$CMakeExecutable = 'cmake'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($env:OS -ne 'Windows_NT') { throw 'Windows packaging requires Windows.' }
$build = (Resolve-Path -LiteralPath $BuildDirectory).Path
$audio = (Resolve-Path -LiteralPath $AudioPrefix).Path
$cuda = (Resolve-Path -LiteralPath $CudaPath).Path
$destinationPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Destination)
# Never reuse a directory that could contain stale DLLs from another build.
if (Test-Path -LiteralPath $destinationPath) { throw 'Destination must not already exist.' }
$inspector = (Get-Command dumpbin.exe -ErrorAction Stop).Source
$cuobjdump = Join-Path $cuda 'bin/cuobjdump.exe'
if (-not (Test-Path -LiteralPath $cuobjdump)) { throw 'CUDA cuobjdump is required.' }
if (-not $env:VCToolsRedistDir) { throw 'Run packaging in an MSVC developer environment.' }
$crt = @(Get-ChildItem -LiteralPath (Join-Path $env:VCToolsRedistDir 'x64') -Directory |
    Where-Object { $_.Name -match '^Microsoft\.VC\d+\.CRT$' })
if ($crt.Count -ne 1) { throw 'Cannot identify the x64 app-local VC++ runtime.' }

$install = Join-Path $destinationPath 'install-staging'
& $CMakeExecutable --install $build --config Release --prefix $install
if ($LASTEXITCODE) { throw "CMake installation failed: $LASTEXITCODE" }
Get-ChildItem -LiteralPath (Join-Path $install 'bin') -File |
    Where-Object { $_.Extension -in @('.exe', '.dll') } |
    Copy-Item -Destination $destinationPath
Get-ChildItem -LiteralPath $crt[0].FullName -Filter '*.dll' -File |
    Copy-Item -Destination $destinationPath

$licenses = Join-Path $destinationPath 'licenses'
New-Item -ItemType Directory -Path $licenses | Out-Null
Copy-Item -LiteralPath (Join-Path $install 'share/licenses/tts.cpp') -Destination $licenses -Recurse
Copy-Item -LiteralPath (Join-Path $cuda 'EULA.txt') -Destination (Join-Path $licenses 'NVIDIA-CUDA-EULA.txt')
# vcpkg carries license notices for codecs and any transitive audio dependencies.
$copyrights = @(Get-ChildItem -LiteralPath (Join-Path $audio 'share') -Filter copyright -Recurse -File)
if (-not $copyrights.Count) { throw 'Audio dependency license notices are missing.' }
foreach ($notice in $copyrights) {
    $folder = Join-Path $licenses "audio/$($notice.Directory.Name)"
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
    Copy-Item -LiteralPath $notice.FullName -Destination $folder
}
# Only this script-created staging directory is removed, after checking its boundary.
$staging = (Resolve-Path -LiteralPath $install).Path
if (-not $staging.StartsWith($destinationPath.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Install staging escaped the package directory.'
}
Remove-Item -LiteralPath $staging -Recurse -Force

# Check every packaged DLL, including imports that are not direct dependencies of the EXEs.
$system = [Environment]::GetFolderPath('System')
foreach ($binary in Get-ChildItem -LiteralPath $destinationPath -File |
    Where-Object { $_.Extension -in @('.exe', '.dll') }) {
    $imports = & $inspector /nologo /dependents $binary.FullName
    if ($LASTEXITCODE) { throw "Cannot inspect $($binary.Name)" }
    foreach ($line in $imports) {
        if ($line -notmatch '^\s+([\w.+-]+\.dll)\s*$') { continue }
        $dependency = $Matches[1]
        if ($dependency -match '^(api-ms-|ext-ms-)') { continue }
        if (Test-Path -LiteralPath (Join-Path $destinationPath $dependency)) { continue }
        # nvcuda.dll belongs to the installed NVIDIA driver, never to the archive.
        if ($dependency -ieq 'nvcuda.dll') { continue }
        if ($dependency -notmatch '^(vcruntime|msvcp|concrt)' -and
            (Test-Path -LiteralPath (Join-Path $system $dependency))) { continue }
        throw "Missing package dependency: $($binary.Name) -> $dependency"
    }
}
foreach ($program in @('tts-server.exe', 'tts.exe')) {
    if (-not (Test-Path -LiteralPath (Join-Path $destinationPath $program))) { throw "Missing $program" }
}
$library = Join-Path $destinationPath 'tts.dll'
$cubins = & $cuobjdump --list-elf $library
if ($LASTEXITCODE) { throw 'Cannot inspect packaged CUDA kernels.' }
$ptx = & $cuobjdump --list-ptx $library
if ($LASTEXITCODE) { throw 'Cannot inspect packaged PTX.' }
foreach ($architecture in $ExpectedArchitectures.Split(';')) {
    if ($architecture -notmatch '^(\d+)(-(real|virtual))?$') { throw "Invalid architecture: $architecture" }
    $number, $kind = $Matches[1], $Matches[3]
    if ($kind -ne 'virtual' -and -not ($cubins -match "\.sm_$number\.")) {
        throw "Missing SM$number CUDA kernels"
    }
    if ($kind -ne 'real' -and -not ($ptx -match "\.sm_$number\.")) {
        throw "Missing SM$number PTX"
    }
}
@'
@echo off
setlocal
"%~dp0tts-server.exe" --quantization group_int8 %*
exit /b %errorlevel%
'@ | Set-Content -LiteralPath (Join-Path $destinationPath 'start-server.cmd') -Encoding ascii
@'
Windows x64, NVIDIA RTX 30 / 40 / 50 series (default workflow architectures).

Extract the entire archive. Keep the DLLs beside the executables.
CUDA Toolkit, Python, Rust and build tools are not required to run this package.
A compatible NVIDIA driver and a separately downloaded model are required.
Model: https://huggingface.co/BreezeBlue/Breeze-TTS-2

Run from a terminal, supplying the actual model directory:
  start-server.cmd --model "MODEL_DIRECTORY"
  tts-server.exe --help

The launcher defaults to group_int8. Additional server options pass through.
The web page is embedded in the server. Its address appears in the server log.
To enable named voice persistence, supply --voice-presets-file "PRESETS_JSON".
Generated web audio remains in memory. Reference audio is not saved in presets.

CI checks DLL imports and startup on a separate runner without a CUDA SDK in PATH.
GitHub-hosted Windows runners do not validate actual GPU speech synthesis.
'@ | Set-Content -LiteralPath (Join-Path $destinationPath 'USAGE.txt') -Encoding ascii
Write-Output "Windows runtime package ready: $destinationPath"
