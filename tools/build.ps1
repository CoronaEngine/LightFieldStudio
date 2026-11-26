param(
    [string]$BuildDirectory = (Join-Path $env:LOCALAPPDATA 'LightFieldStudio/build-v0.1.0'),
    [string]$AssetDirectory = (Join-Path $PSScriptRoot '../assets'),
    [string]$AssetArchive,
    [string]$CMake = 'cmake',
    [int]$Parallel = 4
)
$ErrorActionPreference = 'Stop'
$source = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$BuildDirectory = [IO.Path]::GetFullPath($BuildDirectory)
if (!(Test-Path -LiteralPath (Join-Path $source 'backend/Vision/CMakeLists.txt'))) { throw 'Initialize submodules: git submodule update --init --recursive' }
if (!(Get-Command $CMake -ErrorAction SilentlyContinue)) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (Test-Path -LiteralPath $vswhere) {
        $found = & $vswhere -all -products '*' -find 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
        if ($found) { $CMake = @($found)[0] }
    }
}
if (!(Get-Command $CMake -ErrorAction SilentlyContinue)) { throw 'Install CMake 4.0 or newer, or pass -CMake with its executable path.' }
& (Join-Path $PSScriptRoot 'download-assets.ps1') -Destination $AssetDirectory -Archive $AssetArchive
function Invoke-CMake([string[]]$Arguments) {
    # Windows PowerShell can turn redirected native stderr warnings into errors.
    # Retain the output and use the actual process exit code for build failures.
    $ErrorActionPreference = 'Continue'
    & $CMake @Arguments
    if ($LASTEXITCODE) { throw "CMake failed with exit code $LASTEXITCODE." }
}
Invoke-CMake -Arguments @('-S', $source, '-B', $BuildDirectory, '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_POLICY_VERSION_MINIMUM=3.5')
Invoke-CMake -Arguments @('--build', $BuildDirectory, '--config', 'Release', '--target', 'LightFieldStudio', '--parallel', "$Parallel")
$runtime = Join-Path $BuildDirectory 'bin/Release'
New-Item -ItemType Directory -Force -Path (Join-Path $runtime 'assets') | Out-Null
$manifest = Get-Content -LiteralPath (Join-Path $source 'assets/scenes.json') -Raw | ConvertFrom-Json
foreach ($entry in $manifest.files) {
    $destination = Join-Path $runtime "assets/$($entry.path)"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
    Copy-Item -LiteralPath (Join-Path $AssetDirectory $entry.path) -Destination $destination -Force
}
Write-Host "Built: $(Join-Path $runtime 'LightFieldStudio.exe')"
