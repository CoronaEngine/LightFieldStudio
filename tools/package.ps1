param(
    [Parameter(Mandatory)][string]$RuntimeDirectory,
    [Parameter(Mandatory)][string]$CudaToolkitDirectory,
    [Parameter(Mandatory)][string]$MSVCRuntimeDirectory,
    [Parameter(Mandatory)][string]$RuntimeLicenseDirectory,
    [switch]$IncludeAuthorizedOptixSdk,
    [string]$OutputDirectory = (Join-Path $env:LOCALAPPDATA 'LightFieldStudio/packages'),
    [string]$Version = '0.1.0'
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$source = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runtime = [IO.Path]::GetFullPath($RuntimeDirectory)
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
if ($Version -notmatch '^\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?$') { throw 'Version must be a semantic version such as 0.1.0.' }
$packageName = "LightFieldStudio-v$Version-win-x64"
$zip = Join-Path $OutputDirectory "$packageName.zip"
if (Test-Path -LiteralPath $zip) { throw "Refusing to overwrite an existing package: $zip" }
$stage = Join-Path $OutputDirectory ($packageName + '-' + [Guid]::NewGuid().ToString('N'))
$package = Join-Path $stage 'LightFieldStudio'
New-Item -ItemType Directory -Force -Path $package | Out-Null
function Copy-Required([string]$From, [string]$To) {
    if (!(Test-Path -LiteralPath $From -PathType Leaf)) { throw "Required package input is missing: $From" }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $To) | Out-Null
    Copy-Item -LiteralPath $From -Destination $To
}
Copy-Required (Join-Path $runtime 'LightFieldStudio.exe') (Join-Path $package 'LightFieldStudio.exe')
foreach ($file in Get-ChildItem -LiteralPath $runtime -File -Filter '*.dll') {
    if ($file.Name -match '^(vision-|ocarina|EASTL|xatlas)') { Copy-Required $file.FullName (Join-Path $package $file.Name) }
}
foreach ($name in @('lfs_early_measurement','lfs_encoded_atrous_native','lfs_encoded_atrous_plain','lfs_encoded_atrous_glossy','lfs_encoded_carrier_composer')) {
    Copy-Required (Join-Path $runtime "$name.fatbin") (Join-Path $package "$name.fatbin")
}
foreach ($pattern in @('nvrtc64_*.dll','nvrtc-builtins64_*.dll','cudart64_*.dll')) {
    $files = @(Get-ChildItem -LiteralPath (Join-Path $CudaToolkitDirectory 'bin/x64') -Filter $pattern -File)
    if (!$files.Count) { $files = @(Get-ChildItem -LiteralPath (Join-Path $CudaToolkitDirectory 'bin') -Filter $pattern -File) }
    if ($files.Count -ne 1) { throw "Expected one CUDA runtime matching $pattern" }
    Copy-Required $files[0].FullName (Join-Path $package $files[0].Name)
}
foreach ($file in Get-ChildItem -LiteralPath $MSVCRuntimeDirectory -Filter '*.dll' -File) {
    Copy-Required $file.FullName (Join-Path $package $file.Name)
}
if ((Test-Path -LiteralPath (Join-Path $runtime 'cuda/optix')) -and !$IncludeAuthorizedOptixSdk) {
    throw 'Including OptiX SDK source requires separate redistribution authorization. Supply -IncludeAuthorizedOptixSdk only when that authorization covers this package.'
}
Copy-Item -LiteralPath (Join-Path $runtime 'cuda') -Destination (Join-Path $package 'cuda') -Recurse
# The NVRTC include closure, not the full CUDA Toolkit.
foreach ($header in @('cuda_fp16.h','cuda_fp16.hpp','vector_types.h','crt/host_defines.h')) {
    Copy-Required (Join-Path $CudaToolkitDirectory "include/$header") (Join-Path $package "cuda/include/$header")
}
Copy-Item -LiteralPath (Join-Path $CudaToolkitDirectory 'include/nv') -Destination (Join-Path $package 'cuda/include/nv') -Recurse
$manifest = Get-Content -LiteralPath (Join-Path $source 'assets/scenes.json') -Raw | ConvertFrom-Json
foreach ($entry in $manifest.files) {
    # Package only the versioned scene closure, never a developer's asset folder.
    Copy-Required (Join-Path $runtime "assets/$($entry.path)") (Join-Path $package "assets/$($entry.path)")
    if ((Get-FileHash -LiteralPath (Join-Path $package "assets/$($entry.path)") -Algorithm SHA256).Hash -ne $entry.sha256) { throw "Asset checksum mismatch: $($entry.path)" }
}
Copy-Item -LiteralPath (Join-Path $source 'licenses') -Destination (Join-Path $package 'licenses') -Recurse
foreach ($name in @('CUDA.txt','OptiX.txt','Microsoft-Visual-Cpp.txt')) {
    Copy-Required (Join-Path $RuntimeLicenseDirectory $name) (Join-Path $package "licenses/$name")
}
Copy-Required (Join-Path $source 'LICENSE') (Join-Path $package 'LICENSE')
Copy-Required (Join-Path $source 'THIRD_PARTY_NOTICES.md') (Join-Path $package 'THIRD_PARTY_NOTICES.md')
Copy-Required (Join-Path $source 'docs/GETTING_STARTED.md') (Join-Path $package 'GETTING_STARTED.md')
Copy-Required (Join-Path $source 'docs/CONTROLS.md') (Join-Path $package 'CONTROLS.md')
Copy-Required (Join-Path $source 'assets/scenes.json') (Join-Path $package 'assets/scenes.json')
$studioCommit = (git -C $source rev-parse HEAD)
if ($LASTEXITCODE -ne 0) { throw 'Cannot identify the Studio source commit.' }
$visionCommit = (git -C (Join-Path $source 'backend/Vision') rev-parse HEAD)
if ($LASTEXITCODE -ne 0) { throw 'Cannot identify the Vision source commit.' }
@{ version=$Version; platform='windows-x64'; studio_commit=$studioCommit; vision_commit=$visionCommit; asset_revision=$manifest.revision; asset_sha256=$manifest.sha256 } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $package 'version.json') -Encoding UTF8
$files = Get-ChildItem -LiteralPath $package -Recurse -File | Sort-Object FullName
$checksums = foreach ($file in $files) { (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant() + '  ' + $file.FullName.Substring($package.Length+1).Replace('\','/') }
[IO.File]::WriteAllLines((Join-Path $package 'FILES.sha256'),$checksums)
[IO.Compression.ZipFile]::CreateFromDirectory($stage,$zip,[IO.Compression.CompressionLevel]::Optimal,$false)
(Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant() + "  $packageName.zip" | Set-Content -LiteralPath "$zip.sha256" -Encoding ASCII
Write-Host "Package: $zip"
Write-Host "Staged files: $package"
