param([string]$Destination = (Join-Path $PSScriptRoot '../assets'), [string]$Archive)
$ErrorActionPreference = 'Stop'
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot '../assets/scenes.json') -Raw | ConvertFrom-Json
$Destination = [IO.Path]::GetFullPath($Destination)
$valid = $true
foreach ($entry in $manifest.files) {
    $path = Join-Path $Destination $entry.path
    if (!(Test-Path -LiteralPath $path -PathType Leaf) -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $entry.sha256) { $valid = $false; break }
}
if ($valid) { Write-Host 'Scene assets are ready.'; return }
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$temporary = Join-Path ([IO.Path]::GetTempPath()) ('lfs-scenes-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporary | Out-Null
try {
    $zip = Join-Path $temporary $manifest.filename
    if ($Archive) { Copy-Item -LiteralPath $Archive -Destination $zip }
    else {
        $url = "https://huggingface.co/datasets/$($manifest.repository)/resolve/$($manifest.revision)/$($manifest.filename)"
        Write-Host 'Downloading scene assets...'
        Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    }
    if ((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash -ne $manifest.sha256) { throw 'Scene archive checksum mismatch.' }
    $expanded = Join-Path $temporary 'expanded'
    Expand-Archive -LiteralPath $zip -DestinationPath $expanded
    foreach ($entry in $manifest.files) {
        $source = Join-Path $expanded $entry.path
        if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne $entry.sha256) { throw "Scene file checksum mismatch: $($entry.path)" }
    }
    foreach ($entry in $manifest.files) {
        $target = Join-Path $Destination $entry.path
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
        Copy-Item -LiteralPath (Join-Path $expanded $entry.path) -Destination $target -Force
    }
    Write-Host 'Scene assets are ready.'
} finally {
    $resolved = [IO.Path]::GetFullPath($temporary)
    $root = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    if (![String]::Equals([IO.Path]::GetDirectoryName($resolved), $root.TrimEnd('\','/'), [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetFileName($resolved) -notmatch '^lfs-scenes-[a-f0-9]{32}$') { throw 'Unexpected temporary directory.' }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
