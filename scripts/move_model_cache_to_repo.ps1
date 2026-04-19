param(
    [string]$RepoPath = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if (-not $RepoPath) {
    $RepoPath = (Resolve-Path (Join-Path $PSScriptRoot ".." )).Path
}

function Get-DirSizeSummary {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return "missing"
    }
    $m = Get-ChildItem $Path -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum
    return "files=$($m.Count) bytes=$($m.Sum)"
}

function Move-DirContent {
    param(
        [string]$Source,
        [string]$Destination,
        [bool]$DryRunMode
    )

    if (-not (Test-Path $Source)) {
        Write-Host "SKIP missing: $Source"
        return
    }

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null

    if ($DryRunMode) {
        Write-Host "DRYRUN move: $Source -> $Destination"
        return
    }

    Write-Host "MOVE: $Source -> $Destination"
    robocopy $Source $Destination /E /MOVE /R:2 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
    $code = $LASTEXITCODE
    if ($code -gt 7) {
        throw "Robocopy failed with exit code $code for $Source"
    }

    if (Test-Path $Source) {
        $left = Get-ChildItem -Force $Source -ErrorAction SilentlyContinue
        if (-not $left) {
            Remove-Item $Source -Force -ErrorAction SilentlyContinue
        }
    }

    Write-Host "OK robocopy code=$code"
}

$repo = (Resolve-Path $RepoPath).Path
$destBase = Join-Path $repo ".model_store"
$destHf = Join-Path $destBase "huggingface"
$destTts = Join-Path $destBase "tts"

$srcHf = Join-Path $HOME ".cache\huggingface"
$srcTts = Join-Path $HOME "AppData\Local\tts"

New-Item -ItemType Directory -Force -Path $destBase | Out-Null
New-Item -ItemType Directory -Force -Path $destHf | Out-Null
New-Item -ItemType Directory -Force -Path $destTts | Out-Null

Move-DirContent -Source $srcHf -Destination $destHf -DryRunMode:$DryRun
Move-DirContent -Source $srcTts -Destination $destTts -DryRunMode:$DryRun

Write-Host "DEST HF: $(Get-DirSizeSummary -Path $destHf)"
Write-Host "DEST TTS: $(Get-DirSizeSummary -Path $destTts)"
Write-Host "SRC HF:  $(Get-DirSizeSummary -Path $srcHf)"
Write-Host "SRC TTS: $(Get-DirSizeSummary -Path $srcTts)"
