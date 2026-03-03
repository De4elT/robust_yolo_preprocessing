param(
    [string]$ZipPath = "$(Resolve-Path (Join-Path $PSScriptRoot '..\coco128.zip') -ErrorAction SilentlyContinue)",
    [string]$OutDir = "$(Join-Path (Join-Path $PSScriptRoot '..') 'data\coco\coco128')"
)

$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host "[setup_coco128] $msg" }

if (-not $ZipPath -or -not (Test-Path $ZipPath)) {
    Write-Host "Nie znaleziono coco128.zip. Umieść plik w root projektu (obok setup.py) albo podaj -ZipPath." -ForegroundColor Red
    exit 1
}

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$tmpDir = Join-Path $projectRoot 'data\coco\_tmp_extract'

Write-Info "Projekt: $projectRoot"
Write-Info "Zip: $ZipPath"
Write-Info "Out: $OutDir"

if (Test-Path $tmpDir) { Remove-Item -Recurse -Force $tmpDir }
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

Write-Info "Rozpakowuję archiwum..."
Expand-Archive -Path $ZipPath -DestinationPath $tmpDir -Force

# Znajdź katalog, który zawiera 'images' i 'labels'
$candidate = Get-ChildItem -Path $tmpDir -Recurse -Directory | Where-Object {
    (Test-Path (Join-Path $_.FullName 'images')) -and (Test-Path (Join-Path $_.FullName 'labels'))
} | Select-Object -First 1

if (-not $candidate) {
    Write-Host "Nie znalazłem folderu z images/ i labels/ w rozpakowanym zipie. Sprawdź zawartość coco128.zip." -ForegroundColor Red
    exit 2
}

Write-Info "Znaleziono dataset w: $($candidate.FullName)"

# Przygotuj OutDir
if (Test-Path $OutDir) {
    Write-Info "OutDir już istnieje, usuwam..."
    Remove-Item -Recurse -Force $OutDir
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Info "Kopiuję images/ i labels/ do OutDir..."
Copy-Item -Recurse -Force (Join-Path $candidate.FullName 'images') $OutDir
Copy-Item -Recurse -Force (Join-Path $candidate.FullName 'labels') $OutDir

# Sprzątanie
Remove-Item -Recurse -Force $tmpDir

Write-Info "OK. Sprawdź czy masz:"
Write-Host "  $OutDir\images\train2017" 
Write-Host "  $OutDir\labels\train2017" 
Write-Info "Następnie uruchom: scripts\smoke_test.ps1"
