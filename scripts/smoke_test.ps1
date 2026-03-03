param(
    [string]$DataYaml = "$(Join-Path (Join-Path $PSScriptRoot '..') 'data\coco128.yaml')",
    [string]$WeightsBaseline = "$(Join-Path (Join-Path $PSScriptRoot '..') 'weights\yolov7.pt')",
    [string]$OutRoot = "$(Join-Path (Join-Path $PSScriptRoot '..') 'runs\smoke')",
    [string]$Device = "0",
    [int]$Img = 640,
    [int]$Batch = 8
)

$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host "[smoke_test] $msg" }

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')

Write-Info "Projekt: $projectRoot"
Write-Info "Data yaml: $DataYaml"
Write-Info "Weights baseline: $WeightsBaseline"
Write-Info "Out: $OutRoot"

if (-not (Test-Path $DataYaml)) {
    Write-Host "Brak pliku data yaml: $DataYaml" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $WeightsBaseline)) {
    Write-Host "Brak wag baseline: $WeightsBaseline" -ForegroundColor Red
    Write-Host "Umieść plik YOLOv7 w weights\\yolov7.pt (nie commituj)." -ForegroundColor Yellow
    exit 2
}

# 1) Sprawdź dataset
Write-Info "Sprawdzam dataset..."
python (Join-Path $projectRoot 'tools\check_dataset.py') --data $DataYaml

# 2) Zbuduj warianty (mało, żeby szybko poszło)
Write-Info "Buduję warianty danych (offline corruptions + filter)..."
python (Join-Path $projectRoot 'tools\robust_dataset_builder.py') make-datasets --data $DataYaml --out $OutRoot --splits val `
  --corruptions noise jpeg --severities 2 --filters median3

# 3) Benchmark
Write-Info "Odpalam benchmark (mAP + FPS)..."
python (Join-Path $projectRoot 'tools\bench_yolov7_variants.py') --data $DataYaml --variants_root $OutRoot --split val `
  --weights_baseline $WeightsBaseline --filters median3 --corruptions noise jpeg --severities 2 --img $Img --batch $Batch --device $Device

Write-Info "DONE. Sprawdź pliki:"
Write-Host "  $OutRoot\results_long.csv"
Write-Host "  $OutRoot\summary_by_variant.csv"
Write-Host "  $OutRoot\summary_by_corruption.csv"
Write-Host "  $OutRoot\env_benchmark.json"
