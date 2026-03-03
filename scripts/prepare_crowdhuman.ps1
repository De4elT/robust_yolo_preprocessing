param(
    [string]$RawRoot = "$(Join-Path (Join-Path $PSScriptRoot '..') 'data\crowdhuman\raw')",
    [string]$YoloRoot = "$(Join-Path (Join-Path $PSScriptRoot '..') 'data\crowdhuman\yolo')",
    [string]$TrainImages = "train01\Images",
    [string]$ValImages = "val\Images",
    [string]$TrainOdgt = "annotation_train.odgt",
    [string]$ValOdgt = "annotation_val.odgt"
)

$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host "[prepare_crowdhuman] $msg" }

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$rawTrainPath = Join-Path $RawRoot $TrainImages
$rawValPath   = Join-Path $RawRoot $ValImages
$trainOdgtPath = Join-Path $RawRoot $TrainOdgt
$valOdgtPath   = Join-Path $RawRoot $ValOdgt

$yoloImagesTrain = Join-Path $YoloRoot 'images\train'
$yoloImagesVal   = Join-Path $YoloRoot 'images\val'
$yoloLabelsTrain = Join-Path $YoloRoot 'labels\train'
$yoloLabelsVal   = Join-Path $YoloRoot 'labels\val'

Write-Info "Projekt: $projectRoot"
Write-Info "Raw root: $RawRoot"
Write-Info "YOLO root: $YoloRoot"

# Twórz strukturę
New-Item -ItemType Directory -Force -Path $yoloLabelsTrain | Out-Null
New-Item -ItemType Directory -Force -Path $yoloLabelsVal | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $YoloRoot 'images') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $YoloRoot 'labels') | Out-Null

function Ensure-LowercaseImagesDir([string]$p) {
    if (Test-Path $p) { return $p }
    # spróbuj zamienić ...\Images -> ...\images
    if ($p -match "\\Images$") {
        $alt = ($p -replace "\\Images$", "\\images")
        if (Test-Path $alt) { return $alt }
        $parent = Split-Path $p -Parent
        $imagesUpper = Join-Path $parent 'Images'
        $imagesLower = Join-Path $parent 'images'
        if (Test-Path $imagesUpper -and -not (Test-Path $imagesLower)) {
            Write-Info "Zmieniam nazwę folderu: Images -> images w $parent"
            Rename-Item -Path $imagesUpper -NewName 'images'
            return $imagesLower
        }
    }
    return $p
}

$rawTrainPath = Ensure-LowercaseImagesDir $rawTrainPath
$rawValPath   = Ensure-LowercaseImagesDir $rawValPath

function Try-MakeJunction([string]$linkPath, [string]$targetPath) {
    if (Test-Path $linkPath) { return }
    try {
        New-Item -ItemType Junction -Path $linkPath -Target $targetPath | Out-Null
        Write-Info "Utworzono junction: $linkPath -> $targetPath"
    } catch {
        Write-Host "Nie udało się utworzyć junction (uprawnienia/polityka)." -ForegroundColor Yellow
        Write-Host "Zrób ręcznie: skopiuj/przenieś obrazy do: $linkPath" -ForegroundColor Yellow
        Write-Host "Cel (źródło): $targetPath" -ForegroundColor Yellow
    }
}

# TRAIN
if (Test-Path $rawTrainPath -and (Test-Path $trainOdgtPath)) {
    Write-Info "TRAIN: obrazy: $rawTrainPath"
    Write-Info "TRAIN: odgt : $trainOdgtPath"
    Try-MakeJunction -linkPath $yoloImagesTrain -targetPath $rawTrainPath

    Write-Info "Konwertuję adnotacje TRAIN -> YOLO labels..."
    python (Join-Path $projectRoot 'tools\convert_crowdhuman_odgt_to_yolo.py') --odgt $trainOdgtPath --images_dir $yoloImagesTrain --labels_out $yoloLabelsTrain
} else {
    Write-Host "Pominięto TRAIN: brakuje $rawTrainPath lub $trainOdgtPath" -ForegroundColor Yellow
}

# VAL
if (Test-Path $rawValPath -and (Test-Path $valOdgtPath)) {
    Write-Info "VAL: obrazy: $rawValPath"
    Write-Info "VAL: odgt : $valOdgtPath"
    Try-MakeJunction -linkPath $yoloImagesVal -targetPath $rawValPath

    Write-Info "Konwertuję adnotacje VAL -> YOLO labels..."
    python (Join-Path $projectRoot 'tools\convert_crowdhuman_odgt_to_yolo.py') --odgt $valOdgtPath --images_dir $yoloImagesVal --labels_out $yoloLabelsVal
} else {
    Write-Host "Pominięto VAL: brakuje $rawValPath lub $valOdgtPath" -ForegroundColor Yellow
}

Write-Info "Gotowe. Sprawdź dataset YAML: data\\crowdhuman.yaml"
Write-Info "Następnie odpal: python tools\\check_dataset.py --data data\\crowdhuman.yaml"
