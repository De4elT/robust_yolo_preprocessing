# robust_yolo_preprocessing

Projekt do pracy magisterskiej: **poprawa odporności YOLO na degradacje obrazu** przez porównanie:
- baseline YOLOv7
- klasyczna filtracja (offline, powtarzalna)
- trening z augmentacją zakłóceniami (offline train_aug)
- **Learnable Preprocessing Block (LPB)** uczony end-to-end z detektorem

Repo zawiera narzędzia do budowania *offline* wariantów zbioru (clean/corrupt/filter) oraz skrypt do benchmarku mAP i FPS.

> **Uwaga:** w repo nie trzymamy danych ani wag modeli. Wszystko co ciężkie trafia do `data/` i `weights/` i jest ignorowane przez `.gitignore`.

---

## Szybki start (Smoke test na coco128 – 10–30 min)

### 0) Wymagania
- Python 3.10+ (zalecane)
- PyTorch + CUDA (zgodnie z Twoją kartą/sterownikami)
- Pakiety narzędziowe: OpenCV, PyYAML, pandas

### 1) Instalacja zależności (minimalne)
W katalogu projektu:

```bash
pip install -r tools/requirements_tools.txt
```

PyTorch instalujesz zgodnie z oficjalną instrukcją dla swojej wersji CUDA.

### 2) Przygotuj coco128
1. Pobierz `coco128.zip` (masz lokalnie) i wrzuć go do **root projektu**.
2. Uruchom:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_coco128.ps1
```

Po tym powinieneś mieć strukturę:

```
data/coco/coco128/
  images/train2017/...
  labels/train2017/...
```

### 3) Przygotuj wagi (baseline)
Wgraj plik wag YOLOv7 do:

```
weights/yolov7.pt
```

(Pliku nie commitujemy.)

### 4) Odpal smoke test

```powershell
powershell -ExecutionPolicy Bypass -File scripts\smoke_test.ps1
```

Wyniki znajdziesz w:

```
runs/smoke/
  results_long.csv
  summary_by_variant.csv
  summary_by_corruption.csv
  env_benchmark.json
```

---

## CrowdHuman (docelowy zbiór)

Instrukcje krok po kroku znajdziesz w `data/README.md` oraz w skrypcie `scripts/prepare_crowdhuman.ps1`.
W skrócie:
1) pobierasz CrowdHuman zgodnie z licencją,
2) doprowadzasz do struktury z folderem `images/` (ważne: małe litery),
3) konwertujesz adnotacje `.odgt` do YOLO (`tools/convert_crowdhuman_odgt_to_yolo.py`),
4) sprawdzasz kompletność (`tools/check_dataset.py`),
5) budujesz warianty i benchmarkujesz jak w `tools/README.md`.

---

## Najważniejsze komendy (benchmark)
Zobacz `tools/README.md`.
