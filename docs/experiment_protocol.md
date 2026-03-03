# Protokół eksperymentów (wersja robocza)

## Cel
Ocena odporności detektora YOLOv7 na degradacje obrazu oraz porównanie podejść:
1) **Baseline** (standardowy YOLOv7)
2) **Klasyczna filtracja** (statyczne filtry, warianty offline)
3) **Augmentacja zakłóceniami** (offline train_aug → osobne wagi)
4) **Learnable Preprocessing Block (LPB)** (blok uczony end-to-end z detektorem)

## Zbiory danych
- **Docelowo:** CrowdHuman (detekcja osób)
- **Na etapie „smoke test”/walidacji pipeline’u:** coco128 (mały zbiór do szybkiej weryfikacji)

## Degradacje (offline)
Warianty degradacji generowane narzędziem `tools/robust_dataset_builder.py`:
- `noise` (szum Gaussa)
- `gblur` (rozmycie Gaussa)
- `jpeg` (kompresja JPEG)
- `lowlight` (przyciemnienie / gamma)

Poziomy (severity): `1, 2, 3`.

## Filtry klasyczne (offline)
Warianty filtrów (przykłady):
- `median3` / `median5`
- `bilateral`
- `clahe`
- `unsharp`

## Matryca porównań
Dla każdego zbioru testowego (clean + każda degradacja/severity + degradacja+filtr):

### Modele / warianty wag
- **baseline**: `weights_baseline` na wszystkich wariantach danych
- **static_filter**: nadal `weights_baseline`, ale dane są *offline* przefiltrowane (warianty `__<filter>`)
- **augmented** (opcjonalnie): `weights_aug` trenowane na `train_aug`
- **learnable_preproc**: `weights_preproc` (YOLO z LPB w architekturze)

## Metryki
- mAP@0.5 (mAP50)
- mAP@0.5:0.95 (mAP50-95)
- Precision / Recall (dla kontekstu)
- **ΔmAP**: spadek względem danych czystych (`clean`) na tym samym modelu
- **Wydajność (FPS)**:
  - baseline/aug/LPB: FPS z `test.py` (ms inference/NMS/total)
  - static_filter: do `total_ms` dodawany jest średni czas filtra (pomiar offline na próbie obrazów) → *pipeline FPS*

## Zasady powtarzalności i uczciwości
- Stałe: rozmiar wejścia (`img`), batch, conf/iou, device
- Warm-up GPU przed pomiarem
- Raportowanie środowiska (GPU/CPU/OS/wersje) do `env_benchmark.json`

## Artefakty wyników
Generowane przez `tools/bench_yolov7_variants.py`:
- `results_long.csv` (pełny log)
- `summary_by_variant.csv` (podsumowania per model/filter)
- `summary_by_corruption.csv` (podsumowania per degradacja)
- `env_benchmark.json`
