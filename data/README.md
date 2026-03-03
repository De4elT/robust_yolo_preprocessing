# Dane (data/) – gdzie co trzymać

Ten katalog jest przeznaczony na **lokalne dane**. Nie commituj datasetów do GitHuba.

## 1) coco128 (szybki smoke test)

Docelowa struktura po rozpakowaniu:

```
data/coco/coco128/
  images/train2017/*.jpg
  labels/train2017/*.txt
```

Najprościej:
1. Wrzuć `coco128.zip` do root projektu.
2. Uruchom:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_coco128.ps1
```

Plik YAML: `data/coco128.yaml` (w repo).

---

## 2) CrowdHuman (docelowy zbiór)

**Ważne:** CrowdHuman jest udostępniany na określonej licencji. Pobierz go z oficjalnego źródła i zgodnie z warunkami.

### Docelowa struktura (YOLO format)

```
data/crowdhuman/yolo/
  images/
    train/   (jpg)
    val/     (jpg)
  labels/
    train/   (txt)
    val/     (txt)
```

Plik YAML: `data/crowdhuman.yaml` (w repo).

### Minimalny „plan działania”
1) Pobierz i rozpakuj dane CrowdHuman do katalogu `data/crowdhuman/raw/`.

Przykład (dla train01):

```
data/crowdhuman/raw/train01/Images/*.jpg
```

**Uwaga o nazwie folderu:** YOLOv7 mapuje ścieżki labeli przez zamianę `/images/` → `/labels/`.
Dlatego najlepiej, żeby obrazy były w folderze o nazwie **`images`** (małe litery).
Jeśli po rozpakowaniu masz `Images`, zmień na `images`.

2) Konwersja adnotacji `.odgt` do YOLO:

```powershell
python tools\convert_crowdhuman_odgt_to_yolo.py \
  --odgt data\crowdhuman\raw\annotation_train.odgt \
  --images_dir data\crowdhuman\raw\train01\images \
  --labels_out data\crowdhuman\yolo\labels\train
```

Analogicznie dla walidacji (`annotation_val.odgt` + folder z obrazami val → `labels/val`).

3) Ustaw obrazy w `data/crowdhuman/yolo/images/...`:
- najprościej: **przenieś/zmień nazwę** folderów na docelowe `.../yolo/images/train` i `.../yolo/images/val`
- albo użyj skryptu `scripts/prepare_crowdhuman.ps1` (próbuje utworzyć połączenia katalogów / przygotować strukturę)

4) Sprawdź kompletność:

```powershell
python tools\check_dataset.py --data data\crowdhuman.yaml
```

Jeśli widzisz dużo brakujących labeli, to znaczy że ścieżki `images/` nie są zgodne z oczekiwaniem (małe litery!) albo konwersja wskazywała zły katalog obrazów.
