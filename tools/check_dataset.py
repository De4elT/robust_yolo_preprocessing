"""tools/check_dataset.py

Szybka walidacja datasetu w formacie YOLO (YOLOv7-compatible).

Co sprawdza:
- czy foldery splitów (train/val/test) istnieją
- liczbę obrazów i labeli
- ile obrazów nie ma labela
- (opcjonalnie) próbkową walidację formatów labeli

Użycie:
  python tools/check_dataset.py --data data/crowdhuman.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(p: Path) -> Dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_base_path(data_yaml_path: Path, cfg: Dict) -> Path:
    base = cfg.get("path")
    if base is None:
        return data_yaml_path.parent.resolve()
    p = Path(base)
    return p if p.is_absolute() else (data_yaml_path.parent / p).resolve()


def resolve_split_dir(base_path: Path, split_value: str) -> Path:
    p = Path(split_value)
    return p if p.is_absolute() else (base_path / p).resolve()


def list_images(images_dir: Path) -> List[Path]:
    files = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files


def img_to_label_path(img_path: Path) -> Path:
    # YOLOv7 dataset mapping: /images/ -> /labels/
    s = str(img_path)
    if "\\images\\" in s:
        s = s.replace("\\images\\", "\\labels\\", 1)
    elif "/images/" in s:
        s = s.replace("/images/", "/labels/", 1)
    else:
        # fallback: sibling labels folder
        s = s
    return Path(s).with_suffix(".txt")


def sample_validate_labels(label_paths: List[Path], max_files: int = 50) -> Tuple[int, int, int]:
    """Returns: (ok, bad, empty)"""
    ok = bad = empty = 0
    for p in label_paths[:max_files]:
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not txt:
            empty += 1
            continue
        good_file = True
        for line in txt.splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                good_file = False
                break
            try:
                cls = int(float(parts[0]))
                vals = [float(x) for x in parts[1:]]
            except Exception:
                good_file = False
                break
            if cls < 0:
                good_file = False
                break
            if any(v < 0.0 or v > 1.0 for v in vals):
                good_file = False
                break
        if good_file:
            ok += 1
        else:
            bad += 1
    return ok, bad, empty


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to data.yaml")
    ap.add_argument("--validate", action="store_true", help="Do sample label format validation")
    ap.add_argument("--max_validate", type=int, default=50)
    args = ap.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(f"[check_dataset] Brak pliku: {data_yaml}")

    cfg = load_yaml(data_yaml)
    base_path = resolve_base_path(data_yaml, cfg)

    print(f"[check_dataset] data.yaml : {data_yaml}")
    print(f"[check_dataset] base path : {base_path}")

    for split in ("train", "val", "test"):
        if split not in cfg:
            continue
        split_dir = resolve_split_dir(base_path, cfg[split])
        if not split_dir.exists():
            print(f"[check_dataset] {split:5s}: MISSING DIR -> {split_dir}")
            continue

        imgs = list_images(split_dir)
        labels = [img_to_label_path(p) for p in imgs]
        missing = sum(1 for lp in labels if not lp.exists())

        print(f"[check_dataset] {split:5s}: images={len(imgs):6d}  labels_missing={missing:6d}  images_dir={split_dir}")

        if args.validate and imgs:
            ok, bad, empty = sample_validate_labels(labels, max_files=args.max_validate)
            print(f"[check_dataset] {split:5s}: label sample validate: ok={ok} bad={bad} empty={empty} (sample={min(len(labels), args.max_validate)})")

    # basic sanity: nc and names
    nc = cfg.get("nc")
    names = cfg.get("names")
    if nc is not None and names is not None:
        try:
            if isinstance(names, list) and len(names) != int(nc):
                print(f"[check_dataset] WARNING: nc={nc} ale names ma len={len(names)}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
