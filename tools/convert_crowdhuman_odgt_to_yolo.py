"""tools/convert_crowdhuman_odgt_to_yolo.py

Konwersja adnotacji CrowdHuman (*.odgt) do YOLO TXT.

CrowdHuman odgt to JSON per linia (dict). Typowo:
- record["ID"] -> nazwa pliku bez rozszerzenia (np. 273271, obraz: 273271.jpg)
- record["gtboxes"] -> lista boxów, a każdy ma m.in.:
  - box["fbox"] = [x, y, w, h]
  - box["tag"] = "person" / ...
  - box.get("extra", {}).get("ignore") == 1 dla boxów ignorowanych

Skrypt:
- bierze tylko tag == 'person'
- domyślnie pomija ignore==1
- wypluwa YOLO: class_id xc yc w h (0..1)

Użycie:
  python tools/convert_crowdhuman_odgt_to_yolo.py \
    --odgt data/crowdhuman/raw/annotation_train.odgt \
    --images_dir data/crowdhuman/yolo/images/train \
    --labels_out data/crowdhuman/yolo/labels/train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
from tqdm import tqdm


IMG_EXTS = [".jpg", ".jpeg", ".png"]


def find_image(images_dir: Path, image_id: str) -> Path | None:
    for ext in IMG_EXTS:
        p = images_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    # fallback: sometimes files are nested
    for ext in IMG_EXTS:
        hits = list(images_dir.rglob(f"{image_id}{ext}"))
        if hits:
            return hits[0]
    return None


def xywh_abs_to_yolo(fbox: List[float], w: int, h: int) -> Tuple[float, float, float, float]:
    x, y, bw, bh = fbox
    xc = x + bw / 2.0
    yc = y + bh / 2.0
    # normalize
    xc /= w
    yc /= h
    bw /= w
    bh /= h
    # clip
    def clip01(v: float) -> float:
        return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v

    return clip01(xc), clip01(yc), clip01(bw), clip01(bh)


def iter_odgt(path: Path) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--odgt", required=True, help="Path to annotation .odgt")
    ap.add_argument("--images_dir", required=True, help="Directory with images (expects <ID>.jpg)")
    ap.add_argument("--labels_out", required=True, help="Output labels directory")
    ap.add_argument("--class_id", type=int, default=0)
    ap.add_argument("--keep_ignored", action="store_true", help="If set, keep ignore==1 boxes")
    ap.add_argument("--tag", default="person", help="Which tag to export (default: person)")
    args = ap.parse_args()

    odgt = Path(args.odgt)
    images_dir = Path(args.images_dir)
    labels_out = Path(args.labels_out)

    if not odgt.exists():
        raise SystemExit(f"[convert_crowdhuman] Brak pliku: {odgt}")
    if not images_dir.exists():
        raise SystemExit(f"[convert_crowdhuman] Brak katalogu obrazów: {images_dir}")

    labels_out.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_written = 0
    n_missing_img = 0

    records = list(iter_odgt(odgt))
    for rec in tqdm(records, desc=f"convert {odgt.name}"):
        n_total += 1
        img_id = str(rec.get("ID", "")).strip()
        if not img_id:
            continue

        img_path = find_image(images_dir, img_id)
        if img_path is None:
            n_missing_img += 1
            continue

        im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if im is None:
            n_missing_img += 1
            continue
        h, w = im.shape[:2]

        yolo_lines: List[str] = []
        for box in rec.get("gtboxes", []):
            if str(box.get("tag", "")) != args.tag:
                continue
            extra = box.get("extra", {}) or {}
            ignore = int(extra.get("ignore", 0)) if isinstance(extra, dict) else 0
            if ignore == 1 and not args.keep_ignored:
                continue
            fbox = box.get("fbox", None)
            if not fbox or len(fbox) != 4:
                continue

            xc, yc, bw, bh = xywh_abs_to_yolo([float(x) for x in fbox], w=w, h=h)
            # very small boxes sometimes appear; keep them (thesis can decide later)
            yolo_lines.append(f"{args.class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        out_path = labels_out / f"{img_path.stem}.txt"
        out_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        n_written += 1

    print("[convert_crowdhuman] DONE")
    print(f"[convert_crowdhuman] records_total   : {n_total}")
    print(f"[convert_crowdhuman] labels_written  : {n_written}")
    print(f"[convert_crowdhuman] missing_images  : {n_missing_img}")
    print(f"[convert_crowdhuman] labels_out      : {labels_out}")


if __name__ == "__main__":
    main()
