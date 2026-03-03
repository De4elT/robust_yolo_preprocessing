# tools/robust_dataset_builder.py
"""
Robust Dataset Builder for YOLO-format datasets (YOLOv7-compatible)

Creates offline dataset variants for splits (train/val/test):
- clean__<filter> (optional)
- <corruption>_s<k>
- <corruption>_s<k>__<filter>

Also can create offline corrupted training set:
- datasets/train_aug (train images duplicated and corrupted, val kept clean)

Why offline?
- Fully reproducible experiments (important for thesis)
- No need to modify YOLO dataloader/augmentations
- Enables fair comparisons: same images, controlled degradations

Dependencies:
  pip install opencv-python pyyaml tqdm numpy

Usage (from project root):
  python tools/robust_dataset_builder.py make-datasets --data data/your.yaml --out runs/robust --splits val test \
    --corruptions noise gblur jpeg lowlight --severities 1 2 3 --filters median3 bilateral

  python tools/robust_dataset_builder.py make-aug-train --data data/your.yaml --out runs/robust \
    --corruptions noise gblur jpeg lowlight --severities 1 2 3 --copies_per_image 1
"""

from __future__ import annotations
import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# ----------------------------
# Corruptions (degradacje)
# ----------------------------

def _clip_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)

def corr_gaussian_noise(img: np.ndarray, severity: int, rng: np.random.RandomState) -> np.ndarray:
    sigmas = {1: 5.0, 2: 15.0, 3: 30.0}
    sigma = sigmas.get(severity, 15.0)
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    return _clip_u8(img.astype(np.float32) + noise)

def corr_gaussian_blur(img: np.ndarray, severity: int) -> np.ndarray:
    ks = {1: 3, 2: 5, 3: 9}.get(severity, 5)
    if ks % 2 == 0:
        ks += 1
    return cv2.GaussianBlur(img, (ks, ks), 0)

def corr_motion_blur(img: np.ndarray, severity: int, rng: np.random.RandomState) -> np.ndarray:
    lengths = {1: 7, 2: 13, 3: 21}.get(severity, 13)
    angle = rng.uniform(0, np.pi)
    k = np.zeros((lengths, lengths), dtype=np.float32)
    c = lengths // 2
    for i in range(lengths):
        x = int(c + (i - c) * np.cos(angle))
        y = int(c + (i - c) * np.sin(angle))
        if 0 <= x < lengths and 0 <= y < lengths:
            k[y, x] = 1.0
    k /= (k.sum() + 1e-8)
    return cv2.filter2D(img, -1, k)

def corr_jpeg(img: np.ndarray, severity: int) -> np.ndarray:
    qualities = {1: 60, 2: 35, 3: 15}.get(severity, 35)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), qualities])
    if not ok:
        return img
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

def corr_lowlight(img: np.ndarray, severity: int) -> np.ndarray:
    gammas = {1: 1.6, 2: 2.2, 3: 3.0}.get(severity, 2.2)
    x = img.astype(np.float32) / 255.0
    x = np.power(x, gammas)
    return (x * 255.0).astype(np.uint8)

CORRUPTIONS = {
    "noise": corr_gaussian_noise,
    "gblur": corr_gaussian_blur,
    "mblur": corr_motion_blur,
    "jpeg": corr_jpeg,
    "lowlight": corr_lowlight,
}


# ----------------------------
# Static filters (filtracja)
# ----------------------------

def filt_median(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(img, ksize)

def filt_bilateral(img: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)

def filt_clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    out = cv2.merge([l2, a, b])
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

def filt_unsharp(img: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)

FILTERS = {
    "median3": lambda im: filt_median(im, 3),
    "median5": lambda im: filt_median(im, 5),
    "bilateral": filt_bilateral,
    "clahe": filt_clahe,
    "unsharp": filt_unsharp,
}


# ----------------------------
# Dataset utils (YOLO yaml)
# ----------------------------

def load_yaml(p: Path) -> Dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(p: Path, obj: Dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

def resolve_base_path(data_yaml_path: Path, cfg: Dict) -> Path:
    base = cfg.get("path")
    if base is None:
        return data_yaml_path.parent.resolve()
    p = Path(base)
    return p if p.is_absolute() else (data_yaml_path.parent / p).resolve()

def resolve_split_dir(base_path: Path, split_value: str) -> Path:
    p = Path(split_value)
    return p if p.is_absolute() else (base_path / p).resolve()

def guess_labels_dir(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts)
    return images_dir.parent.parent / "labels" / images_dir.name

def list_images(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files

def label_for_image(img_path: Path, labels_dir: Path) -> Path:
    return labels_dir / (img_path.stem + ".txt")


@dataclass
class VariantSpec:
    name: str
    corruption: Optional[str] = None
    severity: int = 0
    filt: Optional[str] = None


def apply_variant(img: np.ndarray, spec: VariantSpec, rng: np.random.RandomState) -> np.ndarray:
    out = img
    if spec.corruption:
        c = spec.corruption
        if c not in CORRUPTIONS:
            raise ValueError(f"Unknown corruption: {c}")
        if c in ("noise", "mblur"):
            out = CORRUPTIONS[c](out, spec.severity, rng)  # type: ignore
        else:
            out = CORRUPTIONS[c](out, spec.severity)  # type: ignore
    if spec.filt:
        if spec.filt not in FILTERS:
            raise ValueError(f"Unknown filter: {spec.filt}")
        out = FILTERS[spec.filt](out)
    return out


def build_variant_dataset(
    base_data_yaml: Path,
    out_root: Path,
    spec: VariantSpec,
    splits: List[str],
    max_images: Optional[int],
    seed: int,
    overwrite: bool,
) -> Path:
    cfg = load_yaml(base_data_yaml)
    base_path = resolve_base_path(base_data_yaml, cfg)

    variant_root = out_root / "datasets" / spec.name
    yaml_path = variant_root / "data.yaml"

    if variant_root.exists() and overwrite:
        shutil.rmtree(variant_root)

    if yaml_path.exists() and not overwrite:
        return yaml_path

    rng = np.random.RandomState(seed)

    out_cfg: Dict = {"path": str(variant_root.resolve())}
    if "names" in cfg:
        out_cfg["names"] = cfg["names"]
        out_cfg["nc"] = cfg.get("nc", len(cfg["names"]) if isinstance(cfg["names"], list) else len(cfg["names"].keys()))
    elif "nc" in cfg:
        out_cfg["nc"] = cfg["nc"]
    else:
        raise ValueError("data.yaml must contain 'names' or 'nc'")

    for split in splits:
        if split not in cfg:
            raise ValueError(f"Split '{split}' not present in data.yaml")
        images_dir = resolve_split_dir(base_path, cfg[split])
        labels_dir = guess_labels_dir(images_dir)

        out_images = variant_root / "images" / split
        out_labels = variant_root / "labels" / split
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        imgs = list_images(images_dir)
        if max_images is not None:
            imgs = imgs[:max_images]

        for img_path in tqdm(imgs, desc=f"{spec.name} :: {split}"):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Cannot read image: {img_path}")

            img2 = apply_variant(img, spec, rng)
            out_img = out_images / (img_path.stem + ".jpg")
            cv2.imwrite(str(out_img), img2, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            src_lab = label_for_image(img_path, labels_dir)
            dst_lab = out_labels / (img_path.stem + ".txt")
            if src_lab.exists():
                shutil.copy2(src_lab, dst_lab)
            else:
                dst_lab.write_text("", encoding="utf-8")

        out_cfg[split] = f"images/{split}"

    save_yaml(yaml_path, out_cfg)
    return yaml_path


def build_aug_train_dataset(
    base_data_yaml: Path,
    out_root: Path,
    name: str,
    corruptions: List[str],
    severities: List[int],
    copies_per_image: int,
    seed: int,
    overwrite: bool,
    max_images: Optional[int],
) -> Path:
    cfg = load_yaml(base_data_yaml)
    base_path = resolve_base_path(base_data_yaml, cfg)

    if "train" not in cfg or "val" not in cfg:
        raise ValueError("Need train and val in data.yaml for make-aug-train")

    variant_root = out_root / "datasets" / name
    yaml_path = variant_root / "data.yaml"

    if variant_root.exists() and overwrite:
        shutil.rmtree(variant_root)

    if yaml_path.exists() and not overwrite:
        return yaml_path

    rng = np.random.RandomState(seed)

    out_cfg: Dict = {"path": str(variant_root.resolve())}
    if "names" in cfg:
        out_cfg["names"] = cfg["names"]
        out_cfg["nc"] = cfg.get("nc", len(cfg["names"]) if isinstance(cfg["names"], list) else len(cfg["names"].keys()))
    elif "nc" in cfg:
        out_cfg["nc"] = cfg["nc"]
    else:
        raise ValueError("data.yaml must contain 'names' or 'nc'")

    train_images_dir = resolve_split_dir(base_path, cfg["train"])
    train_labels_dir = guess_labels_dir(train_images_dir)

    out_train_images = variant_root / "images" / "train"
    out_train_labels = variant_root / "labels" / "train"
    out_train_images.mkdir(parents=True, exist_ok=True)
    out_train_labels.mkdir(parents=True, exist_ok=True)

    imgs = list_images(train_images_dir)
    if max_images is not None:
        imgs = imgs[:max_images]

    for img_path in tqdm(imgs, desc=f"{name} :: train"):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")

        src_lab = label_for_image(img_path, train_labels_dir)
        lab_text = src_lab.read_text(encoding="utf-8") if src_lab.exists() else ""

        for k in range(copies_per_image):
            c = str(rng.choice(corruptions))
            s = int(rng.choice(severities))
            spec = VariantSpec(name="tmp", corruption=c, severity=s, filt=None)
            img2 = apply_variant(img, spec, rng)

            out_name = f"{img_path.stem}__{c}_s{s}__k{k}.jpg"
            out_img = out_train_images / out_name
            cv2.imwrite(str(out_img), img2, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            out_lab = out_train_labels / (Path(out_name).stem + ".txt")
            out_lab.write_text(lab_text, encoding="utf-8")

    # VAL clean passthrough
    val_images_dir = resolve_split_dir(base_path, cfg["val"])
    val_labels_dir = guess_labels_dir(val_images_dir)

    out_val_images = variant_root / "images" / "val"
    out_val_labels = variant_root / "labels" / "val"
    out_val_images.mkdir(parents=True, exist_ok=True)
    out_val_labels.mkdir(parents=True, exist_ok=True)

    val_imgs = list_images(val_images_dir)
    if max_images is not None:
        val_imgs = val_imgs[:max_images]

    for img_path in tqdm(val_imgs, desc=f"{name} :: val"):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        out_img = out_val_images / (img_path.stem + ".jpg")
        cv2.imwrite(str(out_img), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        src_lab = label_for_image(img_path, val_labels_dir)
        dst_lab = out_val_labels / (img_path.stem + ".txt")
        if src_lab.exists():
            shutil.copy2(src_lab, dst_lab)
        else:
            dst_lab.write_text("", encoding="utf-8")

    out_cfg["train"] = "images/train"
    out_cfg["val"] = "images/val"
    save_yaml(yaml_path, out_cfg)
    return yaml_path


def build_parser():
    p = argparse.ArgumentParser(prog="robust_dataset_builder.py")
    sp = p.add_subparsers(dest="cmd", required=True)

    p1 = sp.add_parser("make-datasets")
    p1.add_argument("--data", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--splits", nargs="+", default=["val"])
    p1.add_argument("--corruptions", nargs="+", default=["noise", "gblur", "jpeg", "lowlight"])
    p1.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3])
    p1.add_argument("--filters", nargs="+", default=["median3"])
    p1.add_argument("--max_images", type=int, default=0)
    p1.add_argument("--seed", type=int, default=123)
    p1.add_argument("--overwrite", action="store_true")

    p2 = sp.add_parser("make-aug-train")
    p2.add_argument("--data", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--corruptions", nargs="+", default=["noise", "gblur", "jpeg", "lowlight"])
    p2.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3])
    p2.add_argument("--copies_per_image", type=int, default=1)
    p2.add_argument("--max_images", type=int, default=0)
    p2.add_argument("--seed", type=int, default=123)
    p2.add_argument("--overwrite", action="store_true")

    return p


def main():
    args = build_parser().parse_args()
    base_yaml = Path(args.data).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    max_images = None if getattr(args, "max_images", 0) == 0 else int(args.max_images)

    if args.cmd == "make-datasets":
        for filt in args.filters:
            if filt not in FILTERS:
                raise ValueError(f"Unknown filter: {filt}. Available: {list(FILTERS.keys())}")
            spec = VariantSpec(name=f"clean__{filt}", filt=filt)
            build_variant_dataset(base_yaml, out_root, spec, args.splits, max_images, args.seed, args.overwrite)

        for c in args.corruptions:
            if c not in CORRUPTIONS:
                raise ValueError(f"Unknown corruption: {c}. Available: {list(CORRUPTIONS.keys())}")
            for s in args.severities:
                spec = VariantSpec(name=f"{c}_s{s}", corruption=c, severity=s)
                build_variant_dataset(base_yaml, out_root, spec, args.splits, max_images, args.seed, args.overwrite)

                for filt in args.filters:
                    specf = VariantSpec(name=f"{c}_s{s}__{filt}", corruption=c, severity=s, filt=filt)
                    build_variant_dataset(base_yaml, out_root, specf, args.splits, max_images, args.seed, args.overwrite)

        print("OK: dataset variants generated.")

    elif args.cmd == "make-aug-train":
        for c in args.corruptions:
            if c not in CORRUPTIONS:
                raise ValueError(f"Unknown corruption: {c}. Available: {list(CORRUPTIONS.keys())}")

        y = build_aug_train_dataset(
            base_data_yaml=base_yaml,
            out_root=out_root,
            name="train_aug",
            corruptions=args.corruptions,
            severities=args.severities,
            copies_per_image=args.copies_per_image,
            seed=args.seed,
            overwrite=args.overwrite,
            max_images=max_images,
        )
        print(f"OK: train_aug created at: {y}")


if __name__ == "__main__":
    main()
