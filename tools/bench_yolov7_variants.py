# tools/bench_yolov7_variants.py
"""
Benchmark runner for YOLOv7-style test.py to compare thesis variants:

Variants in thesis:
1) baseline            : weights_baseline on clean + corrupt datasets
2) static_filter       : same baseline weights, but inputs pre-filtered offline (clean__filter and corrupt__filter)
3) augmented (optional): weights_aug on clean + corrupt datasets
4) learnable_preproc   : weights_preproc on clean + corrupt datasets (LPB inside model)

This script assumes you already built dataset variants using:
  python tools/robust_dataset_builder.py make-datasets --data ... --out runs/robust --splits val test ...

Then it will:
- Run YOLOv7 evaluation (utils/test.py) for each dataset variant and each model variant
- Parse mAP50 and mAP50-95 (AP@0.5:0.95) from stdout
- Parse speed (ms inference/NMS/total)
- For static filters, add measured preprocessing time to YOLO total time -> pipeline_total_ms/FPS
- Save:
    results_long.csv
    summary_by_variant.csv
    summary_by_corruption.csv
    env.json (reproducibility)

Run from PROJECT ROOT (important):
  python tools/bench_yolov7_variants.py --data data/my.yaml --variants_root runs/robust \
      --split val --weights_baseline path/to/baseline.pt \
      --weights_aug path/to/aug.pt --weights_preproc path/to/preproc.pt \
      --filters median3 bilateral --corruptions noise gblur jpeg lowlight --severities 1 2 3 --img 640 --batch 16 --device 0

Notes:
- The speed reported by test.py already includes NMS; we treat that as part of pipeline.
- If you also want to include preprocessing cost for filtered datasets, we add preproc_ms to total_ms.
"""

from __future__ import annotations
import argparse
import json
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml


# ---- Filters (must match dataset builder names for preproc timing) ----

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


# ---- Data.yaml helpers ----

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
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def measure_filter_time_ms(base_data_yaml: Path, split: str, filt_name: str, max_images: int = 300) -> float:
    cfg = load_yaml(base_data_yaml)
    base_path = resolve_base_path(base_data_yaml, cfg)
    images_dir = resolve_split_dir(base_path, cfg[split])
    imgs = list_images(images_dir)[:max_images]
    f = FILTERS[filt_name]

    times = []
    # warmup
    for p in imgs[:5]:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is not None:
            _ = f(im)

    for p in imgs:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            continue
        t0 = time.perf_counter()
        _ = f(im)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return float(np.mean(times)) if times else float("nan")


# ---- Parse YOLOv7 test output ----

RE_ALL_LINE = re.compile(r"^\s*all\s+(\d+)\s+(\d+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*$")
RE_SPEED_LINE = re.compile(r"Speed:\s*([0-9.]+)\/([0-9.]+)\/([0-9.]+)\s*ms inference\/NMS\/total", re.IGNORECASE)

def parse_test_stdout(stdout: str) -> Dict[str, float]:
    out = {
        "precision": np.nan,
        "recall": np.nan,
        "mAP50": np.nan,
        "mAP50_95": np.nan,
        "inf_ms": np.nan,
        "nms_ms": np.nan,
        "total_ms": np.nan,
    }

    for line in stdout.splitlines():

        l = line.strip()

        # ---- YOLO summary line ----
        if l.startswith("all"):
            parts = l.split()

            if len(parts) >= 7:
                try:
                    out["precision"] = float(parts[3])
                    out["recall"] = float(parts[4])
                    out["mAP50"] = float(parts[5])
                    out["mAP50_95"] = float(parts[6])
                except:
                    pass

        # ---- speed ----
        if "Speed:" in l and "inference" in l:
            try:
                sp = l.split("Speed:")[1].split("ms")[0].strip()
                inf, nms, total = sp.split("/")
                out["inf_ms"] = float(inf)
                out["nms_ms"] = float(nms)
                out["total_ms"] = float(total)
            except:
                pass

    return out


def run_yolov7_test(
    project_root: Path,
    weights: Path,
    data_yaml: Path,
    task: str,
    img: int,
    batch: int,
    device: str,
    conf_thres: float,
    iou_thres: float,
    exist_ok: bool = True,
) -> Dict[str, float]:
    """
    Executes: python -m utils.test --weights ... --data ... --task val ...
    Captures stdout and parses metrics.
    """
    cmd = [
        "python", "-m", "utils.test",
        "--weights", str(weights),
        "--data", str(data_yaml),
        "--task", task,
        "--img-size", str(img),
        "--batch-size", str(batch),
        "--device", device,
        "--conf-thres", str(conf_thres),
        "--iou-thres", str(iou_thres),
    ]
    if exist_ok:
        cmd.append("--exist-ok")

    # remove ewentualne puste stringi (na wszelki wypadek)
    cmd = [c for c in cmd if c]

    p = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
    stdout = (p.stdout or "") + "\n" + (p.stderr or "")

    if p.returncode != 0:
        # w razie faila spróbujmy coś wyciągnąć, ale zaznacz błąd
        metrics = parse_test_stdout(stdout)
        metrics["error"] = 1.0
        metrics["returncode"] = float(p.returncode)
        return metrics

    metrics = parse_test_stdout(stdout)
    metrics["error"] = 0.0
    metrics["returncode"] = 0.0
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Base clean data.yaml")
    ap.add_argument("--variants_root", required=True, help="Folder used in robust_dataset_builder.py --out (contains datasets/...)")
    ap.add_argument("--split", default="val", help="Which split to evaluate: val or test")
    ap.add_argument("--weights_baseline", required=True)
    ap.add_argument("--weights_aug", default="")
    ap.add_argument("--weights_preproc", default="")
    ap.add_argument("--filters", nargs="+", default=["median3"])
    ap.add_argument("--corruptions", nargs="+", default=["noise", "gblur", "jpeg", "lowlight"])
    ap.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf_thres", type=float, default=0.001)
    ap.add_argument("--iou_thres", type=float, default=0.65)
    ap.add_argument("--max_preproc_images", type=int, default=300)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    base_yaml = Path(args.data).resolve()
    variants_root = Path(args.variants_root).resolve()
    out_dir = variants_root
    out_dir.mkdir(parents=True, exist_ok=True)

    # env info
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "args": vars(args),
    }
    (out_dir / "env_benchmark.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    # preproc timing (per filter)
    pre_ms: Dict[str, float] = {}
    for f in args.filters:
        if f not in FILTERS:
            raise ValueError(f"Unknown filter {f}. Available: {list(FILTERS.keys())}")
        pre_ms[f] = measure_filter_time_ms(base_yaml, args.split, f, max_images=args.max_preproc_images)

    def var_yaml(name: str) -> Path:
        return variants_root / "datasets" / name / "data.yaml"

    # model variants list: (variant_name, weights, filter_name or "none")
    variants: List[Tuple[str, Path, str]] = [("baseline", Path(args.weights_baseline).resolve(), "none")]
    if args.weights_aug.strip():
        variants.append(("augmented", Path(args.weights_aug).resolve(), "none"))
    if args.weights_preproc.strip():
        variants.append(("learnable_preproc", Path(args.weights_preproc).resolve(), "none"))
    for f in args.filters:
        variants.append((f"static_filter__{f}", Path(args.weights_baseline).resolve(), f))

    rows = []

    # CLEAN
    for vname, w, f in variants:
        if f == "none":
            metrics = run_yolov7_test(project_root, w, base_yaml, args.split, args.img, args.batch, args.device, args.conf_thres, args.iou_thres)
            add_pre = 0.0
        else:
            y = var_yaml(f"clean__{f}")
            metrics = run_yolov7_test(project_root, w, y, args.split, args.img, args.batch, args.device, args.conf_thres, args.iou_thres)
            add_pre = pre_ms.get(f, float("nan"))

        pipeline_total_ms = (metrics["total_ms"] + add_pre) if (not np.isnan(metrics["total_ms"]) and not np.isnan(add_pre)) else np.nan
        pipeline_fps = (1000.0 / pipeline_total_ms) if (not np.isnan(pipeline_total_ms) and pipeline_total_ms > 0) else np.nan

        rows.append({
            "variant": vname,
            "dataset": "clean",
            "corruption": "none",
            "severity": 0,
            "filter": f,
            **metrics,
            "preproc_ms": add_pre,
            "pipeline_total_ms": pipeline_total_ms,
            "pipeline_fps": pipeline_fps,
        })

    # CORRUPT
    for c in args.corruptions:
        for s in args.severities:
            for vname, w, f in variants:
                if f == "none":
                    y = var_yaml(f"{c}_s{s}")
                    add_pre = 0.0
                else:
                    y = var_yaml(f"{c}_s{s}__{f}")
                    add_pre = pre_ms.get(f, float("nan"))

                if not y.exists():
                    raise FileNotFoundError(f"Missing dataset variant: {y}. Run robust_dataset_builder.py make-datasets")

                metrics = run_yolov7_test(project_root, w, y, args.split, args.img, args.batch, args.device, args.conf_thres, args.iou_thres)

                pipeline_total_ms = (metrics["total_ms"] + add_pre) if (not np.isnan(metrics["total_ms"]) and not np.isnan(add_pre)) else np.nan
                pipeline_fps = (1000.0 / pipeline_total_ms) if (not np.isnan(pipeline_total_ms) and pipeline_total_ms > 0) else np.nan

                rows.append({
                    "variant": vname,
                    "dataset": y.parent.parent.name,  # datasets/<name>/data.yaml -> use <name>
                    "corruption": c,
                    "severity": s,
                    "filter": f,
                    **metrics,
                    "preproc_ms": add_pre,
                    "pipeline_total_ms": pipeline_total_ms,
                    "pipeline_fps": pipeline_fps,
                })

    df = pd.DataFrame(rows)

    # deltas vs clean
    clean = df[df["dataset"] == "clean"][["variant", "mAP50_95", "pipeline_fps"]].rename(
        columns={"mAP50_95": "clean_mAP50_95", "pipeline_fps": "clean_pipeline_fps"}
    )
    df = df.merge(clean, on="variant", how="left")
    df["delta_mAP50_95"] = df["mAP50_95"] - df["clean_mAP50_95"]
    df["delta_pipeline_fps"] = df["pipeline_fps"] - df["clean_pipeline_fps"]

    df_corrupt = df[df["dataset"] != "clean"].copy()

    summary_by_variant = (
        df_corrupt.groupby(["variant"], as_index=False)[
            ["mAP50", "mAP50_95", "precision", "recall", "pipeline_fps", "delta_mAP50_95", "delta_pipeline_fps"]
        ].mean().sort_values("mAP50_95", ascending=False)
    )

    summary_by_corruption = (
        df_corrupt.groupby(["corruption", "severity", "variant"], as_index=False)[
            ["mAP50", "mAP50_95", "precision", "recall", "pipeline_fps"]
        ].mean().sort_values(["corruption", "severity", "variant"])
    )

    (out_dir / "results_long.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (out_dir / "summary_by_variant.csv").write_text(summary_by_variant.to_csv(index=False), encoding="utf-8")
    (out_dir / "summary_by_corruption.csv").write_text(summary_by_corruption.to_csv(index=False), encoding="utf-8")

    print("OK. Saved:")
    print(f"- {out_dir / 'results_long.csv'}")
    print(f"- {out_dir / 'summary_by_variant.csv'}")
    print(f"- {out_dir / 'summary_by_corruption.csv'}")
    print(f"- {out_dir / 'env_benchmark.json'}")


if __name__ == "__main__":
    main()
