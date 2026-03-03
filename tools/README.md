# Tools (robustness benchmark)

## 1) Build dataset variants (offline corruptions + optional filters)
Run from project root:

python tools/robust_dataset_builder.py make-datasets --data PATH/TO/data.yaml --out runs/robust --splits val test \
  --corruptions noise gblur jpeg lowlight --severities 1 2 3 --filters median3 bilateral

This creates:
runs/robust/datasets/<variant_name>/data.yaml

Variants:
- clean__<filter>
- <corruption>_s<k>
- <corruption>_s<k>__<filter>

## 2) Build offline corrupted training set for augmented training
python tools/robust_dataset_builder.py make-aug-train --data PATH/TO/data.yaml --out runs/robust \
  --corruptions noise gblur jpeg lowlight --severities 1 2 3 --copies_per_image 1

Creates:
runs/robust/datasets/train_aug/data.yaml

You can train YOLOv7 on it via:
python utils/train.py --data runs/robust/datasets/train_aug/data.yaml --weights <pretrained_or_baseline.pt> --cfg <your_cfg.yaml> --name aug

## 3) Benchmark variants (YOLOv7 test.py)
python tools/bench_yolov7_variants.py --data PATH/TO/data.yaml --variants_root runs/robust --split val \
  --weights_baseline PATH/baseline.pt --weights_aug PATH/aug.pt --weights_preproc PATH/preproc.pt \
  --filters median3 bilateral --corruptions noise gblur jpeg lowlight --severities 1 2 3 --img 640 --batch 16 --device 0

Outputs in runs/robust:
- results_long.csv
- summary_by_variant.csv
- summary_by_corruption.csv
- env_benchmark.json

## Notes
- For static_filter variants we add measured preprocessing time (CPU filter) to YOLO total time -> pipeline FPS.
- For learnable_preproc you need weights trained with LPB inside model (core/yolo.py uses core/lpb.py).

## 4) Dataset sanity check
Przed treningiem/benchmarkiem warto sprawdzić strukturę i labele:

python tools/check_dataset.py --data data/crowdhuman.yaml --validate

## 5) CrowdHuman .odgt -> YOLO labels
Jeśli masz adnotacje CrowdHuman w .odgt:

python tools/convert_crowdhuman_odgt_to_yolo.py --odgt <annotation_train.odgt> --images_dir <images/train> --labels_out <labels/train>
