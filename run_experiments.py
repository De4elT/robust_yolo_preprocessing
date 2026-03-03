import subprocess
import re
import csv

tests = [
    ("clean", "data/coco128.yaml"),
    ("noise_s2", "runs/smoke/datasets/noise_s2/data.yaml"),
    ("noise_s2_median3", "runs/smoke/datasets/noise_s2__median3/data.yaml"),
    ("jpeg_s2", "runs/smoke/datasets/jpeg_s2/data.yaml"),
    ("jpeg_s2_median3", "runs/smoke/datasets/jpeg_s2__median3/data.yaml"),
]

weights = "weights/yolov7.pt"

results = []

for name, data_yaml in tests:

    print("Running:", name)

    cmd = [
        "python",
        "-m",
        "utils.test",
        "--weights", weights,
        "--data", data_yaml,
        "--task", "val",
        "--img-size", "640",
        "--batch-size", "8",
        "--device", "cpu"
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)

    output = proc.stdout + proc.stderr
    print(output)

    precision = recall = map50 = map5095 = None

    for line in output.splitlines():

        if line.strip().startswith("all"):
            parts = line.split()

            if len(parts) >= 7:
                precision = parts[3]
                recall = parts[4]
                map50 = parts[5]
                map5095 = parts[6]

    results.append([
        name,
        precision,
        recall,
        map50,
        map5095
    ])


with open("experiment_results.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "dataset",
        "precision",
        "recall",
        "mAP50",
        "mAP50_95"
    ])

    writer.writerows(results)

print("DONE")
print("Results saved to experiment_results.csv")