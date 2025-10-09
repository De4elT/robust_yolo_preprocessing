import os

folders = [
    "core",
    "data/coco",
    "data/configs",
    "experiments",
    "augmentations",
    "evaluation",
    "results/plots",
    "results/checkpoints",
    "utils"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("✔️ Wszystkie foldery zostały utworzone w bieżącym katalogu.")