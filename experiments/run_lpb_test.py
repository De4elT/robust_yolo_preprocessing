import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from core.yolo import Model

cfg_path = 'data/configs/yolov7.yaml'
model = Model(cfg=cfg_path, ch=3, nc=80).cuda().eval()

dummy_input = torch.randn(1, 3, 640, 640).cuda()

with torch.no_grad():
    out = model(dummy_input)

print("Model działa! Output shape:", out[0].shape)
