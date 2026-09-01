"""Convert .pt to .onnx.

Usage:
    python export_onnx.py                                   # both models, from config
    python export_onnx.py <folder> [folder ...]             # every .pt inside
    python export_onnx.py model.pt out.onnx [more.pt more.onnx] [--imgsz N]
"""

import json
import sys
from pathlib import Path

from ultralytics import YOLO

args = sys.argv[1:]
imgsz = 640
if "--imgsz" in args:
    i = args.index("--imgsz")
    imgsz = int(args[i + 1])
    del args[i : i + 2]

if not args:
    cfg = json.loads(Path("config.development.json").read_text(encoding="utf-8"))
    jobs = [
        (cfg["model_settings"]["moto_model_path"], 640, None),
        (
            cfg["model_settings"]["helmet_model_path"],
            cfg["detection"]["helmet_imgsz"],
            None,
        ),
    ]
else:
    dirs = [a for a in args if Path(a).is_dir()]
    files = [a for a in args if not Path(a).is_dir()]
    if files and len(files) % 2:
        sys.exit(
            "usage: python export_onnx.py [folder ... | model.pt out.onnx ...] [--imgsz N]"
        )
    jobs = [(str(p), imgsz, None) for d in dirs for p in sorted(Path(d).glob("*.pt"))]
    jobs += [(files[i], imgsz, files[i + 1]) for i in range(0, len(files), 2)]

for pt, size, out in jobs:
    result = YOLO(pt).export(format="onnx", imgsz=size)
    if out and Path(result) != Path(out):
        Path(result).replace(out)  # ultralytics always writes next to the .pt
        result = out
    print(f"{pt} -> {result} (imgsz={size}, classes={YOLO(result).names})")
# .\venv\Scripts\python.exe export_onnx.py train/best_8_250.pt train/best_8_250.onnx --imgsz 1280
