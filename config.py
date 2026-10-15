"""两阶段流水线的共享配置。

阶段 1：Ultralytics YOLO 在整图上做目标检测。
阶段 2：把检测框外扩后裁成 ROI，送入 segmentation_models_pytorch 的 UNet++ 做实例分割。
分割训练使用与推理相同外扩比例裁出的自制样本，两个模型分开训练。
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"

CLASSES = ["circle", "rectangle", "triangle"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}

IMG_SIZE = 640
SEG_IMAGE_SIZE = 256
# 检测框外扩比例。自制分割样本和推理裁剪必须使用同一个值。
BBOX_MARGIN = 0.15

ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"
SEG_IN_CHANNELS = 3
SEG_CLASSES = 1
SEG_THRESHOLD = 0.5

YOLO_WEIGHTS = "yolo11n.pt"
YOLO_RUN_NAME = "exp"
SEG_CKPT_NAME = "best.pth"

DATA_YAML = DATA_DIR / "data.yaml"
SEG_TRAIN_DIR = DATA_DIR / "seg" / "train"
SEG_VAL_DIR = DATA_DIR / "seg" / "val"
YOLO_BEST = RUNS_DIR / "yolo" / YOLO_RUN_NAME / "weights" / "best.pt"
SEG_BEST = RUNS_DIR / "seg" / SEG_CKPT_NAME
YOLO_ENGINE = YOLO_BEST.with_suffix(".engine")
YOLO_ENGINE_META = YOLO_BEST.with_suffix(".engine.json")
SEG_ENGINE = RUNS_DIR / "seg" / "best.engine"
SEG_ENGINE_META = RUNS_DIR / "seg" / "best.engine.json"
SEG_MAX_BATCH = 8
