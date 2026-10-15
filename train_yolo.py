"""步骤 2：单独训练 YOLO 一阶段检测模型。

用法：
    python train_yolo.py
    python train_yolo.py --epochs 30 --batch 8 --imgsz 640 --device auto
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from common import resolve_device


def count_images(data_dir: Path, split: str) -> int:
    folder = data_dir / "images" / split
    if not folder.exists():
        return 0
    return len([p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])


def main() -> None:
    parser = argparse.ArgumentParser(description="步骤2：训练 YOLO 检测模型")
    parser.add_argument("--data", type=Path, default=config.DATA_YAML, help="data.yaml 路径")
    parser.add_argument("--weights", type=str, default=config.YOLO_WEIGHTS, help="预训练权重或模型 yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--name", type=str, default=config.YOLO_RUN_NAME)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_yaml = args.data.resolve()
    if not data_yaml.exists():
        raise SystemExit(f"找不到 {data_yaml}，请先运行: python make_dataset.py")

    data_dir = data_yaml.parent
    n_train = count_images(data_dir, "train")
    n_val = count_images(data_dir, "val")
    if n_train == 0 or n_val == 0:
        raise SystemExit("检测数据集为空，请先运行: python make_dataset.py")

    device = resolve_device(args.device)
    print(f"YOLO 训练  图片 train/val={n_train}/{n_val}  device={device}  weights={args.weights}")

    model = YOLO(args.weights)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(config.RUNS_DIR / "yolo"),
        name=args.name,
        exist_ok=True,
        workers=args.workers,
        seed=args.seed,
        patience=100,
        verbose=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"YOLO 权重: {best}")
    print("下一步: python train_seg.py")


if __name__ == "__main__":
    main()
