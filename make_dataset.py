"""步骤 1：自制几何形状数据集。

每张图随机放置若干不重叠的圆、矩形、三角形，同时写出：
- YOLO 检测标签（整图、紧框）
- UNet++ 分割样本（按与推理相同的比例外扩后裁剪的图像和实例掩码）

用法：
    python make_dataset.py
    python make_dataset.py --train 60 --val 20 --size 640 --seed 42
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def separated(a, b, gap: int) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax2 + gap <= bx1 or bx2 + gap <= ax1 or ay2 + gap <= by1 or by2 + gap <= ay1


def jitter_color(base_bgr, rng: np.random.Generator):
    return tuple(int(np.clip(c + int(rng.integers(-30, 31)), 0, 255)) for c in base_bgr)


def draw_shape(image, mask, kind: int, cx: int, cy: int, side: int, color, rng: np.random.Generator) -> None:
    # 掩码不用抗锯齿，保证标签是 0/255。图形限制在 side 范围内，避免和相邻目标重叠。
    limit = max(8, side // 2)
    if kind == 0:
        radius = limit
        cv2.circle(image, (cx, cy), radius, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(mask, (cx, cy), radius, 255, thickness=-1)
        return
    if kind == 1:
        half_w = max(8, int(limit * float(rng.uniform(0.75, 1.0))))
        half_h = max(8, int(limit * float(rng.uniform(0.45, 0.9))))
        cv2.rectangle(image, (cx - half_w, cy - half_h), (cx + half_w, cy + half_h), color, thickness=-1)
        cv2.rectangle(mask, (cx - half_w, cy - half_h), (cx + half_w, cy + half_h), 255, thickness=-1)
        return
    radius = limit
    pts = np.array(
        [
            [cx, cy - radius],
            [cx - radius, cy + int(radius * 0.8)],
            [cx + radius, cy + int(radius * 0.8)],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [pts], color)
    cv2.fillPoly(mask, [pts], 255)


def expand_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int, margin: float):
    bw = x2 - x1
    bh = y2 - y1
    ex1 = int(np.floor(x1 - bw * margin))
    ey1 = int(np.floor(y1 - bh * margin))
    ex2 = int(np.ceil(x2 + bw * margin))
    ey2 = int(np.ceil(y2 + bh * margin))
    ex1 = max(0, ex1)
    ey1 = max(0, ey1)
    ex2 = min(width, max(ex1 + 1, ex2))
    ey2 = min(height, max(ey1 + 1, ey2))
    return ex1, ey1, ex2, ey2


def make_background(rng: np.random.Generator, size: int) -> np.ndarray:
    base = int(rng.integers(35, 80))
    image = np.full((size, size, 3), base, dtype=np.int16)
    noise = rng.integers(0, 28, size=(size, size, 1), dtype=np.int16)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return image


def yolo_line(cls_id: int, x1: int, y1: int, x2: int, y2: int, size: int) -> str:
    cx = ((x1 + x2) / 2) / size
    cy = ((y1 + y2) / 2) / size
    bw = (x2 - x1) / size
    bh = (y2 - y1) / size
    cx = float(np.clip(cx, 0.0, 1.0))
    cy = float(np.clip(cy, 0.0, 1.0))
    bw = float(np.clip(bw, 1e-6, 1.0))
    bh = float(np.clip(bh, 1e-6, 1.0))
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def generate_one(rng: np.random.Generator, size: int, margin: float, stem: str, split_dir: dict, counts: dict):
    split_dir["pending"].clear()
    image = make_background(rng, size)
    semantic = np.zeros((size, size), dtype=np.uint8)
    occupied = []
    labels = []
    n_obj = int(rng.integers(1, 5))
    saved = 0

    for obj_i in range(n_obj):
        placed = False
        for _ in range(40):
            kind = obj_i % len(config.CLASSES)
            if rng.random() < 0.35:
                kind = int(rng.integers(0, len(config.CLASSES)))
            side = int(rng.integers(80, 170))
            half = side // 2 + 4
            if half * 2 + 8 >= size:
                continue
            cx = int(rng.integers(half + 4, size - half - 4))
            cy = int(rng.integers(half + 4, size - half - 4))
            reserved = (cx - half, cy - half, cx + half, cy + half)
            if any(not separated(reserved, prev, gap=18) for prev in occupied):
                continue

            inst = np.zeros((size, size), dtype=np.uint8)
            # BGR：圆偏红、矩形偏绿、三角偏蓝，并加少量颜色扰动
            base = ((40, 40, 220), (50, 190, 50), (220, 90, 40))[kind]
            draw_shape(image, inst, kind, cx, cy, side, jitter_color(base, rng), rng)
            ys, xs = np.where(inst > 0)
            if xs.size < 30:
                continue
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            occupied.append(reserved)
            semantic[inst > 0] = kind + 1
            labels.append(yolo_line(kind, x1, y1, x2, y2, size))
            counts[config.CLASSES[kind]] += 1
            placed = True

            # 外扩框先记下来，等 JPEG 回读后再裁，使分割样本和推理输入一致
            ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, size, size, margin)
            crop_name = f"{stem}_{saved}.png"
            # 临时把实例掩码裁下来；图像裁剪放到 JPEG 回读之后
            mask_crop = inst[ey1:ey2, ex1:ex2]
            cv2.imwrite(str(split_dir["seg_masks"] / crop_name), mask_crop)
            # 把外扩框存到 labels 同步的 sidecar，回读后用
            split_dir["pending"].append((crop_name, ex1, ey1, ex2, ey2))
            saved += 1
            break
        if not placed:
            continue

    if not labels:
        return False

    img_path = split_dir["images"] / f"{stem}.jpg"
    cv2.imwrite(str(img_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    reloaded = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if reloaded is None:
        raise RuntimeError(f"回读失败: {img_path}")

    for crop_name, ex1, ey1, ex2, ey2 in split_dir["pending"]:
        crop = reloaded[ey1:ey2, ex1:ex2]
        if crop.size == 0:
            continue
        cv2.imwrite(str(split_dir["seg_images"] / crop_name), crop)
    split_dir["pending"].clear()

    (split_dir["labels"] / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    cv2.imwrite(str(split_dir["masks"] / f"{stem}.png"), semantic)
    return True


def write_data_yaml(out: Path) -> None:
    # 不写绝对 path。Ultralytics 会把本文件所在目录当作数据集根目录。
    lines = [
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx, name in enumerate(config.CLASSES):
        lines.append(f"  {idx}: {name}")
    (out / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_split(root: Path, split: str) -> dict:
    paths = {
        "images": root / "images" / split,
        "labels": root / "labels" / split,
        "masks": root / "masks" / split,
        "seg_images": root / "seg" / split / "images",
        "seg_masks": root / "seg" / split / "masks",
        "pending": [],
    }
    for key, path in paths.items():
        if key == "pending":
            continue
        reset_dir(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="步骤1：自制 YOLO + UNet++ 数据集")
    parser.add_argument("--train", type=int, default=60, help="训练集图片数")
    parser.add_argument("--val", type=int, default=20, help="验证集图片数")
    parser.add_argument("--size", type=int, default=config.IMG_SIZE, help="图片边长")
    parser.add_argument("--margin", type=float, default=config.BBOX_MARGIN, help="分割裁剪外扩比例")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=config.DATA_DIR)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    counts = {name: 0 for name in config.CLASSES}
    summary = {"train": 0, "val": 0, "crops": 0}

    for split, total in (("train", args.train), ("val", args.val)):
        paths = prepare_split(out, split)
        made = 0
        guard = 0
        while made < total:
            guard += 1
            if guard > total * 10:
                raise RuntimeError(f"{split} 只生成了 {made}/{total} 张，请放宽尺寸或目标数量")
            stem = f"{made:04d}"
            ok = generate_one(rng, args.size, args.margin, stem, paths, counts)
            if not ok:
                continue
            made += 1
            summary[split] = made
        summary["crops"] += len(list(paths["seg_images"].glob("*.png")))

    write_data_yaml(out)
    meta = {
        "classes": config.CLASSES,
        "image_size": args.size,
        "bbox_margin": args.margin,
        "train_images": summary["train"],
        "val_images": summary["val"],
        "seg_crops": summary["crops"],
        "class_counts": counts,
        "seed": args.seed,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("数据集已生成")
    print(f"  目录: {out}")
    print(f"  训练图片: {summary['train']}  验证图片: {summary['val']}  分割裁剪: {summary['crops']}")
    print(f"  类别数量: {counts}")
    print(f"  YOLO 配置: {out / 'data.yaml'}")
    print("下一步: python train_yolo.py")


if __name__ == "__main__":
    main()
