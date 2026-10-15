"""步骤 4：YOLO 检测 + UNet++ 分割推理，并打印各阶段耗时和总延时。

阶段计时（相加约等于端到端总延时）：
1. 读图
2. YOLO 一阶段检测（另附 Ultralytics 内部 preprocess / inference / postprocess）
3. 按检测框外扩并裁剪 ROI
4. 分割预处理（官网 encoder preprocessing）
5. UNet++ 二阶段分割
6. 掩码缩放回贴到原图
7. 可视化保存

YOLO 与 UNet++ 都使用 TensorRT engine。

用法：
    python export_engines.py
    python infer.py
    python infer.py --source data/images/val
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import simplejpeg
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from common import cuda_sync, resolve_device, torch_device
from seg_engine import SegEngine
from yolo_runtime import YoloEngine

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
STAGE_ORDER = ("load", "yolo", "crop", "seg_pre", "seg", "paste", "vis")
STAGE_NAMES = {
    "load": "读图",
    "yolo": "YOLO 一阶段检测",
    "crop": "ROI 裁剪",
    "seg_pre": "分割预处理",
    "seg": "UNet++ 二阶段分割",
    "paste": "掩码回贴",
    "vis": "结果可视化",
}
CLASS_COLORS = {
    0: (40, 40, 220),
    1: (50, 190, 50),
    2: (220, 90, 40),
}


def list_images(source: Path):
    if source.is_file():
        return [source]
    if not source.exists():
        raise SystemExit(f"找不到输入: {source}")
    paths = [p for p in sorted(source.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if not paths:
        raise SystemExit(f"目录中没有图片: {source}")
    return paths


def expand_box(x1, y1, x2, y2, width, height, margin):
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    ex1 = int(np.floor(x1 - bw * margin))
    ey1 = int(np.floor(y1 - bh * margin))
    ex2 = int(np.ceil(x2 + bw * margin))
    ey2 = int(np.ceil(y2 + bh * margin))
    ex1 = max(0, min(ex1, width - 1))
    ey1 = max(0, min(ey1, height - 1))
    ex2 = max(ex1 + 1, min(width, ex2))
    ey2 = max(ey1 + 1, min(height, ey2))
    return ex1, ey1, ex2, ey2


def imagenet_stats(device: torch.device):
    """与 SMP get_preprocessing_fn('resnet34', 'imagenet') 相同的均值和标准差，放在 GPU 上做归一化。"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return mean, std


def load_seg_engine(path: Path):
    if path.suffix != ".engine":
        raise SystemExit(f"UNet++ 需要 TensorRT engine，当前是 {path}。请先运行: python export_engines.py")
    return SegEngine(path, path.with_suffix(".engine.json"))


def gpu_seg_batch(bgr_hwc: torch.Tensor, rois, seg_size: int, mean, std):
    """在已经上传的图像上裁 ROI，并在 GPU 上缩放、做 ImageNet 归一化。"""
    pieces = []
    for x1, y1, x2, y2 in rois:
        roi = bgr_hwc[y1:y2, x1:x2]
        if roi.numel() == 0:
            continue
        chw = roi.permute(2, 0, 1).unsqueeze(0)[:, [2, 1, 0]].float()
        pieces.append(F.interpolate(chw, size=(seg_size, seg_size), mode="bilinear", align_corners=False))
    if not pieces:
        return None
    batch = torch.cat(pieces, 0).mul_(1.0 / 255.0)
    return batch.sub(mean).div(std).contiguous()


def warmup(yolo: YoloEngine, seg_model, device, imgsz, seg_size, mean, std):
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    batch = torch.zeros(4, 3, seg_size, seg_size, device=device)
    for _ in range(5):
        bgr = torch.from_numpy(dummy).pin_memory().to(device, non_blocking=True)
        images, ratio, pad_x, pad_y = yolo.preprocess(bgr, imgsz)
        preds = yolo.infer(images)
        yolo.decode(preds, 0.25, 0.7, (imgsz, imgsz), ratio, pad_x, pad_y)
        seg_model(batch)
        gpu_seg_batch(bgr, [(40, 40, 200, 200), (240, 240, 420, 420)], seg_size, mean, std)
    cuda_sync()


_COLOR_LUT = np.array([CLASS_COLORS[i] for i in range(len(CLASS_COLORS))], dtype=np.uint8)


def draw_result(image, boxes, class_map, mask):
    """只在前景像素上做整数混合，避免整图转成 float32。"""
    vis = image.copy()
    foreground = mask > 0
    if np.any(foreground):
        pixels = vis[foreground].astype(np.uint16)
        colors = _COLOR_LUT[class_map[foreground]].astype(np.uint16)
        vis[foreground] = ((pixels * 9 + colors * 7) >> 4).astype(np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        color = CLASS_COLORS.get(box["cls"], (0, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_8)
        label = f"{box['name']} {box['conf']:.2f}"
        cv2.putText(
            vis,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_8,
        )
    return vis


def ms_since(start: float) -> float:
    cuda_sync()
    return (time.perf_counter() - start) * 1000.0


def infer_one(image_path, yolo, seg_model, mean, std, device, imgsz, seg_size, margin, threshold, conf, iou):
    timings = {}
    total_start = time.perf_counter()
    cuda_sync()

    start = time.perf_counter()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    timings["load"] = ms_since(start)

    height, width = image.shape[:2]
    start = time.perf_counter()
    bgr = torch.from_numpy(np.ascontiguousarray(image)).pin_memory().to(device, non_blocking=True)
    yolo_in, ratio, pad_x, pad_y = yolo.preprocess(bgr, imgsz)
    timings_pre = ms_since(start)

    start = time.perf_counter()
    preds = yolo.infer(yolo_in)
    timings_inf = ms_since(start)

    start = time.perf_counter()
    boxes, scores, classes = yolo.decode(preds, conf, iou, (height, width), ratio, pad_x, pad_y)
    xyxy = boxes.detach().cpu().numpy()
    confs = scores.detach().cpu().numpy()
    cls_ids = classes.detach().cpu().numpy().astype(np.int32)
    timings_post = ms_since(start)
    timings["yolo"] = timings_pre + timings_inf + timings_post
    speed = {"preprocess": timings_pre, "inference": timings_inf, "postprocess": timings_post}

    start = time.perf_counter()
    metas = []
    detections = []
    for box, cls_id, score in zip(xyxy, cls_ids, confs):
        x1, y1, x2, y2 = [float(v) for v in box]
        ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, width, height, margin)
        if ex2 <= ex1 or ey2 <= ey1:
            continue
        metas.append((ex1, ey1, ex2, ey2))
        detections.append(
            {
                "cls": int(cls_id),
                "name": yolo.names.get(int(cls_id), str(int(cls_id))),
                "conf": float(score),
                "xyxy": (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                "roi": (ex1, ey1, ex2, ey2),
            }
        )
    timings["crop"] = ms_since(start)

    mask = np.zeros((height, width), dtype=np.uint8)
    class_map = np.full((height, width), -1, dtype=np.int16)

    start = time.perf_counter()
    tensor = gpu_seg_batch(bgr, [item["roi"] for item in detections], seg_size, mean, std)
    timings["seg_pre"] = ms_since(start)

    probs = None
    start = time.perf_counter()
    if tensor is not None:
        logits = seg_model(tensor)
        probs = torch.sigmoid(logits)[:, 0]
    timings["seg"] = ms_since(start)

    start = time.perf_counter()
    prob_maps = probs.detach().cpu().numpy() if probs is not None else []
    for det, (ex1, ey1, ex2, ey2), prob in zip(detections, metas, prob_maps):
        roi_w = ex2 - ex1
        roi_h = ey2 - ey1
        pred = cv2.resize(prob, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR) >= threshold
        region_mask = mask[ey1:ey2, ex1:ex2]
        region_cls = class_map[ey1:ey2, ex1:ex2]
        region_mask[pred] = 255
        region_cls[pred] = det["cls"]
        mask[ey1:ey2, ex1:ex2] = region_mask
        class_map[ey1:ey2, ex1:ex2] = region_cls
        det["mask_pixels"] = int(pred.sum())
    timings["paste"] = ms_since(start)

    start = time.perf_counter()
    vis = draw_result(image, detections, class_map, mask)
    timings["vis"] = ms_since(start)
    timings["total"] = (time.perf_counter() - total_start) * 1000.0
    return vis, mask, detections, timings, speed


def display_width(text: str) -> int:
    return sum(2 if ord(ch) > 127 else 1 for ch in text)


def pad_display(text: str, width: int) -> str:
    return text + " " * max(1, width - display_width(text))


def print_report(image_path: Path, detections, timings, speed, out_path: Path) -> None:
    print(f"\n图像: {image_path.name}  检测数: {len(detections)}")
    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        print(
            f"  - {det['name']} conf={det['conf']:.2f} "
            f"box=({x1},{y1},{x2},{y2}) mask_pixels={det.get('mask_pixels', 0)}"
        )
    print("阶段耗时:")
    for key in STAGE_ORDER:
        print(f"  {pad_display(STAGE_NAMES[key], 22)} {timings[key]:8.2f} ms")
    if speed:
        print("  YOLO 内部细分（已包含在检测阶段内）:")
        for key in ("preprocess", "inference", "postprocess"):
            if key in speed:
                print(f"    {key:<20} {speed[key]:8.2f} ms")
    print(f"端到端总延时: {timings['total']:.2f} ms")
    print(f"输出: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="步骤4：两阶段推理并统计耗时")
    parser.add_argument("--source", type=Path, default=config.DATA_DIR / "images" / "val")
    parser.add_argument("--yolo", type=Path, default=config.YOLO_ENGINE)
    parser.add_argument("--seg", type=Path, default=config.SEG_ENGINE)
    parser.add_argument("--out", type=Path, default=config.RUNS_DIR / "infer")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    parser.add_argument("--conf", type=float, default=0.25, help="推理时的置信度阈值，不写入 engine")
    parser.add_argument("--iou", type=float, default=0.7, help="带 NMS 的模型在 GPU 上使用的 IoU 阈值")
    parser.add_argument("--limit", type=int, default=0, help="只推理前 N 张，0 表示全部")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    device_name = resolve_device(args.device)
    device = torch_device(device_name)
    if device.type != "cuda":
        raise SystemExit("TensorRT engine 需要 CUDA GPU。")

    yolo = YoloEngine(args.yolo.resolve(), device)
    seg_model = load_seg_engine(args.seg.resolve())
    mean, std = imagenet_stats(device)
    seg_size = seg_model.image_size
    margin = float(seg_model.meta.get("margin", config.BBOX_MARGIN))
    threshold = float(seg_model.meta.get("threshold", config.SEG_THRESHOLD))

    images = list_images(args.source.resolve())
    if args.limit > 0:
        images = images[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    mode = "端到端，GPU 按 conf 过滤" if yolo.filter_mode == "end2end" else "GPU NMS，conf/iou 可调"
    print(f"YOLO: {args.yolo.resolve()}  backend=TensorRT engine  过滤={mode}  conf={args.conf}  iou={args.iou}")
    print(f"UNet++: {args.seg.resolve()}  backend=TensorRT engine  device=cuda:{device.index}  input={seg_size}")

    if not args.no_warmup:
        warmup(yolo, seg_model, device, args.imgsz, seg_size, mean, std)
        print("预热完成。置信度在本次推理中过滤，没有写进 engine。")

    reports = []
    for image_path in images:
        vis, mask, detections, timings, speed = infer_one(
            image_path,
            yolo,
            seg_model,
            mean,
            std,
            device,
            args.imgsz,
            seg_size,
            margin,
            threshold,
            args.conf,
            args.iou,
        )
        stem = image_path.stem
        vis_path = args.out / f"{stem}_vis.jpg"
        mask_path = args.out / f"{stem}_mask.png"
        time_path = args.out / f"{stem}_time.json"
        save_start = time.perf_counter()
        vis_path.write_bytes(
            simplejpeg.encode_jpeg(np.ascontiguousarray(vis), quality=85, colorspace="BGR", fastdct=True)
        )
        cv2.imwrite(str(mask_path), mask, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
        save_ms = (time.perf_counter() - save_start) * 1000.0
        timings["vis"] += save_ms
        timings["total"] += save_ms
        payload = {
            "image": image_path.name,
            "detections": detections,
            "timings_ms": {key: round(timings[key], 3) for key in list(STAGE_ORDER) + ["total"]},
            "yolo_speed_ms": speed,
        }
        time_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print_report(image_path, detections, timings, speed, vis_path)
        reports.append(timings)

    if len(reports) > 1:
        print("\n平均耗时:")
        for key in STAGE_ORDER:
            avg = sum(item[key] for item in reports) / len(reports)
            print(f"  {pad_display(STAGE_NAMES[key], 22)} {avg:8.2f} ms")
        avg_total = sum(item["total"] for item in reports) / len(reports)
        print(f"平均端到端总延时: {avg_total:.2f} ms  （{len(reports)} 张）")


if __name__ == "__main__":
    main()
