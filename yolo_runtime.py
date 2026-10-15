"""YOLO TensorRT 推理。

置信度和 IoU 在推理时传入，不写进 engine。
- 检测头带 NMS 的模型（YOLO11 等）：在 GPU 上做 NMS，使用本次的 conf / iou。
- 端到端、没有 NMS 的模型（YOLO26 等）：engine 已输出候选框，只在 GPU 上按 conf 过滤。
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics.utils.ops import scale_boxes

import config


def letterbox_params(height: int, width: int, size: int):
    """与 Ultralytics LetterBox(center=True) 相同的缩放和填充。"""
    ratio = min(size / height, size / width)
    new_w = round(width * ratio)
    new_h = round(height * ratio)
    dw = (size - new_w) / 2
    dh = (size - new_h) / 2
    top = round(dh - 0.1)
    left = round(dw - 0.1)
    return ratio, left, top, new_h, new_w


class YoloEngine:
    def __init__(self, path: Path, device: torch.device):
        if path.suffix != ".engine":
            raise SystemExit(f"YOLO 需要 TensorRT engine，当前是 {path}。请先运行: python export_engines.py")
        if not path.exists():
            raise SystemExit(f"找不到 YOLO engine {path}，请先运行: python export_engines.py")
        from ultralytics.nn.autobackend import AutoBackend

        meta_path = path.with_suffix(".engine.json")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.device = device
        self.backend = AutoBackend(str(path), device=device, fp16=False, fuse=False, verbose=False)
        self.end2end = bool(self.meta.get("end2end", getattr(self.backend, "end2end", False)))
        names = self.meta.get("names") or getattr(self.backend, "names", None) or config.CLASSES
        if isinstance(names, dict):
            self.names = {int(key): str(value) for key, value in names.items()}
        else:
            self.names = {index: str(name) for index, name in enumerate(names)}
        self.imgsz = int(self.meta.get("imgsz", config.IMG_SIZE))
        self._warm_filter()

    def _warm_filter(self) -> None:
        """预热 GPU 过滤。空白图不会进入 NMS 内核，第一次真实检测会多出几十毫秒。"""
        import torchvision  # noqa: F401

        if self.end2end:
            sample = torch.zeros((1, 16, 6), device=self.device)
            sample[:, :, 4] = 0.9
            self._filter_scores(sample[0], 0.25)
            return
        from ultralytics.utils import nms

        channels = len(self.names) + 4
        sample = torch.zeros((1, channels, 8400), device=self.device)
        sample[:, 0, :8] = 100
        sample[:, 1, :8] = 100
        sample[:, 2, :8] = 40
        sample[:, 3, :8] = 40
        sample[:, 4, :8] = 0.9
        for _ in range(2):
            kept = nms.non_max_suppression(sample.clone(), 0.25, 0.7, max_det=300)[0]
        if kept.numel():
            scale_boxes((self.imgsz, self.imgsz), kept[:, :4], (self.imgsz, self.imgsz), ratio_pad=((1.0, 1.0), (0.0, 0.0)))

    @property
    def filter_mode(self) -> str:
        return "end2end" if self.end2end else "gpu_nms"

    def preprocess(self, bgr_hwc: torch.Tensor, size: int):
        height, width = bgr_hwc.shape[:2]
        ratio, left, top, new_h, new_w = letterbox_params(height, width, size)
        rgb = bgr_hwc[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float()
        if new_h != height or new_w != width:
            rgb = F.interpolate(rgb, size=(new_h, new_w), mode="bilinear", align_corners=False)
        right = size - new_w - left
        bottom = size - new_h - top
        if left or right or top or bottom:
            rgb = F.pad(rgb, (left, right, top, bottom), value=114.0)
        rgb.div_(255.0)
        return rgb.contiguous(), ratio, left, top

    def infer(self, images: torch.Tensor) -> torch.Tensor:
        preds = self.backend(images)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        return preds

    def decode(self, preds, conf, iou, orig_hw, ratio, pad_x, pad_y):
        """返回原图像素坐标。带 NMS 的模型在 GPU 上做 NMS；端到端模型只按 conf 过滤。"""
        if self._already_decoded(preds):
            det = self._filter_scores(preds[0], conf)
        else:
            det = self._gpu_nms(preds, conf, iou)
        if det is None or det.numel() == 0:
            empty_box = torch.zeros((0, 4), device=self.device)
            empty = torch.zeros((0,), device=self.device)
            return empty_box, empty, empty
        boxes = scale_boxes(
            (self.imgsz, self.imgsz),
            det[:, :4].clone(),
            orig_hw,
            ratio_pad=((ratio, ratio), (pad_x, pad_y)),
        )
        return boxes, det[:, 4], det[:, 5]

    def _already_decoded(self, preds: torch.Tensor) -> bool:
        # 端到端或图内已经做过 NMS 的输出是 (batch, keep, 6)：xyxy, conf, cls。
        return self.end2end or (preds.ndim == 3 and preds.shape[-1] == 6 and preds.shape[1] <= 300)

    @staticmethod
    def _filter_scores(pred: torch.Tensor, conf: float) -> torch.Tensor:
        if pred.numel() == 0:
            return pred
        return pred[pred[:, 4] > conf]

    @staticmethod
    def _gpu_nms(preds: torch.Tensor, conf: float, iou: float) -> torch.Tensor:
        from ultralytics.utils import nms

        dets = nms.non_max_suppression(preds, conf, iou, max_det=300)
        return dets[0]
