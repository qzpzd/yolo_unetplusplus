"""按 segmentation_models_pytorch 官方 Quick Start 构建 UNet++。

官方流程：
1. smp.UnetPlusPlus(encoder_name, encoder_weights, in_channels, classes)
2. encoders.get_preprocessing_fn 做与预训练编码器一致的归一化
3. 训练时 model(images) 得到 logits，再用 smp.losses 计算损失
4. 评估时用 smp.metrics.get_stats + iou_score
"""

from typing import Callable, Optional

import numpy as np
import segmentation_models_pytorch as smp


def _normalize_weights(encoder_weights: Optional[str]) -> Optional[str]:
    if encoder_weights is None:
        return None
    text = str(encoder_weights).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return encoder_weights


def build_unetplusplus(
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> smp.UnetPlusPlus:
    """创建 UNet++。activation 保持 None，输出为 logits，配合 from_logits=True 的损失。"""
    return smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=_normalize_weights(encoder_weights),
        in_channels=in_channels,
        classes=classes,
    )


def build_preprocess(
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
) -> Callable[[np.ndarray], np.ndarray]:
    """返回 HWC RGB 图像的预处理函数。

    预训练编码器使用官网 get_preprocessing_fn（先映射到 input_range，再减均值除标准差）。
    不加载预训练权重时只做 /255。
    """
    weights = _normalize_weights(encoder_weights)
    if weights is None:
        def preprocess(image: np.ndarray) -> np.ndarray:
            return image.astype(np.float32) / 255.0

        return preprocess

    preprocess_input = smp.encoders.get_preprocessing_fn(encoder_name, pretrained=weights)

    def preprocess(image: np.ndarray) -> np.ndarray:
        return preprocess_input(image.astype(np.float32))

    return preprocess


def build_loss():
    """官网常用的二值分割损失：Dice + Soft BCE。输入为 logits，标签为 0/1。"""
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce = smp.losses.SoftBCEWithLogitsLoss()

    def loss_fn(logits, target):
        return dice(logits, target) + bce(logits, target)

    return loss_fn


def batch_iou(logits, target):
    """按官网 metrics 接口统计一个 batch 的混淆矩阵。"""
    pred = (logits.sigmoid() > 0.5).long()
    tp, fp, fn, tn = smp.metrics.get_stats(pred, target.long(), mode="binary")
    return tp, fp, fn, tn


def dataset_iou(tp, fp, fn, tn) -> float:
    score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    return float(score)
