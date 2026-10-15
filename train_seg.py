"""步骤 3：单独训练 UNet++ 二阶段分割模型。

模型、预处理、损失和 IoU 都走 segmentation_models_pytorch 官方接口：
- smp.UnetPlusPlus
- smp.encoders.get_preprocessing_fn
- smp.losses.DiceLoss + SoftBCEWithLogitsLoss
- smp.metrics.get_stats / iou_score

用法：
    python train_seg.py
    python train_seg.py --epochs 20 --batch 8 --image-size 256 --device auto
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from common import resolve_device, seed_everything, torch_device
from smp_unet import batch_iou, build_loss, build_preprocess, build_unetplusplus, dataset_iou


def augment_pair(image: np.ndarray, mask: np.ndarray):
    if np.random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
    if np.random.random() < 0.5:
        image = np.ascontiguousarray(image[::-1, :])
        mask = np.ascontiguousarray(mask[::-1, :])

    height, width = mask.shape[:2]
    if min(height, width) > 32 and np.random.random() < 0.7:
        cut_h = max(1, int(height * 0.08))
        cut_w = max(1, int(width * 0.08))
        top = int(np.random.randint(0, cut_h + 1))
        bottom = int(np.random.randint(0, cut_h + 1))
        left = int(np.random.randint(0, cut_w + 1))
        right = int(np.random.randint(0, cut_w + 1))
        y2 = height - bottom
        x2 = width - right
        if y2 - top > 16 and x2 - left > 16 and mask[top:y2, left:x2].sum() > 0:
            image = image[top:y2, left:x2]
            mask = mask[top:y2, left:x2]
    return image, mask


class CropSegDataset(Dataset):
    """读取 make_dataset 写出的外扩裁剪图和实例掩码。"""

    def __init__(self, root: Path, image_size: int, preprocess, augment: bool):
        self.image_paths = sorted((root / "images").glob("*.png"))
        self.mask_dir = root / "masks"
        self.image_size = image_size
        self.preprocess = preprocess
        self.augment = augment
        if not self.image_paths:
            raise FileNotFoundError(f"分割样本为空: {root / 'images'}，请先运行 python make_dataset.py")
        missing = [p.name for p in self.image_paths if not (self.mask_dir / p.name).exists()]
        if missing:
            raise FileNotFoundError(f"缺少掩码 {missing[:3]}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.mask_dir / path.name), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.augment:
            image, mask = augment_pair(image, mask)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        array = np.ascontiguousarray(self.preprocess(image))
        tensor = torch.from_numpy(array).permute(2, 0, 1).float()
        target = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return tensor, target


def run_epoch(model, loader, optimizer, loss_fn, device, train: bool):
    model.train(train)
    losses = []
    tps, fps, fns, tns = [], [], [], []
    for images, masks in tqdm(loader, desc="train" if train else "val", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = loss_fn(logits, masks)
            if train:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach()))
        tp, fp, fn, tn = batch_iou(logits.detach(), masks)
        tps.append(tp.cpu())
        fps.append(fp.cpu())
        fns.append(fn.cpu())
        tns.append(tn.cpu())

    tp = torch.cat(tps, dim=0)
    fp = torch.cat(fps, dim=0)
    fn = torch.cat(fns, dim=0)
    tn = torch.cat(tns, dim=0)
    return float(np.mean(losses)), dataset_iou(tp, fp, fn, tn)


def main() -> None:
    parser = argparse.ArgumentParser(description="步骤3：训练 SMP UNet++ 分割模型")
    parser.add_argument("--train-dir", type=Path, default=config.SEG_TRAIN_DIR)
    parser.add_argument("--val-dir", type=Path, default=config.SEG_VAL_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=config.SEG_IMAGE_SIZE)
    parser.add_argument("--encoder", type=str, default=config.ENCODER_NAME)
    parser.add_argument("--encoder-weights", type=str, default=config.ENCODER_WEIGHTS)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if args.image_size % 32 != 0:
        raise SystemExit("UNet++ 输入边长需要能被 32 整除")

    seed_everything(args.seed)
    device_name = resolve_device(args.device)
    device = torch_device(device_name)
    meta_path = config.DATA_DIR / "meta.json"
    margin = config.BBOX_MARGIN
    if meta_path.exists():
        margin = float(json.loads(meta_path.read_text(encoding="utf-8")).get("bbox_margin", margin))

    preprocess = build_preprocess(args.encoder, args.encoder_weights)
    train_set = CropSegDataset(args.train_dir, args.image_size, preprocess, augment=True)
    val_set = CropSegDataset(args.val_dir, args.image_size, preprocess, augment=False)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(
        f"UNet++ 训练  encoder={args.encoder} weights={args.encoder_weights}  "
        f"samples train/val={len(train_set)}/{len(val_set)}  device={device}"
    )
    model = build_unetplusplus(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights,
        in_channels=config.SEG_IN_CHANNELS,
        classes=config.SEG_CLASSES,
    ).to(device)
    loss_fn = build_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = config.RUNS_DIR / "seg"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_iou = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_iou = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        val_loss, val_iou = run_epoch(model, val_loader, optimizer, loss_fn, device, train=False)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_iou": train_iou,
                "val_loss": val_loss,
                "val_iou": val_iou,
            }
        )
        print(
            f"epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_loss:.4f} train_iou={train_iou:.4f}  "
            f"val_loss={val_loss:.4f} val_iou={val_iou:.4f}"
        )
        if val_iou >= best_iou:
            best_iou = val_iou
            ckpt = {
                "state_dict": model.state_dict(),
                "encoder_name": args.encoder,
                "preprocess_pretrained": args.encoder_weights,
                "in_channels": config.SEG_IN_CHANNELS,
                "classes": config.SEG_CLASSES,
                "image_size": args.image_size,
                "threshold": config.SEG_THRESHOLD,
                "margin": margin,
                "best_iou": best_iou,
                "epoch": epoch,
            }
            torch.save(ckpt, out_dir / config.SEG_CKPT_NAME)

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"分割权重: {out_dir / config.SEG_CKPT_NAME}  best_iou={best_iou:.4f}")
    print(f"下一步: python infer.py --source {config.DATA_DIR / 'images' / 'val'}")


if __name__ == "__main__":
    main()
