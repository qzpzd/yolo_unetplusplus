"""把微调后的 YOLO 和 UNet++ 都导出成 TensorRT FP16 engine。

用法：
    python export_engines.py
    python export_engines.py --force
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from seg_engine import SegEngine
from smp_unet import build_unetplusplus


def export_yolo(force: bool) -> Path:
    from ultralytics import YOLO

    if not config.YOLO_BEST.exists():
        raise SystemExit(f"找不到 {config.YOLO_BEST}，请先运行: python train_yolo.py")
    if config.YOLO_ENGINE.exists() and not force:
        print(f"YOLO engine 已存在: {config.YOLO_ENGINE}")
        return config.YOLO_ENGINE
    model = YOLO(str(config.YOLO_BEST))
    end2end = bool(getattr(model.model, "end2end", False))
    if end2end:
        print("检测到端到端模型（无 NMS，如 YOLO26）。engine 不嵌入 NMS，推理时在 GPU 上按 conf 过滤。")
    else:
        print("检测到带 NMS 的检测模型。engine 只含网络，推理时在 GPU 上按本次 conf/iou 做 NMS。")
    exported = model.export(
        format="engine",
        imgsz=config.IMG_SIZE,
        quantize=16,
        device=0,
        nms=False,
        verbose=False,
    )
    engine = Path(exported)
    names = [model.names[index] for index in sorted(model.names)]
    meta = {
        "end2end": end2end,
        "names": names,
        "imgsz": config.IMG_SIZE,
        "nms_in_graph": False,
    }
    config.YOLO_ENGINE_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"YOLO engine: {engine}  end2end={end2end}")
    return engine


def export_unet(force: bool) -> Path:
    import tensorrt as trt

    if not config.SEG_BEST.exists():
        raise SystemExit(f"找不到 {config.SEG_BEST}，请先运行: python train_seg.py")
    if config.SEG_ENGINE.exists() and config.SEG_ENGINE_META.exists() and not force:
        print(f"UNet++ engine 已存在: {config.SEG_ENGINE}")
        return config.SEG_ENGINE

    ckpt = torch.load(config.SEG_BEST, map_location="cpu", weights_only=False)
    image_size = int(ckpt["image_size"])
    model = build_unetplusplus(
        encoder_name=ckpt["encoder_name"],
        encoder_weights=None,
        in_channels=int(ckpt["in_channels"]),
        classes=int(ckpt["classes"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    onnx_path = config.SEG_ENGINE.with_suffix(".onnx")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, image_size, image_size)
    print(f"导出 UNet++ ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    print(f"构建 UNet++ TensorRT FP16 engine: {config.SEG_ENGINE}")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        details = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise SystemExit(f"ONNX 解析失败:\n{details}")

    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if builder.platform_has_fast_fp16:
        trt_config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    spatial = (3, image_size, image_size)
    profile.set_shape(
        "images",
        (1, *spatial),
        (4, *spatial),
        (config.SEG_MAX_BATCH, *spatial),
    )
    trt_config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, trt_config)
    if serialized is None:
        raise SystemExit("TensorRT 构建 UNet++ engine 失败")
    config.SEG_ENGINE.write_bytes(serialized)

    meta = {
        "image_size": image_size,
        "margin": float(ckpt.get("margin", config.BBOX_MARGIN)),
        "threshold": float(ckpt.get("threshold", config.SEG_THRESHOLD)),
        "encoder_name": ckpt["encoder_name"],
        "input_name": "images",
        "output_name": "logits",
        "max_batch": config.SEG_MAX_BATCH,
        "precision": "fp16",
    }
    config.SEG_ENGINE_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with torch.inference_mode():
        sample = torch.randn(2, 3, image_size, image_size)
        reference = model(sample)
    engine = SegEngine(config.SEG_ENGINE, config.SEG_ENGINE_META)
    compared = engine(sample.cuda()).cpu()
    diff = float((reference - compared).abs().max())
    print(f"UNet++ engine: {config.SEG_ENGINE}  与 PyTorch logits 最大误差 {diff:.5f}")
    return config.SEG_ENGINE


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 YOLO 与 UNet++ 的 TensorRT engine")
    parser.add_argument("--force", action="store_true", help="已存在时重新构建")
    parser.add_argument("--only", choices=("all", "yolo", "seg"), default="all")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("导出 TensorRT engine 需要 CUDA GPU")
    yolo_engine = config.YOLO_ENGINE
    seg_engine = config.SEG_ENGINE
    if args.only in {"all", "seg"}:
        seg_engine = export_unet(args.force)
    if args.only in {"all", "yolo"}:
        yolo_engine = export_yolo(args.force)
    print(f"完成。推理请使用:\n  python infer.py --yolo {yolo_engine} --seg {seg_engine}")


if __name__ == "__main__":
    main()
