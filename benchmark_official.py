"""按官网测法计时，并和公布数字对照。

YOLO11（https://docs.ultralytics.com/models/yolo11）:
- GPU 列是 TensorRT 10、640、batch=1、只计模型前向。YOLO11n 公布值 1.5 ms（T4）。
- CPU 列是 ONNX。YOLO11n 公布值 56.1 ms。
- 本地复现使用 ultralytics.utils.benchmarks.ProfileModels，和速度表同一条导出、计时路径。

UNet++:
- segmentation_models_pytorch 文档没有速度表。
- 仓库测速用纯前向：ResNet34、128x128、batch=100、CUDA 同步后统计 images/s。
  Tesla T4 上 UNet++ 为 446 images/s。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

YOLO_OFFICIAL_TRT_MS = 1.5
YOLO_OFFICIAL_ONNX_CPU_MS = 56.1
UNET_OFFICIAL_IPS = 446.0  # ResNet34, 128, batch 100, Tesla T4


def smp_throughput(batch_size: int, resolution: int, repeats: int = 30, warmup: int = 10):
    """与 SMP 仓库测速相同：eval、CUDA、只计 model(x)，用 CUDA event 计时。"""
    import segmentation_models_pytorch as smp

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).cuda().eval()
    images = torch.randn(batch_size, 3, resolution, resolution, device="cuda")
    starter, ender = torch.cuda.Event(True), torch.cuda.Event(True)
    with torch.inference_mode():
        for _ in range(warmup):
            model(images)
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            starter.record()
            model(images)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))
    batch_ms = float(np.mean(times))
    images_per_s = batch_size / (batch_ms / 1000.0)
    per_image_ms = batch_ms / batch_size
    del model, images
    torch.cuda.empty_cache()
    return batch_ms, per_image_ms, images_per_s


def run_smp() -> None:
    print("=== UNet++ 按 SMP 仓库测速协议（纯前向）===")
    print(f"对照值: Tesla T4, ResNet34, 128x128, batch=100, {UNET_OFFICIAL_IPS:.0f} images/s")
    for batch_size, resolution in ((100, 128), (4, 256), (1, 256)):
        try:
            batch_ms, per_image_ms, images_per_s = smp_throughput(batch_size, resolution)
        except RuntimeError as exc:
            print(f"  batch={batch_size} size={resolution} 失败: {exc}")
            torch.cuda.empty_cache()
            continue
        ratio = images_per_s / UNET_OFFICIAL_IPS
        print(
            f"  batch={batch_size:<3} size={resolution}  "
            f"batch={batch_ms:.2f} ms  每张={per_image_ms:.3f} ms  "
            f"{images_per_s:.1f} images/s  相对对照 {ratio:.2f}x"
        )


def run_yolo(min_time: float) -> None:
    from ultralytics import YOLO
    from ultralytics.utils.benchmarks import ProfileModels, benchmark

    weights = "yolo11n.pt"
    print("=== YOLO11n 按 Ultralytics 速度表路径 ===")
    print(f"对照值: T4 TensorRT10 {YOLO_OFFICIAL_TRT_MS} ms, CPU ONNX {YOLO_OFFICIAL_ONNX_CPU_MS} ms, 640, 只计前向")

    print("\n[PyTorch FP16] benchmark(format='-', quantize=16, data=coco8.yaml, imgsz=640)")
    pytorch_df = benchmark(
        model=weights,
        data="coco8.yaml",
        imgsz=640,
        quantize=16,
        device="0",
        format="-",
    )
    print(pytorch_df)

    try:
        import tensorrt  # noqa: F401
    except ImportError:
        print("未安装 TensorRT，跳过官网 GPU 列（engine）复现。")
        return

    print("\n[ONNX CPU + TensorRT FP16] ProfileModels，与速度表同一计时字段 speed['inference']")
    profiler = ProfileModels(
        [weights],
        imgsz=640,
        quantize=16,
        trt=True,
        num_warmup_runs=10,
        num_timed_runs=50,
        min_time=min_time,
        device="0",
    )
    results = profiler.run()
    for row in results:
        onnx_ms = row.get("onnx_time") or row.get("cpu_onnx")
        print(row)
    # 再单独打一次本机微调权重的 TensorRT 前向，架构同为 YOLO11n
    trained = config.YOLO_BEST
    if trained.exists():
        print(f"\n[本项目微调权重] {trained}")
        model = YOLO(str(trained))
        engine = model.export(format="engine", imgsz=640, quantize=16, device=0, verbose=False)
        engine_model = YOLO(engine)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(10):
            engine_model(dummy, imgsz=640, verbose=False)
        samples = []
        for _ in range(50):
            result = engine_model(dummy, imgsz=640, verbose=False)
            samples.append(float(result[0].speed["inference"]))
        mean = float(np.mean(samples))
        print(f"  TensorRT FP16 前向 {mean:.2f} ms，官网 T4 为 {YOLO_OFFICIAL_TRT_MS} ms，比值 {mean / YOLO_OFFICIAL_TRT_MS:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description="按官网协议测速并对照公布值")
    parser.add_argument("--part", choices=("all", "yolo", "smp"), default="all")
    parser.add_argument("--min-time", type=float, default=15.0, help="YOLO TensorRT 计时最少秒数")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    if args.part in {"all", "smp"}:
        run_smp()
    if args.part in {"all", "yolo"}:
        run_yolo(args.min_time)


if __name__ == "__main__":
    main()
