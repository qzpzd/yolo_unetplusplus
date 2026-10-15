# YOLO + UNet++ 两阶段分割

整图先用 YOLO 做目标检测，再把检测框按比例外扩、裁成 ROI，送入 [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) 的 UNet++ 做实例分割，最后把掩码贴回原图。检测和分割分开训练。推理阶段两个模型都走 TensorRT FP16 engine，并打印各阶段耗时和总延时。

类别是自制几何形状：`circle`、`rectangle`、`triangle`。检测框外扩比例是 0.15，分割输入 256，检测输入 640。

推理时的置信度和 IoU 不写进 engine。带 NMS 的模型（如 YOLO11）在 GPU 上做 NMS；YOLO26 这类端到端模型只按置信度过滤。细节、导出约定和耗时见 [docs/推理方案.md](docs/推理方案.md)。

## 环境

需要 NVIDIA GPU。本仓库在 RTX 4060 Laptop、CUDA 13.1、PyTorch 2.6（cu124）、Ultralytics 8.4、TensorRT 10.16 上跑通。

```text
python -m pip install -r requirements.txt
python -m pip install "tensorrt==10.16.1.11" --extra-index-url https://pypi.nvidia.com
```

TensorRT 轮子在 `pypi.nvidia.com`。只用国内镜像时，哈希可能对不上。

## 目录

| 文件 | 作用 |
|---|---|
| `make_dataset.py` | 生成圆、矩形、三角形数据集，同时写 YOLO 标签和 UNet++ 裁剪样本 |
| `train_yolo.py` | 单独训练 YOLO |
| `train_seg.py` | 单独训练 UNet++（ResNet34、ImageNet 预处理、Dice + SoftBCE） |
| `export_engines.py` | 把两个模型导出成 TensorRT FP16 engine |
| `infer.py` | 两阶段推理，打印各阶段耗时 |
| `yolo_runtime.py` | YOLO engine 的 GPU 前处理和 GPU 过滤 |
| `seg_engine.py` | UNet++ engine 推理 |
| `data/` | 已生成的数据集，60 张训练、20 张验证 |
| `runs/yolo/exp/weights/best.pt` | 微调后的 YOLO11n |
| `runs/seg/best.pth` | 微调后的 UNet++ |

`*.engine` 和 `*.onnx` 不在仓库里。engine 只能在导出它的 GPU 和 TensorRT 版本上使用，克隆后需要在本机重新导出。

## 使用

数据集已经在 `data/` 中。若要重做：

```text
python make_dataset.py
```

默认 60 张训练图、20 张验证图、640×640、随机种子 42。

训练（仓库里已有对应权重，可跳过）：

```text
python train_yolo.py
python train_seg.py
```

导出 engine 并推理。`--conf` 和 `--iou` 在本次推理生效：

```text
python export_engines.py
python infer.py --source data/images/val/0006.jpg --conf 0.25 --iou 0.7
```

只重建其中一个 engine：

```text
python export_engines.py --only yolo --force
python export_engines.py --only seg --force
```

输出在 `runs/infer/`：可视化 JPEG、掩码 PNG，以及各阶段耗时 JSON。
