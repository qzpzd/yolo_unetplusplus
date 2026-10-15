"""UNet++ 的 TensorRT engine 推理。输入输出保持 FP32，网络内部按导出时的 FP16 计算。"""

import json
from pathlib import Path

import torch


class SegEngine:
    def __init__(self, engine_path: Path, meta_path: Path):
        import tensorrt as trt

        if not engine_path.exists():
            raise SystemExit(f"找不到分割 engine {engine_path}，请先运行: python export_engines.py")
        if not meta_path.exists():
            raise SystemExit(f"找不到分割 engine 配置 {meta_path}，请先运行: python export_engines.py")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.input_name = self.meta["input_name"]
        self.output_name = self.meta["output_name"]
        self.max_batch = int(self.meta["max_batch"])
        self.image_size = int(self.meta["image_size"])
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise SystemExit(f"无法加载 TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("分割输入必须是 NCHW")
        images = images.contiguous()
        outputs = []
        for start in range(0, images.shape[0], self.max_batch):
            chunk = images[start : start + self.max_batch]
            outputs.append(self._forward(chunk))
        return torch.cat(outputs, dim=0)

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        shape = tuple(int(v) for v in images.shape)
        if not self.context.set_input_shape(self.input_name, shape):
            raise RuntimeError(f"设置输入形状失败: {shape}")
        out_shape = tuple(int(v) for v in self.context.get_tensor_shape(self.output_name))
        output = torch.empty(out_shape, device=images.device, dtype=torch.float32)
        self.context.set_tensor_address(self.input_name, images.data_ptr())
        self.context.set_tensor_address(self.output_name, output.data_ptr())
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return output
