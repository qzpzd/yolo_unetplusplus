"""设备选择与随机种子。"""

import random

import numpy as np
import torch


def resolve_device(name: str) -> str:
    """返回 Ultralytics 可接受的设备字符串：'cpu' 或 '0'。"""
    if name == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return name


def torch_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name.isdigit():
        return torch.device(f"cuda:{name}")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
