import base64
import binascii
from pathlib import Path
from typing import List, Optional
import warnings

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from service.schemas import ROI


class YOLOMaskService:
    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        conf_threshold: float = 0.25,
        img_size: int = 640,
    ):
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        
        # 自动检测设备可用性
        self.device = self._select_device(device)
        
        self.model = YOLO(str(self.weights_path))

    @staticmethod
    def _select_device(requested_device: str) -> str:
        """
        智能选择设备：如果请求的设备不可用，自动回退到可用设备
        
        Args:
            requested_device: 用户请求的设备 ('cpu', 'cuda', '0', 'cuda:0', etc.)
        
        Returns:
            实际使用的设备
        """
        # 如果请求的是 CPU，直接使用
        if requested_device.lower() == 'cpu':
            return 'cpu'
        
        # 检查 CUDA 是否可用
        cuda_available = torch.cuda.is_available()
        
        if not cuda_available:
            warnings.warn(
                f"请求使用 {requested_device}，但 CUDA 不可用。自动切换到 CPU 模式。",
                UserWarning
            )
            return 'cpu'
        
        # CUDA 可用，验证请求的设备
        if requested_device.lower() in ['cuda', '0', 'cuda:0']:
            return '0'  # 使用第一个 GPU
        
        # 其他情况，直接使用用户请求的设备
        return requested_device

    @staticmethod
    def decode_base64_image(image_base64: str) -> np.ndarray: