#!/usr/bin/env python3
"""
scripts/seg_render.py — 检测框裁剪 + 分割渲染管线

流程:
  1. 解析 YOLO 检测标签 (cls_name x1 y1 x2 y2, 像素绝对坐标)
  2. 从原图裁剪检测框区域
  3. 送入分割模型推理，提取掩膜
  4. 将掩膜以半透明方式渲染回原图（不同分割类不同颜色）
  5. 保留 YOLO 检测框和类名
  6. 支持 --ignore_classes 跳过指定检测类别

用法：
  python scripts/seg_render.py \
      --image_dir    /path/to/image \
      --label_dir    /path/to/label \
      --weights      runs/segment/runs_0714/train/exp/weights/last.pt \
      --output_dir   ./seg_render_output \
      --device       cuda:0 \
      --conf_threshold 0.25 \
      --img_size     640 \
      --ignore_classes "xiangti chuandongbufen"
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# ─────────────────────────────────────────────
# 14 个分割类的 BGR 调色板（半透明填充用）
# ─────────────────────────────────────────────
SEG_PALETTE = {
    0:  (255, 0,   0),    # she_bei_biao_shi — 红
    1:  (0,   255, 0),    # fa_lan           — 绿
    2:  (0,   0,   255),  # jie_di_yin_xia_xian — 蓝
    3:  (255, 255, 0),    # san_re_qi        — 青
    4:  (255, 0,   255),  # jue_yuan_zi      — 紫
    5:  (0,   255, 255),  # ji_gou_xiang     — 黄
    6:  (128, 128, 128),  # biao_ji          — 灰
    7:  (128, 0,   0),    # hu_xi_qi         — 深红
    8:  (0,   128, 0),    # jun_ya_huan      — 深绿
    9:  (0,   0,   128),  # ji_dian_qi       — 深蓝
    10: (128, 128, 0),    # yin_xian_jie_tou — 橄榄
    11: (128, 0,   128),  # peng_zhang_qi    — 紫红
    12: (0,   128, 128),  # ge_li_kai_guan   — 青绿
    13: (0,   200, 200),  # shen_suo_jie     — 橙黄
}

_DEFAULT_SEG_COLOR = (100, 100, 100)  # 未知分割类


def _generate_det_colors(n: int):
    """生成 n 个视觉上分布均匀的 BGR 颜色（用于检测框）"""
    colors = []
    golden_angle = 137.508
    for i in range(n):
        hue = (i * golden_angle) % 180
        # OpenCV HSV: H [0,179], S [0,255], V [0,255]
        hsv = np.array([[[hue, 255, 230]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


# ─────────────────────────────────────────────
# 核心管线
# ─────────────────────────────────────────────
class SegRenderPipeline:
    """检测框裁剪 → 分割推理 → 掩膜渲染管线"""

    def __init__(
        self,
        weights: str,
        device: str = "cuda:0",
        conf_threshold: float = 0.25,
        img_size: int = 640,
        ignore_classes: list | None = None,
        alpha: float = 0.4,
    ):
        self.weights = weights
        self.device = device
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.ignore_set = set(ignore_classes or [])
        self.alpha = alpha
        self._model = None

    # ── 模型懒加载 ──────────────────────────
    @property
    def model(self) -> YOLO:
        if self._model is None:
            print(f"[加载模型] {self.weights}  (device={self.device})")
            self._model = YOLO(str(self.weights))
        return self._model

    # ── 匹配图片 ↔ 标签 ──────────────────────
    def match_images_and_labels(self, image_dir: str, label_dir: str):
        """
        自动扫描 image_dir 下所有子目录，按 basename 匹配标签。
        返回 [(img_path, label_path), ...] 列表。
        """
        image_dir = Path(image_dir)
        label_dir = Path(label_dir)

        # 收集所有 label 文件 basename → label_path
        label_map = {}
        for f in sorted(label_dir.glob("*.txt")):
            stem = f.stem  # 不含扩展名的文件名
            label_map[stem] = str(f)

        # 扫描 image 子目录
        matches = []
        for subdir in sorted(image_dir.iterdir()):
            if not subdir.is_dir():
                continue
            for f in sorted(subdir.iterdir()):
                if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                stem = f.stem
                if stem in label_map:
                    matches.append((str(f), label_map[stem]))

        matches.sort(key=lambda x: x[0])
        return matches

    # ── 解析单张标签 ────────────────────────
    @staticmethod
    def parse_labels(label_path: str):
        """
        解析 YOLO 检测标签，返回 [(cls_name, x1, y1, x2, y2), ...]。
        格式：class_name x1 y1 x2 y2 (像素绝对坐标)
        """
        boxes = []
        with open(label_path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_name = parts[0]
                try:
                    coords = list(map(float, parts[1:5]))
                except ValueError:
                    continue
                x1, y1, x2, y2 = map(int, coords)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append((cls_name, x1, y1, x2, y2))
        return boxes

    # ── 单张图片推理 + 渲染 ──────────────────
    def process_image(self, img_path: str, label_path: str) -> np.ndarray | None:
        """返回渲染后的图像 (BGR)，若图片加载失败则返回 None"""
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [警告] 无法读取: {img_path}", file=sys.stderr)
            return None

        # 1. 解析检测标签
        all_boxes = self.parse_labels(label_path)
        # 过滤 ignore_classes
        boxes = [b for b in all_boxes if b[0] not in self.ignore_set]

        if not boxes:
            # 无检测框：保存原图不修改（或跳过？这里保存原图以便对比）
            return img

        det_colors = _generate_det_colors(len(self.model.names or []))

        # ── 逐框裁剪 → 分割推理 → 累积掩膜 ──
        overlay = np.zeros_like(img, dtype=np.float32)

        for cls_name, x1, y1, x2, y2 in boxes:
            crop = img[y1:y2, x1:x2]
            if crop.shape[0] < 2 or crop.shape[1] < 2:
                continue

            results = self.model.predict(
                source=crop,
                conf=self.conf_threshold,
                imgsz=self.img_size,
                device=self.device,
                verbose=False,
            )

            if not results or len(results) == 0:
                continue
            result = results[0]
            if not hasattr(result, "masks") or result.masks is None:
                continue

            masks_data = result.masks.data
            if masks_data is None or len(masks_data) == 0:
                continue

            crop_h, crop_w = crop.shape[:2]

            for i in range(len(masks_data)):
                # mask shape: (1, H, W) or (H, W)
                mask_np = masks_data[i].cpu().numpy().squeeze()
                if mask_np.ndim != 2:
                    continue

                # resize 到 crop 尺寸，保持坐标对齐
                if mask_np.shape != (crop_h, crop_w):
                    mask_np = cv2.resize(mask_np, (crop_w, crop_h),
                                         interpolation=cv2.INTER_NEAREST)

                # 阈值化
                mask_bin = (mask_np > 0.5).astype(np.uint8)

                # 获取该 mask 对应的分割类 ID
                cls_id = 0
                if hasattr(result, "boxes") and result.boxes is not None and i < len(result.boxes):
                    cls_id = int(result.boxes.cls[i].cpu().item())

                seg_color = SEG_PALETTE.get(cls_id, _DEFAULT_SEG_COLOR)

                # 在 overlay 上着色
                roi_mask = mask_bin > 0
                overlay[y1:y2, x1:x2][roi_mask] = seg_color

        # ── 渲染 ─────────────────────────────
        # 先混合掩膜，再画检测框（框在顶层）
        result_img = img.copy().astype(np.float32)

        # 半透明混合掩膜
        mask_overlay = (overlay > 0).any(axis=2)
        if mask_overlay.any():
            alpha = self.alpha
            result_img[mask_overlay] = (
                result_img[mask_overlay] * (1 - alpha)
                + overlay[mask_overlay] * alpha
            )

        result_img = result_img.astype(np.uint8)

        # ── 绘制 YOLO 检测框 + 类名 ──────────
        # 为每个检测类分配颜色（按在 boxes 中首次出现顺序）
        cls_order = {}
        for b in boxes:
            if b[0] not in cls_order:
                cls_order[b[0]] = len(cls_order)
        det_color_map = {c: det_colors[i % len(det_colors)]
                         for c, i in cls_order.items()}

        for cls_name, x1, y1, x2, y2 in boxes:
            color = det_color_map.get(cls_name, (0, 255, 0))
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 3)

            # 类名文字背景
            (tw, th), baseline = cv2.getTextSize(
                cls_name, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
            )
            label_y1 = max(y1 - th - 6, 0)
            cv2.rectangle(
                result_img,
                (x1, label_y1),
                (x1 + tw + 6, y1),
                color,
                -1,
            )
            cv2.putText(
                result_img,
                cls_name,
                (x1 + 3, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return result_img

    # ── 完整管线 ─────────────────────────────
    def run(self, image_dir: str, label_dir: str, output_dir: str):
        """
        执行完整管线：匹配 → 逐图处理 → 保存渲染结果。
        以平铺方式保存到 output_dir，文件名格式 {basename}_seg.jpg
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        matches = self.match_images_and_labels(image_dir, label_dir)
        if not matches:
            print("[错误] 未找到任何图片-标签匹配对。请检查 --image_dir / --label_dir。",
                  file=sys.stderr)
            sys.exit(1)

        print(f"[开始] 共 {len(matches)} 张图片需要处理")
        print(f"       忽略检测类别: {sorted(self.ignore_set) or '(无)'}")
        print(f"       分割模型参数: conf={self.conf_threshold}, "
              f"imgsz={self.img_size}, device={self.device}")
        print(f"       输出目录: {output_dir.resolve()}")
        print()

        success = 0
        fail = 0
        total_boxes = 0

        pbar = tqdm(matches, desc="渲染进度", unit="img")
        for img_path, label_path in pbar:
            # 更新进度条描述
            basename = Path(img_path).stem
            pbar.set_postfix_str(basename[:50])

            rendered = self.process_image(img_path, label_path)
            if rendered is None:
                fail += 1
                continue

            out_name = f"{basename}_seg.jpg"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), rendered)

            # 统计本图检测框数
            boxes = [b for b in self.parse_labels(label_path)
                     if b[0] not in self.ignore_set]
            total_boxes += len(boxes)
            success += 1

        pbar.close()
        print()
        print(f"[完成] 成功: {success}  失败: {fail}  总检测框数: {total_boxes}")
        print(f"       输出目录: {output_dir.resolve()}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="检测框裁剪 → 分割推理 → 掩膜渲染管线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image_dir", type=str, required=True,
                        help="图片目录（会自动扫描子目录）")
    parser.add_argument("--label_dir", type=str, required=True,
                        help="YOLO 检测标签目录（平铺 .txt 文件）")
    parser.add_argument("--weights", type=str, required=True,
                        help="分割模型权重路径（.pt）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="渲染结果输出目录")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="推理设备 (default: cuda:0)")
    parser.add_argument("--conf_threshold", type=float, default=0.25,
                        help="分割模型置信度阈值 (default: 0.25)")
    parser.add_argument("--img_size", type=int, default=640,
                        help="分割模型推理输入尺寸 (default: 640)")
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="掩膜半透明填充透明度 (default: 0.4)")
    parser.add_argument("--ignore_classes", type=str, default="",
                        help="空格分隔的检测类名列表，跳过这些类别的检测框 "
                             "(default: 空，即处理所有)")
    return parser.parse_args(argv)


def main():
    args = parse_args()

    ignore_list = args.ignore_classes.split() if args.ignore_classes.strip() else []

    pipeline = SegRenderPipeline(
        weights=args.weights,
        device=args.device,
        conf_threshold=args.conf_threshold,
        img_size=args.img_size,
        ignore_classes=ignore_list,
        alpha=args.alpha,
    )

    pipeline.run(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
