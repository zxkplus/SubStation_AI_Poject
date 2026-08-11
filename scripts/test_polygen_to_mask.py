import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import cv2

# 直接从 Ultralytics 源码导入官方 polygon2mask 函数
# 注意：需先安装 ultralytics (pip install ultralytics)
from ultralytics.data.utils import polygon2mask

# ======================
# 1. 构造测试数据：带空洞的多边形
# ======================
# 图像尺寸 (H, W)
imgsz = (200, 200)

# 外轮廓：顺时针矩形 (关键：顺时针定义主体区域)
outer = np.array([
    [30, 30], [170, 30], [170, 170], [30, 170]  # 顺时针
], dtype=np.float32)

# 内轮廓：逆时针矩形 (关键：逆时针定义空洞)
inner = np.array([
    [70, 70], [130, 70], [130, 130], [70, 130]  # 逆时针
], dtype=np.float32)

# Ultralytics 存储格式：所有点拼接成一个列表（无显式空洞标记）
polygons = [outer, inner]  # 注意：这是官方要求的输入格式 [外轮廓, 内轮廓]

# ======================
# 2. 调用官方 polygon2mask 生成掩码
# ======================
mask = polygon2mask(imgsz, polygons)  # 直接使用官方函数

# ======================
# 3. 可视化结果
# ======================
plt.figure(figsize=(12, 8))

# 子图1：原始多边形（展示拓扑关系）
ax1 = plt.subplot(1, 2, 1)
ax1.set_title("原始多边形（外轮廓顺时针 + 内轮廓逆时针）", fontsize=12)

# 绘制外轮廓（红色）
ax1.add_collection(PatchCollection([Polygon(outer, closed=True)], 
                                  facecolor='none', edgecolor='r', linewidth=2, label="外轮廓"))

# 绘制内轮廓（蓝色）
ax1.add_collection(PatchCollection([Polygon(inner, closed=True)], 
                                  facecolor='none', edgecolor='b', linewidth=2, label="内轮廓（空洞）"))

ax1.set_xlim(0, imgsz[1])
ax1.set_ylim(imgsz[0], 0)  # Y轴翻转以匹配图像坐标
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# 子图2：生成的二值掩码
ax2 = plt.subplot(1, 2, 2)
ax2.set_title("官方 polygon2mask 生成的二值掩码（空洞=0）", fontsize=12)
ax2.imshow(mask, cmap='gray_r', vmin=0, vmax=1)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.text(10, 30, "空洞区域值=0", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

# 标注关键点
ax2.plot(100, 100, 'ro', markersize=8)  # 外轮廓内点
ax2.plot(100, 100, 'r+', markersize=12)
ax2.text(110, 100, "主体区域=1", color='red')

ax2.plot(100, 90, 'bo', markersize=8)  # 空洞区域点
ax2.plot(100, 90, 'b+', markersize=12)
ax2.text(110, 90, "空洞区域=0", color='blue')

plt.tight_layout()
plt.savefig("polygon2mask_demo.png", dpi=150)
plt.show()

# ======================
# 4. 验证：用 OpenCV 从掩码重新提取轮廓
# ======================
# 查找轮廓（RETR_CCOMP 会同时提取外轮廓和内轮廓）
contours, hierarchy = cv2.findContours(
    mask.astype(np.uint8), 
    cv2.RETR_CCOMP, 
    cv2.CHAIN_APPROX_SIMPLE
)

# 打印轮廓信息（验证空洞被识别为子轮廓）
print("【轮廓提取结果】")
print(f"总轮廓数: {len(contours)}")
print(f"层级关系: {hierarchy[0]}")  # 格式: [next, prev, first_child, parent]

# 输出示例:
# 总轮廓数: 2
# 层级关系: [[-1, -1, 1, -1]]  # [0]是外轮廓，[1]是它的子轮廓（空洞）