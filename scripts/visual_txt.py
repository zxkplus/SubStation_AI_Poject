import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics.data.utils import polygons2masks  # Ultralytics 官方多边形转掩码函数

# ======================
# 1. 解析标注数据（关键步骤）
# ======================
label_str = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6"
data = list(map(float, label_str.split()))

class_id = int(data[0])  # 类别ID (0)
norm_points = np.array(data[1:]).reshape(-1, 2)  # 归一化坐标点 [N, 2]

# 验证数据结构：前4点=外轮廓，后4点=空洞
print(f"外部轮廓点数: {len(norm_points[:4])}, 空洞点数: {len(norm_points[4:])}")
print("归一化坐标:\n", norm_points)

# ======================
# 2. 生成可视化图像
# ======================
img_size = 640  # YOLO默认输入尺寸
img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 220  # 浅灰色背景

# 将归一化坐标转为图像绝对坐标
abs_points = (norm_points * img_size).astype(np.int32)

# ======================
# 3. 使用Ultralytics官方函数生成掩码
# ======================
# 注意：polygons2masks要求输入格式为 [num_instances, num_points, 2]
polygons = [norm_points[:4], norm_points[4:]]  # 封装为单实例列表
masks = polygons2masks((img_size, img_size), polygons, 1)  # 官方函数生成二进制掩码

# 提取第一个（也是唯一一个）实例的掩码
mask = masks[0]  # 形状: [640, 640], 空洞区域为False

# ======================
# 4. 可视化验证（关键：检查空洞是否被排除）
# ======================
plt.figure(figsize=(12, 4))

# 子图1: 原始标注点
plt.subplot(131)
plt.imshow(img)
plt.plot(abs_points[:4, 0], abs_points[:4, 1], 'b-', linewidth=2, label='外部轮廓')
plt.plot(abs_points[4:, 0], abs_points[4:, 1], 'r--', linewidth=2, label='空洞轮廓')
plt.scatter(abs_points[:, 0], abs_points[:, 1], c=['blue']*4 + ['red']*4, s=50)
plt.title("标注点坐标 (蓝色=外轮廓, 红色=空洞)")
plt.legend()

# 子图2: Ultralytics生成的掩码
plt.subplot(132)
plt.imshow(mask, cmap='gray_r', alpha=0.7)
plt.contour(mask, colors='red', linewidths=1, levels=[0.5])
plt.title("Ultralytics生成的掩码\n(白色=物体区域, 黑色=空洞/背景)")

# 子图3: 掩码叠加到原图
plt.subplot(133)
overlay = img.copy()
overlay[mask] = [0, 165, 255]  # 橙色填充物体区域
overlay = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)  # 透明叠加
plt.imshow(overlay)
plt.contour(mask, colors='red', linewidths=1, levels=[0.5])
plt.title("掩码叠加效果\n(红色轮廓=物体边界, 中心黑色=空洞)")

plt.tight_layout()
plt.savefig("yolo_hollow_demo.png", dpi=150, bbox_inches='tight')
print("\n✅ 生成结果已保存至: yolo_hollow_demo.png")
plt.show()

# ======================
# 5. 关键验证：检查空洞区域是否被正确排除
# ======================
hole_center = (0.5, 0.5)  # 空洞理论中心 (0.4~0.6范围)
hole_pixel = (int(hole_center[1] * img_size), int(hole_center[0] * img_size))

print(f"\n🔍 验证空洞中心点 ({hole_pixel[1]}, {hole_pixel[0]}) 的掩码值: {mask[hole_pixel]}")
print("→ 若输出 'False'，表示空洞区域被正确排除（符合预期）")
print("→ 若输出 'True'，表示空洞未生效（标注格式错误）")