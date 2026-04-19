"""
诊断脚本：检查YOLO模型是否输出mask
用于排查为什么val_batch1_pred.jpg中没有mask
"""

import sys
import torch
from pathlib import Path
from ultralytics import YOLO
import yaml

print("=" * 60)
print("YOLO模型诊断 - Mask输出检查")
print("=" * 60)

# 配置
weights_path = sys.argv[1] if len(sys.argv) > 1 else "path/to/your/weights.pt"
data_config_path = sys.argv[2] if len(sys.argv) > 2 else "path/to/your/data.yaml"

print(f"模型权重: {weights_path}")
print(f"数据配置: {data_config_path}")
print()

# 加载模型
print("1. 加载模型...")
model = YOLO(weights_path)

# 检查模型信息
print("\n2. 检查模型信息...")
print(f"模型类型: {type(model)}")
print(f"模型任务: {model.task}")
print(f"模型类别: {model.names}")

# 检查模型架构
print("\n3. 检查模型架构...")
if hasattr(model, 'model'):
    print(f"模型对象: {type(model.model)}")
    print(f"模型参数: {sum(p.numel() for p in model.model.parameters()):,}")

# 检查是否有mask头
print("\n4. 检查是否有mask头...")
has_mask = False
if hasattr(model.model, 'named_modules'):
    for name, module in model.model.named_modules():
        if 'mask' in name.lower() or 'seg' in name.lower():
            print(f"  发现mask相关模块: {name} - {type(module)}")
            has_mask = True

if not has_mask:
    print("  警告: 未发现mask相关模块！")
    print("  这可能意味着模型被错误地加载为检测模型而非分割模型")

# 加载数据配置
print("\n5. 加载数据配置...")
with open(data_config_path, 'r') as f:
    data_config = yaml.safe_load(f)

val_images_dir = Path(data_config['path']) / data_config['val'] / 'images'
image_files = list(val_images_dir.glob('*.jpg')) + list(val_images_dir.glob('*.png'))

if len(image_files) == 0:
    print(f"错误: 未找到验证图片: {val_images_dir}")
    sys.exit(1)

print(f"验证集图片数量: {len(image_files)}")

# 推理一张图片
print("\n6. 推理测试...")
test_image = image_files[0]
print(f"测试图片: {test_image}")

results = model(str(test_image), verbose=False)

if len(results) == 0:
    print("错误: 模型未返回任何结果")
    sys.exit(1)

result = results[0]

# 检查结果中是否有mask
print("\n7. 检查推理结果...")
print(f"结果类型: {type(result)}")
print(f"结果属性: {dir(result)}")

# 检查boxes
if hasattr(result, 'boxes') and result.boxes is not None:
    print(f"\n检测框:")
    print(f"  数量: {len(result.boxes)}")
    print(f"  属性: {dir(result.boxes)}")
    if len(result.boxes) > 0:
        print(f"  第一个框的置信度: {result.boxes.conf[0]}")
        print(f"  第一个框的类别: {result.boxes.cls[0]}")
else:
    print("\n未发现检测框")

# 检查masks（关键！）
print("\n8. 检查Mask输出（关键）...")
if hasattr(result, 'masks'):
    if result.masks is not None:
        print("  ✓ 发现mask输出！")
        print(f"  Mask数量: {len(result.masks)}")
        print(f"  Mask数据形状: {result.masks.data.shape}")
        print(f"  Mask数据类型: {result.masks.data.dtype}")

        # 检查mask内容
        if len(result.masks) > 0:
            first_mask = result.masks.data[0].cpu().numpy()
            print(f"  第一个mask的统计:")
            print(f"    形状: {first_mask.shape}")
            print(f"    最小值: {first_mask.min()}")
            print(f"    最大值: {first_mask.max()}")
            print(f"    均值: {first_mask.mean():.4f}")
            print(f"    非零像素数: {(first_mask > 0.5).sum()}")

            # 可视化第一个mask
            import cv2
            import matplotlib.pyplot as plt
            matplotlib.use('Agg')

            mask_vis = (first_mask * 255).astype(np.uint8)
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(first_mask, cmap='gray')
            plt.title('Raw Mask')
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.imshow((first_mask > 0.5).astype(np.uint8), cmap='gray')
            plt.title('Binary Mask (threshold=0.5)')

            plt.tight_layout()
            plt.savefig('/tmp/mask_diagnosis.png', dpi=150)
            print(f"\n  已保存mask可视化: /tmp/mask_diagnosis.png")
    else:
        print("  ✗ 结果中有masks属性，但为None！")
        print("  这是问题的根源！")
else:
    print("  ✗ 结果中没有masks属性！")
    print("  这是问题的根源！")

# 检查模型任务的正确性
print("\n9. 诊断总结...")
print(f"模型任务: {model.task}")
print(f"预期任务: segment（实例分割）")

if model.task != 'segment':
    print("\n⚠️  问题诊断:")
    print("  模型任务不是'segment'，而是: ", model.task)
    print("\n  可能的原因:")
    print("  1. 训练时未指定task='segment'")
    print("  2. 模型被错误地保存为检测模型")
    print("  3. 加载模型时未指定任务类型")
    print("\n  解决方案:")
    print("  检查训练脚本中的task参数，确保设置为'segment'")
else:
    print("\n✓ 模型任务正确")
    if hasattr(result, 'masks') and result.masks is not None:
        print("✓ Mask输出正常")
        print("\n问题可能在验证可视化阶段，Ultralytics默认的验证可视化可能不显示mask")
        print("建议使用 validate_with_mask.py 脚本进行自定义可视化")
    else:
        print("✗ Mask输出异常")

print("\n" + "=" * 60)
