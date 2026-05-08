"""
SubStation AI 客户端使用示例

演示如何以编程方式调用推理服务
"""

from client_app import InferenceClient, ResultRenderer
from PIL import Image
import json


def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("示例1: 基本推理调用")
    print("=" * 60)
    
    # 创建客户端
    client = InferenceClient(base_url="http://localhost:8000")
    
    # 执行推理
    try:
        result = client.predict(
            image_path="test_image.jpg",  # 替换为你的图片路径
            roi={"x1": 100, "y1": 100, "x2": 500, "y2": 500},
            weights_path="yolov8n-seg.pt",
            conf_threshold=0.25,
            img_size=640,
            device="cpu"
        )
        
        # 打印结果
        print(f"图片尺寸: {result['image_width']} x {result['image_height']}")
        print(f"ROI数量: {len(result['results'])}")
        
        for i, roi_result in enumerate(result['results']):
            print(f"\nROI {i+1}:")
            print(f"  区域: ({roi_result['roi']['x1']}, {roi_result['roi']['y1']}) - "
                  f"({roi_result['roi']['x2']}, {roi_result['roi']['y2']})")
            print(f"  检测目标数: {len(roi_result['detections'])}")
            
            for j, detection in enumerate(roi_result['detections']):
                print(f"    目标 {j+1}:")
                print(f"      类别: {detection['class_id']}")
                print(f"      置信度: {detection['confidence']:.2f}")
                print(f"      边界框: {detection['bbox']}")
                print(f"      轮廓点数: {len(detection['contours'])}")
        
        # 保存结果为JSON
        with open("inference_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("\n✅ 结果已保存到 inference_result.json")
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")


def example_visualize_results():
    """结果可视化示例"""
    print("\n" + "=" * 60)
    print("示例2: 结果可视化")
    print("=" * 60)
    
    # 假设已经有推理结果
    result = {
        "image_width": 800,
        "image_height": 600,
        "results": [
            {
                "roi": {"x1": 100, "y1": 100, "x2": 500, "y2": 500},
                "detections": [
                    {
                        "bbox": [150, 150, 300, 350],
                        "confidence": 0.95,
                        "class_id": 0,
                        "contours": [
                            {
                                "points": [
                                    [150, 150], [300, 150], 
                                    [300, 350], [150, 350]
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    # 加载图片
    try:
        image = Image.open("test_image.jpg")  # 替换为你的图片路径
        
        # 绘制结果
        result_image = ResultRenderer.draw_results_on_image(image, result)
        
        # 保存结果图片
        result_image.save("result_visualization.jpg")
        print("✅ 可视化结果已保存到 result_visualization.jpg")
        
    except FileNotFoundError:
        print("⚠️  图片文件不存在，跳过可视化示例")
    except Exception as e:
        print(f"❌ 可视化失败: {e}")


def example_custom_colors():
    """自定义颜色示例"""
    print("\n" + "=" * 60)
    print("示例3: 自定义类别颜色")
    print("=" * 60)
    
    # 定义自定义颜色映射
    custom_colors = {
        0: (255, 0, 0),      # 红色 - 绝缘子
        1: (0, 255, 0),      # 绿色 - 导线
        2: (0, 0, 255),      # 蓝色 - 塔架
        3: (255, 255, 0),    # 黄色 - 其他设备
    }
    
    print("自定义颜色映射:")
    for class_id, color in custom_colors.items():
        print(f"  类别 {class_id}: RGB{color}")
    
    print("\n💡 提示: 在 ResultRenderer.draw_results_on_image() 中传入 class_colors 参数即可使用自定义颜色")


def example_batch_processing():
    """批量处理示例"""
    print("\n" + "=" * 60)
    print("示例4: 批量处理多张图片")
    print("=" * 60)
    
    from pathlib import Path
    
    # 图片目录
    image_dir = Path("test_images")
    
    if not image_dir.exists():
        print("⚠️  测试图片目录不存在，跳过批量处理示例")
        return
    
    # 创建客户端
    client = InferenceClient(base_url="http://localhost:8000")
    
    # 获取所有图片
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 批量处理
    for i, image_path in enumerate(image_files, 1):
        print(f"\n处理 [{i}/{len(image_files)}]: {image_path.name}")
        
        try:
            result = client.predict(
                image_path=str(image_path),
                roi={"x1": 0, "y1": 0, "x2": 800, "y2": 600},  # 全图ROI
                weights_path="yolov8n-seg.pt",
                conf_threshold=0.25,
                img_size=640,
                device="cpu"
            )
            
            num_detections = sum(len(r['detections']) for r in result['results'])
            print(f"  ✅ 检测到 {num_detections} 个目标")
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
    
    print("\n✅ 批量处理完成")


if __name__ == "__main__":
    print("SubStation AI 客户端使用示例\n")
    
    # 运行示例
    example_basic_usage()
    example_visualize_results()
    example_custom_colors()
    example_batch_processing()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)
