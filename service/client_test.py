import sys
import os
import subprocess
import time
import signal
import atexit
from pathlib import Path
# 将项目根目录添加到Python路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

"""
SubStation AI 客户端使用示例

演示如何以编程方式调用推理服务
"""

from service.client_app import InferenceClient, ResultRenderer
from PIL import Image
import json
import requests


# 全局变量存储服务进程
_service_process = None


def start_inference_service():
    """启动推理服务"""
    global _service_process
    
    print("🚀 正在启动推理服务...")
    
    # 检查服务是否已经在运行
    try:
        response = requests.get("http://localhost:8000/docs", timeout=2)
        if response.status_code == 200:
            print("✅ 推理服务已在运行")
            return True
    except requests.exceptions.RequestException:
        pass  # 服务未运行，继续启动
    
    # 启动服务进程
    cmd = [
        sys.executable, "-m", "uvicorn",
        "service.app:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ]
    
    try:
        _service_process = subprocess.Popen(
            cmd,
            cwd=project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # 等待服务启动
        for i in range(10):
            time.sleep(1)
            try:
                response = requests.get("http://localhost:8000/docs", timeout=2)
                if response.status_code == 200:
                    print("✅ 推理服务启动成功")
                    return True
            except requests.exceptions.RequestException:
                continue
        
        print("❌ 推理服务启动超时")
        stop_inference_service()
        return False
        
    except Exception as e:
        print(f"❌ 启动推理服务失败: {e}")
        return False


def stop_inference_service():
    """停止推理服务"""
    global _service_process
    
    if _service_process and _service_process.poll() is None:
        print("🛑 正在停止推理服务...")
        _service_process.terminate()
        try:
            _service_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _service_process.kill()
        _service_process = None
        print("✅ 推理服务已停止")


def cleanup_service():
    """清理服务进程（程序退出时自动调用）"""
    stop_inference_service()


# 注册退出清理函数
atexit.register(cleanup_service)


def test_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("示例1: 基本推理调用")
    print("=" * 60)
    
    # 启动服务
    if not start_inference_service():
        print("❌ 无法启动推理服务，测试终止")
        return
    
    # 创建客户端
    client = InferenceClient(base_url="http://localhost:8000")
    
    # 检查测试图片是否存在
    test_image_path = "test_data/part2/test1.jpeg"
    if not os.path.exists(test_image_path):
        print(f"⚠️  测试图片不存在: {test_image_path}")
        print("💡 请确保测试图片存在，或修改为有效的图片路径")
        # 询问用户是否继续
        user_input = input("是否继续使用示例图片路径进行测试? (y/n): ").strip().lower()
        if user_input != 'y':
            return
        else:
            print("⚠️  注意: 以下测试可能会因为图片不存在而失败")
    
    # 检查权重文件是否存在
    weights_path = "/home/industai/workspace/SubStation_AI_Poject/runs/segment/runs2/train/exp/weights/best.pt"
    if not os.path.exists(weights_path):
        print(f"⚠️  权重文件不存在: {weights_path}")
        print("💡 请确保训练好的模型权重文件存在，或修改为有效的权重路径")
        # 提供默认权重选项
        default_weights = "yolov8n-seg.pt"
        print(f"🔄 将使用默认权重: {default_weights}")
        weights_path = default_weights
    
    # 执行推理
    try:
        result = client.predict(
            image_path=test_image_path,
            roi={"x1": 434, "y1": 115, "x2": 599, "y2": 650},
            weights_path=weights_path,
            conf_threshold=0.25,
            img_size=1024,
            device="cuda"
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
        
        # 添加结果可视化，在原图上画出矩形框和轮廓
        try:
            # 加载原始图片
            original_image = Image.open(test_image_path)
            
            # 使用ResultRenderer绘制结果
            result_image = ResultRenderer.draw_results_on_image(original_image, result)
            
            # 生成输出文件名
            output_filename = f"visualization_result_{int(time.time())}.jpg"
            
            # 保存可视化结果
            result_image.save(output_filename)
            print(f"\n✅ 结果可视化完成！已保存到: {output_filename}")
            
            # 可选：显示图片（如果在支持的环境中）
            try:
                result_image.show()
                print("🖼️  已尝试在默认图片查看器中打开结果")
            except:
                print("ℹ️  无法自动打开图片查看器，请手动查看保存的文件")
                
        except Exception as e:
            print(f"❌ 可视化失败: {e}")
            print("💡 可能的原因:")
            print("   - 图片路径不正确")
            print("   - PIL/Pillow库问题")
            print("   - 内存不足")
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        print("💡 可能的原因:")
        print("   - 图片路径不正确")
        print("   - 权重文件路径不正确")
        print("   - CUDA不可用（如果指定device='cuda'）")
        print("   - 服务启动失败")


def test_visualize_results():
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


def test_custom_colors():
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


def test_batch_processing():
    """批量处理示例"""
    print("\n" + "=" * 60)
    print("示例4: 批量处理多张图片")
    print("=" * 60)
    
    from pathlib import Path
    
    # 启动服务
    if not start_inference_service():
        print("❌ 无法启动推理服务，测试终止")
        return
    
    # 图片目录
    image_dir = Path("test_images")
    
    if not image_dir.exists():
        print("⚠️  测试图片目录不存在，跳过批量处理示例")
        return
    
    # 创建客户端
    client = InferenceClient(base_url="http://localhost:8000")
    
    # 获取所有图片
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    if not image_files:
        print("⚠️  测试图片目录中没有找到图片文件，跳过批量处理示例")
        return
    
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
    test_basic_usage()
#     test_visualize_results()
#     test_custom_colors()
#     test_batch_processing()
    
#     print("\n" + "=" * 60)
#     print("所有示例运行完成!")
#     print("=" * 60)
    
    # 询问是否保持服务运行
    print("\n" + "=" * 60)
    keep_running = input("是否保持推理服务运行以便后续测试? (y/n): ").strip().lower()
    if keep_running != 'y':
        stop_inference_service()
    else:
        print("💡 推理服务将继续运行，您可以手动终止程序来停止服务")