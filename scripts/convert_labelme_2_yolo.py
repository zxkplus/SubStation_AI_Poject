import os
import json
import cv2
import numpy as np
from glob import glob

def find_image_file(json_dir, base_name):
    """
    根据 JSON 文件名查找对应的图片文件。
    优先使用 JSON 内部的 imagePath，如果不存在或文件不存在，则尝试常见后缀。
    """
    # 优先检查 JSON 内部记录的 imagePath
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    for ext in image_extensions:
        img_path = os.path.join(json_dir, base_name + ext)
        if os.path.exists(img_path):
            return img_path
    return None

def convert_labelme_to_yolo(json_file_path, class_names):
    """
    将单个 LabelMe JSON 文件转换为 YOLO 格式字符串列表。
    
    Args:
        json_file_path (str): LabelMe JSON 文件路径。
        class_names (list): 类别名称列表，如 ['class1', 'class2']。
        
    Returns:
        list: 每一行 YOLO 格式字符串的列表。
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取图像尺寸
    img_dir = os.path.dirname(json_file_path)
    img_base_name = os.path.splitext(os.path.basename(json_file_path))[0]
    
    # 尝试读取图片以获取尺寸
    img_path = find_image_file(img_dir, img_base_name)
    if not img_path:
        print(f"⚠️ 未找到 {img_base_name} 对应的图片文件，无法获取尺寸，跳过 {json_file_path}")
        return []

    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 无法读取图片 {img_path}，跳过。")
        return []
        
    img_h, img_w = img.shape[:2]

    yolo_lines = []

    # 用于存储已处理的形状，以便按对象分组
    # 假设 LabelMe 中，属于同一个对象的轮廓（外+内）是连续的
    current_object_shapes = []
    current_label = None

    def process_object():
        nonlocal current_object_shapes, current_label
        if not current_object_shapes:
            return
        
        # 1. 提取类别ID
        try:
            cls_id = class_names.index(current_label)
        except ValueError:
            print(f"⚠️ 警告: 类别 '{current_label}' 不在 class_names 列表中，跳过该对象。")
            return

        # 2. 构建 YOLO 格式点序列
        yolo_points = []
        
        # 处理外轮廓 (第一个 shape)
        outer_points = np.array(current_object_shapes[0])
        yolo_points.extend(outer_points)
        
        # 处理内轮廓 (后续 shapes)
        for inner_idx in range(1, len(current_object_shapes)):
            inner_points = np.array(current_object_shapes[inner_idx])
            if len(inner_points) < 3: # 无效轮廓
                continue
                
            # 获取连接点：外轮廓最后一个点 -> 内轮廓第一个点
            last_outer = outer_points[-1]
            first_inner = inner_points[0]
            
            # 添加连接线 (退化线段，逻辑连接)
            yolo_points.append(last_outer)
            yolo_points.append(first_inner)
            
            # 添加内轮廓
            yolo_points.extend(inner_points)
            
            # 添加断开线 (回到外轮廓尾，逻辑断开)
            yolo_points.append(last_outer)
        
        # 3. 归一化坐标
        yolo_points = np.array(yolo_points)
        yolo_points[:, 0] /= img_w  # 归一化 x
        yolo_points[:, 1] /= img_h # 归一化 y
        
        # 4. 展平并生成行
        flattened = [cls_id] + yolo_points.flatten().tolist()
        line = " ".join(map(str, flattened))
        yolo_lines.append(line)
        
        # 清空
        current_object_shapes = []
        current_label = None

    # 遍历 JSON 中的所有 shapes
    for shape in data['shapes']:
        if shape['shape_type'] not in ['polygon', 'polyline']:
            continue
            
        points = shape['points']
        label = shape['label']
        
        # 简单的逻辑：如果当前没有对象，或者标签变了，先处理旧对象
        if current_label is None:
            current_label = label
        elif current_label != label:
            process_object()
            current_label = label
        
        current_object_shapes.append(points)
    
    # 处理最后一个对象
    process_object()

    return yolo_lines

def main():
    # =================== 配置区 ===================
    # 1. 设置包含 JSON 和 图片 的文件夹路径
    folder_path = "/media/industai/data1/SEG_DATA/part/test_beijing_2/1"  # 默认为脚本所在目录，可修改为绝对路径，如 "C:/your/data/"
    
    # 2. 设置类别名称列表 (必须与你的数据集类别顺序一致)
    # 例如：如果你的标注只有 "触头、导电臂"，则写成：
    class_names = ["触头、导电臂"] 
    # 如果有多个类别，请按顺序添加，如 ["class1", "class2", "class3"]
    # ============================================
    
    # 获取所有 JSON 文件
    json_files = glob(os.path.join(folder_path, "*.json"))
    
    if not json_files:
        print("❌ 未在指定文件夹找到 JSON 文件。")
        return

    converted_count = 0
    for json_file in json_files:
        try:
            yolo_lines = convert_labelme_to_yolo(json_file, class_names)
            
            if yolo_lines:
                # 生成同名 txt 文件
                txt_file = os.path.splitext(json_file)[0] + ".txt"
                with open(txt_file, 'w', encoding='utf-8') as f:
                    for line in yolo_lines:
                        f.write(line + '\n')
                print(f"✅ 已转换: {os.path.basename(json_file)} -> {os.path.basename(txt_file)}")
                converted_count += 1
            else:
                # 生成空文件 (表示无标注)
                txt_file = os.path.splitext(json_file)[0] + ".txt"
                open(txt_file, 'w').close()
                print(f"ℹ️  无标注/跳过: {os.path.basename(json_file)}")
                
        except Exception as e:
            print(f"❌ 转换失败 {json_file}: {e}")

    print(f"\n--- 转换完成 ---")
    print(f"成功处理 {converted_count} 个文件。")

if __name__ == "__main__":
    main()