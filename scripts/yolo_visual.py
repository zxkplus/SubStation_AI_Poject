import cv2
import numpy as np
from pathlib import Path

def visualize_polygon_annotations(image_path, label_path, label_map):
    """
    可视化多边形标注格式的YOLO数据
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    h, w = image.shape[:2]
    
    # 读取标注文件
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        values = list(map(float, line.split()))
        if len(values) < 3:  # 至少需要class_id + 一个(x,y)坐标对
            continue
            
        class_id = int(values[0])
        # 剩余的值是x,y坐标对
        coords = values[1:]
        
        # 将归一化的坐标转换为像素坐标
        points = []
        for i in range(0, len(coords), 2):
            if i + 1 < len(coords):
                x = int(coords[i] * w)
                y = int(coords[i + 1] * h)
                points.append([x, y])
        
        if len(points) >= 3:  # 至少需要3个点才能形成多边形
            points = np.array(points, dtype=np.int32)
            # 绘制多边形轮廓
            cv2.polylines(image, [points], True, (0, 255, 0), 2)
            
            # 添加标签文本
            if class_id in label_map:
                label_text = label_map[class_id]
                # 在多边形的第一个点附近显示标签
                if len(points) > 0:
                    x, y = points[0]
                    cv2.putText(image, label_text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 保存结果图像
    output_path = Path(image_path).parent / f"visualized_{Path(image_path).name}"
    cv2.imwrite(str(output_path), image)
    print(f"可视化结果已保存到: {output_path}")

label_map = {  # Define the label map with all annotated class labels.
    0: "gelikaigan",
}

# Visualize
visualize_polygon_annotations(
    "/media/industai/data11/SEG_DATA/part/yolo_data/test/images/temp_3_1.jpg",  # Input image path.
    "/media/industai/data11/SEG_DATA/part/yolo_data/test/labels/temp_3_1.txt",  # Annotation file path for the image.
    label_map,
)