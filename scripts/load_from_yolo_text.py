from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes, Masks
import numpy as np
import torch
from pathlib import Path

def load_yolo_seg_annotations(txt_path, img_path, class_names):
    """
    将 YOLO 格式分割标注文件 (.txt) 反序列化为 Ultralytics Results 对象
    
    参数:
        txt_path (str): YOLO 标注文件路径
        img_path (str): 原始图像路径 (用于获取图像尺寸)
        class_names (dict): 类别ID到名称的映射 {0: "person", 1: "car", ...}
    
    返回:
        Results 对象 (可直接用于 visualize_image_annotations 或 plot())
    """
    # 1. 读取图像尺寸
    from PIL import Image
    img = Image.open(img_path)
    img_w, img_h = img.size  # 注意：PIL 返回 (width, height)
    
    # 2. 解析 YOLO 格式文本文件
    boxes_data = []  # 存储 [cls, x1, y1, x2, y2, ...] (归一化坐标)
    masks_xy = []    # 存储掩码顶点列表 [array1, array2, ...]
    
    if not Path(txt_path).exists():
        print(f"⚠️ 标注文件不存在: {txt_path}，返回空结果")
        return Results(
            orig_img=np.zeros((img_h, img_w, 3), dtype=np.uint8),
            path=img_path,
            names=class_names,
            boxes=torch.empty((0, 6)),  # 空Boxes tensor
            masks=torch.empty((0, img_h, img_w))  # 空Masks tensor
        )
    
    with open(txt_path, 'r') as f:
        for line in f:
            data = list(map(float, line.strip().split()))
            if not data: 
                continue
                
            class_id = int(data[0])
            points = np.array(data[1:]).reshape(-1, 2)  # 转为 (N, 2) 数组
            
            # 2.1 归一化坐标 → 像素坐标
            points[:, 0] *= img_w  # x
            points[:, 1] *= img_h  # y
            
            # 2.2 重建边界框 (从掩码顶点计算)
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            boxes_data.append([x_min, y_min, x_max, y_max, 1.0, class_id])  # [xyxy, conf=1.0, cls]
            
            # 2.3 保存掩码顶点 (保持原始顶点顺序)
            masks_xy.append(points.astype(np.float32))
    
    # 3. 构造 boxes tensor
    if boxes_data:
        # 转为 tensor: [N, 6] -> [xyxy, conf, cls]
        boxes_tensor = torch.tensor(boxes_data, dtype=torch.float32)
    else:
        boxes_tensor = torch.empty((0, 6))
    
    # 4. 构造 masks tensor
    if masks_xy:
        # 将顶点列表转换为完整的mask tensor
        # 首先创建空的mask tensor
        mask_tensor = torch.zeros((len(masks_xy), img_h, img_w), dtype=torch.float32)
        
        # 为每个掩码填充对应的区域
        from PIL import Image, ImageDraw
        for i, points in enumerate(masks_xy):
            # 创建单个掩码图像
            mask_img = Image.new('L', (img_w, img_h), 0)
            draw = ImageDraw.Draw(mask_img)
            # 将点坐标转换为整数元组
            polygon_points = [(int(x), int(y)) for x, y in points]
            if len(polygon_points) >= 3:  # 至少需要3个点才能形成多边形
                draw.polygon(polygon_points, outline=1, fill=1)
            mask_array = np.array(mask_img)
            mask_tensor[i] = torch.from_numpy(mask_array).float()
    else:
        mask_tensor = torch.empty((0, img_h, img_w))
    
    # 5. 构建 Results 对象
    return Results(
        orig_img=np.array(img),        # 原始图像 (RGB)
        path=img_path,                 # 图像路径
        names=class_names,             # 类别名称映射 (用于可视化)
        boxes=boxes_tensor,            # boxes tensor
        masks=mask_tensor              # masks tensor
    )

# ===================== 使用示例 =====================
if __name__ == "__main__":
    # 配置参数
    txt_path = "/media/industai/data1/SEG_DATA/yolo_dataset/test/labels/隔离开关_1553_1.txt"          # YOLO标注文件路径
    img_path = "/media/industai/data1/SEG_DATA/yolo_dataset/test/images/隔离开关_1553_1.jpg"          # 原始图像路径

    class_names = {0: "she_bei_biao_shi", 1: "fa_lan",2: "jie_di_yin_xia_xian",3: "peng_zhang_qi",4: "触头、导电臂",5: "shen_suo_jie",6: "hu_xi_qi",7: "ji_dian_qi",8: "jue_yuan_zi",9: "ji_gou_xiang"}  # 按需替换你的类别
    
    # 反序列化
    results = load_yolo_seg_annotations(txt_path, img_path, class_names)
    
    # 验证: 使用Results对象的plot方法进行可视化
    annotated_img = results.plot()
    print("✅ 成功创建Results对象并生成可视化图像！")
    print(f"检测到 {len(results)} 个对象")
    
    from PIL import Image
    Image.fromarray(annotated_img).save("reconstructed.jpg")
    print("✅ 反序列化成功! 可视化结果已保存为 reconstructed.jpg")