import os
import json
from pathlib import Path
from collections import Counter, defaultdict
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import orjson
from datetime import datetime
import random
import shutil
import threading
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 颜色表用于不同类别
COLORS = [
    (255, 0, 0),      # 红色
    (0, 255, 0),      # 绿色
    (0, 0, 255),      # 蓝色
    (255, 255, 0),    # 黄色
    (255, 0, 255),    # 品红
    (0, 255, 255),    # 青色
    (255, 128, 0),    # 橙色
    (128, 0, 255),    # 紫色
]

FONT_PATHS = [
    '/usr/share/fonts/truetype/arphic/ukai.ttc',
    '/usr/share/fonts/truetype/arphic/uming.ttc',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
]


def find_chinese_font(font_size=18):
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text_pil(img, text, x, y, font, fill=(255, 255, 255), background=(0, 0, 0)):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    text_size = draw.textbbox((0, 0), text, font=font)
    text_w = text_size[2] - text_size[0]
    text_h = text_size[3] - text_size[1]
    draw.rectangle([x, y, x + text_w + 10, y + text_h + 8], fill=background)
    draw.text((x + 5, y + 4), text, font=font, fill=fill)
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR), text_w, text_h

def visualize_sample(img_path, json_path, output_path):
    """将标注的多边形可视化到图片上"""
    try:
        # 加载图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法加载图片: {img_path}")
            return False
        
        img_height, img_width = img.shape[:2]
        
        # 加载JSON标注
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # 解析多边形
        polygons = []
        labels = []
        
        if 'shapes' in json_data:
            # LabelMe格式
            for shape in json_data['shapes']:
                if shape.get('shape_type') == 'polygon':
                    points = shape.get('points', [])
                    if points:
                        label = shape.get('label', 'unknown')
                        polygons.append(points)
                        labels.append(label)
        elif 'segmentation' in json_data:
            # COCO格式
            segmentation = json_data['segmentation']
            category_name = json_data.get('category_name', json_data.get('label', 'mask'))
            if isinstance(segmentation, list):
                for idx, polygon in enumerate(segmentation):
                    if not polygon:
                        continue
                    # 转换为 [[x,y], [x,y]] 格式
                    if isinstance(polygon[0], list):
                        points = polygon
                    else:
                        points = [[polygon[i], polygon[i+1]] for i in range(0, len(polygon), 2)]
                    polygons.append(points)
                    if len(segmentation) > 1:
                        labels.append(f"{category_name}_{idx+1}")
                    else:
                        labels.append(category_name)
        elif 'rois' in json_data:
            # 自定义 ROI 格式
            for roi in json_data['rois']:
                points = roi.get('points') or []
                if not points:
                    continue
                converted = []
                for p in points:
                    if isinstance(p, dict) and 'x' in p and 'y' in p:
                        converted.append([p['x'], p['y']])
                    elif isinstance(p, (list, tuple)) and len(p) >= 2:
                        converted.append([p[0], p[1]])
                if not converted:
                    continue
                label = roi.get('name', roi.get('label', 'unknown'))
                polygons.append(converted)
                labels.append(label)
        elif 'entities' in json_data:
            # .annotate 格式
            for entity in json_data['entities']:
                entity_label = entity.get('label', 'unknown')
                shapes = entity.get('shapes', [])
                for shape in shapes:
                    if shape.get('type') == 'Polygon':
                        coordinates = shape.get('coordinates', [])
                        if coordinates:
                            polygons.append(coordinates)
                            labels.append(entity_label)
        
        if not polygons:
            print(f"未找到多边形标注或不支持的格式: {json_path}, keys={list(json_data.keys())}")
            return False
        
        # 在图片上绘制多边形和标签
        for i, (points, label) in enumerate(zip(polygons, labels)):
            # 转换为numpy数组
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            
            # 绘制多边形半透明填充，保持原图可见
            color = COLORS[i % len(COLORS)]
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], color)
            alpha = 0.35
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            
            # 绘制轮廓
            cv2.polylines(img, [pts], True, (255, 255, 255), 2)
            
            # 计算边界框顶部中心用于放置标签
            if len(points) > 0:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                cx = int((min_x + max_x) / 2)
                cy = min_y - 10  # 在顶部向上偏移10像素
                
                font_size = 18
                font = find_chinese_font(font_size)
                
                # 预测文本宽高并调整位置
                dummy_img = Image.new('RGB', (10, 10))
                dummy_draw = ImageDraw.Draw(dummy_img)
                bbox = dummy_draw.textbbox((0, 0), label, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                
                if cy - text_h - 8 < 0:
                    cy = min_y + 10
                if cx - text_w // 2 - 5 < 0:
                    cx = text_w // 2 + 5
                if cx + text_w // 2 + 5 > img_width:
                    cx = img_width - text_w // 2 - 5
                
                draw_x = cx - text_w // 2 - 5
                draw_y = cy - text_h - 5
                
                img, _, _ = draw_text_pil(img, label, draw_x, draw_y, font,
                                         fill=(255, 255, 255), background=(0, 0, 0))
        
        # 保存结果
        cv2.imwrite(output_path, img)
        return True
        
    except Exception as e:
        print(f"可视化失败 {img_path}: {str(e)}")
        return False

def process_file(filepath: str):
    """单个文件处理函数（供多进程调用）"""
    try:
        with open(filepath, 'rb') as f:
            data = orjson.loads(f.read())

        filename = os.path.basename(filepath)
        labels = []

        if filename.endswith('.annotate'):
            for entity in data.get('entities', []):
                label = entity.get('label')
                if label:
                    labels.append(label)
        
        elif filename.endswith('.json'):
            for roi in data.get('rois', []):
                name = roi.get('name')
                if name:
                    labels.append(name)
        
        return labels, filename.endswith('.annotate'), filepath
    
    except Exception:
        return [], False, filepath  # 出错返回空列表


def _resolve_unique_path(dest_path: str):
    dest_path = Path(dest_path)
    if not dest_path.exists():
        return dest_path

    stem = dest_path.stem
    suffix = dest_path.suffix
    parent = dest_path.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def _load_completed_annotations(progress_path: Path):
    if not progress_path.exists():
        return set()
    with open(progress_path, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}


def _append_completed_annotation(progress_path: Path, annotation_path: str, lock: threading.Lock):
    with lock:
        with open(progress_path, 'a', encoding='utf-8') as f:
            f.write(annotation_path + '\n')


def _atomic_copy(src_path: str, dst_path: str):
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    tmp_path = dst_path.with_suffix(dst_path.suffix + '.tmp_copy')

    if tmp_path.exists():
        tmp_path.unlink()

    with open(src_path, 'rb') as src_file, open(tmp_path, 'wb') as dst_file:
        shutil.copyfileobj(src_file, dst_file)

    os.replace(tmp_path, dst_path)


def _find_image_for_annotation(annotation_path: str):
    annotation_path = Path(annotation_path)
    ann_dir = annotation_path.parent
    ann_name = annotation_path.name
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    stems = []
    if ann_name.endswith('.annotate'):
        stems.append(ann_name[:-len('.annotate')])
    else:
        stems.append(annotation_path.stem)

    for stem in stems:
        # 如果已包含图片后缀，则直接尝试原名
        for ext in image_exts:
            if stem.lower().endswith(ext):
                candidate = ann_dir / stem
                if candidate.exists():
                    return str(candidate)
        # 否则尝试添加常见后缀
        for ext in image_exts:
            candidate = ann_dir / f"{stem}{ext}"
            if candidate.exists():
                return str(candidate)

    return None


def copy_dataset_by_label(root_dir: str, target_dir: str, ignore_labels=None, max_workers: int = None, show_progress: bool = True, resume: bool = True):
    """将图片和对应标注按类别复制到目标目录。支持中断后继续运行。"""
    if ignore_labels is None:
        ignore_labels = ['通用-不识别']

    annotation_files = []
    with tqdm(desc="扫描标注文件", unit="file", colour="blue") as pbar:
        for root, _, files in os.walk(root_dir):
            for filename in files:
                if filename.endswith(('.annotate', '.json')):
                    annotation_files.append(os.path.join(root, filename))
                    pbar.update(1)

    if not annotation_files:
        print(f"未找到标注文件，无法执行复制操作: {root_dir}")
        return {}

    max_workers = max_workers or os.cpu_count() or 4
    copied_counts = defaultdict(int)
    skipped_count = 0
    resumed_count = 0
    error_count = 0

    target_root = Path(target_dir)
    target_root.mkdir(parents=True, exist_ok=True)

    progress_file = target_root / '.copy_progress.txt'
    completed_annotations = _load_completed_annotations(progress_file) if resume else set()
    if completed_annotations:
        print(f"检测到已有复制进度记录，自动跳过 {len(completed_annotations)} 个已完成标注文件。")
    progress_lock = threading.Lock()

    def _copy_pair(annotation_path: str):
        if annotation_path in completed_annotations:
            labels, _, filepath = process_file(annotation_path)
            category = labels[0] if labels else None
            return 'already_done', category, annotation_path

        labels, _, filepath = process_file(annotation_path)
        valid_labels = [label for label in labels if label not in ignore_labels]
        if not valid_labels:
            return 'skipped', None, filepath

        category = valid_labels[0]
        image_path = _find_image_for_annotation(filepath)
        if image_path is None:
            return 'no_image', category, filepath

        dest_dir = target_root / category
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_image = _resolve_unique_path(os.path.join(dest_dir, os.path.basename(image_path)))
        dest_annotation = _resolve_unique_path(os.path.join(dest_dir, os.path.basename(filepath)))

        try:
            if dest_image.exists() and dest_image.stat().st_size == os.path.getsize(image_path):
                pass
            else:
                if dest_image.exists():
                    dest_image.unlink()
                _atomic_copy(image_path, str(dest_image))

            if dest_annotation.exists() and dest_annotation.stat().st_size == os.path.getsize(filepath):
                pass
            else:
                if dest_annotation.exists():
                    dest_annotation.unlink()
                _atomic_copy(filepath, str(dest_annotation))

            _append_completed_annotation(progress_file, annotation_path, progress_lock)
            completed_annotations.add(annotation_path)
            return 'copied', category, filepath
        except Exception as e:
            return 'error', category, f"{filepath} -> {e}"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_annotation = {executor.submit(_copy_pair, ann_path): ann_path for ann_path in annotation_files}
        for future in tqdm(as_completed(future_to_annotation), total=len(annotation_files), desc="复制样本", unit="file", colour="cyan"):
            status, category, info = future.result()
            if status == 'copied':
                copied_counts[category] += 1
            elif status == 'already_done':
                resumed_count += 1
            elif status == 'skipped':
                skipped_count += 1
            elif status == 'no_image':
                error_count += 1
                print(f"未找到对应图片，跳过: {info} (类别: {category})")
            else:
                error_count += 1
                print(f"复制失败: {info}")

    print("\n" + "=" * 80)
    print("复制完成！")
    print(f"总标注文件: {len(annotation_files)}")
    print(f"已复制样本: {sum(copied_counts.values())}")
    print(f"已跳过已完成: {resumed_count}")
    print(f"跳过文件: {skipped_count}")
    print(f"失败文件: {error_count}")
    print(f"目标目录: {target_root}")
    print("=" * 80)

    return copied_counts


def count_labels(root_dir: str, max_workers: int = None, output_file: str = None, sample_output_dir: str = None, sample_per_class: int = 10):
    """高速统计 + 保存结果到文本文件 + 随机采样"""
    print("正在扫描所有标注文件...")

    # 收集所有标注文件
    annotation_files = []
    with tqdm(desc="扫描标注文件", unit="file", colour="blue") as pbar:
        for root, _, files in os.walk(root_dir):
            for filename in files:
                if filename.endswith(('.annotate', '.json')):
                    annotation_files.append(os.path.join(root, filename))
                    pbar.update(1)

    total_files = len(annotation_files)
    print(f"共发现 {total_files} 个标注文件，开始并行统计...\n")

    annotate_counter = Counter()
    json_counter = Counter()
    total_counter = Counter()
    label_to_files = defaultdict(list)  # 新增：标签到文件列表的映射
    error_count = 0

    # 多进程并行处理
    max_workers = max_workers or os.cpu_count() or 4
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_file, fp): fp for fp in annotation_files}
        
        for future in tqdm(as_completed(future_to_file), total=total_files, 
                          desc="并行处理标注文件", unit="file", colour="cyan"):
            try:
                labels, is_annotate, filepath = future.result()
                for label in labels:
                    if is_annotate:
                        annotate_counter[label] += 1
                    else:
                        json_counter[label] += 1
                    total_counter[label] += 1
                    label_to_files[label].append(filepath)  # 记录文件路径
            except Exception:
                error_count += 1

    # ======================== 输出到控制台 ========================
    print("\n" + "=" * 80)
    print("✅ 统计完成！")
    print(f"处理文件总数     : {total_files}")
    print(f".annotate 文件   : {sum(annotate_counter.values())}")
    print(f".json 文件       : {sum(json_counter.values())}")
    print(f"失败文件         : {error_count}")
    print(f"使用的进程数     : {max_workers}")
    print("=" * 80)

    # ======================== 保存到文本文件 ========================
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"label_statistics_{timestamp}.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("==================== 数据集标注类别统计报告 ====================\n")
        f.write(f"统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据集路径: {os.path.abspath(root_dir)}\n")
        f.write(f"总文件数: {total_files}  (失败: {error_count})\n")
        f.write("=" * 60 + "\n\n")

        # .annotate 统计
        f.write("【1. .annotate 文件统计】\n")
        if annotate_counter:
            for label, count in sorted(annotate_counter.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{label:<30} : {count:>6} 次\n")
            f.write(f"{'-'*50}\n小计: {sum(annotate_counter.values())} 个标签\n\n")
        else:
            f.write("（未找到 .annotate 文件）\n\n")

        # .json 统计
        f.write("【2. .json 文件统计】\n")
        if json_counter:
            for name, count in sorted(json_counter.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{name:<30} : {count:>6} 次\n")
            f.write(f"{'-'*50}\n小计: {sum(json_counter.values())} 个标签\n\n")
        else:
            f.write("（未找到 .json 文件）\n\n")

        # 汇总统计
        f.write("【3. 汇总统计（.annotate + .json）】\n")
        if total_counter:
            for label, count in sorted(total_counter.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{label:<30} : {count:>6} 次\n")
            f.write(f"{'-'*50}\n总计: {sum(total_counter.values())} 个标签\n")
        else:
            f.write("（未找到任何标注文件）\n")

    print(f"\n✅ 统计结果已保存到文件：")
    print(f"   {os.path.abspath(output_file)}")

    # 同时在控制台也打印主要结果（方便查看）
    print("\n【汇总统计（前10类）】")
    for label, count in sorted(total_counter.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{label:<25} : {count:>6} 次")

    # ======================== 随机采样功能 ========================
    if sample_output_dir:
        print(f"\n开始随机采样，每个类别选择 {sample_per_class} 个样本...")
        os.makedirs(sample_output_dir, exist_ok=True)
        
        sampled_count = 0
        for label, files in label_to_files.items():
            if not files:
                continue
            
            # 随机选择样本
            selected_files = random.sample(files, min(sample_per_class, len(files)))
            
            # 创建类别子文件夹
            label_dir = os.path.join(sample_output_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            for ann_file in selected_files:
                # 只处理 .json 和 .annotate 文件
                if not ann_file.endswith(('.json', '.annotate')):
                    continue

                # 先取文件名，再剥离后缀，避免完整路径截断错误
                ann_dir = os.path.dirname(ann_file)
                ann_base = os.path.basename(ann_file)
                if ann_base.endswith('.jpg.annotate'):
                    ann_name = ann_base[:-len('.jpg.annotate')]
                elif ann_base.endswith('.jpeg.annotate'):
                    ann_name = ann_base[:-len('.jpeg.annotate')]
                elif ann_base.endswith('.png.annotate'):
                    ann_name = ann_base[:-len('.png.annotate')]
                elif ann_base.endswith('.bmp.annotate'):
                    ann_name = ann_base[:-len('.bmp.annotate')]
                elif ann_base.endswith('.tiff.annotate'):
                    ann_name = ann_base[:-len('.tiff.annotate')]
                elif ann_base.endswith('.annotate'):
                    ann_name = ann_base[:-len('.annotate')]
                else:
                    ann_name = os.path.splitext(ann_base)[0]

                # 可能的图片扩展名
                image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                image_file = None
                for ext in image_exts:
                    candidate = os.path.join(ann_dir, ann_name + ext)
                    if os.path.exists(candidate):
                        image_file = candidate
                        break
                
                if image_file:
                    # 生成可视化图片
                    vis_filename = f"{ann_name}_visualized.jpg"
                    vis_path = os.path.join(label_dir, vis_filename)
                    
                    if visualize_sample(image_file, ann_file, vis_path):
                        sampled_count += 1
                        print(f"✓ 已生成可视化: {label}/{vis_filename}")
                    else:
                        print(f"✗ 可视化失败: {ann_file}")
                else:
                    print(f"警告：未找到 {ann_file} 对应的图片文件")
        
        print(f"✅ 采样完成！共生成 {sampled_count} 个可视化样本，保存到 {sample_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="高速数据集标注统计工具（支持保存到文本文件和随机采样）")
    parser.add_argument("root_dir", nargs='?', default=".", 
                        help="数据集根目录路径（默认当前目录）")
    parser.add_argument("--workers", type=int, default=None, 
                        help="进程数（默认使用所有CPU核心）")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出文本文件名（默认自动生成 label_statistics_时间戳.txt）")
    parser.add_argument("--sample-dir", type=str, default=None,
                        help="采样输出目录（如果指定，将为每个类别随机选择样本）")
    parser.add_argument("--sample-per-class", type=int, default=10,
                        help="每个类别采样的样本数量（默认10）")
    parser.add_argument("--copy-dir", type=str, default=None,
                        help="将图片和标注文件按类别复制到目标目录，忽略 通用-不识别 类别")
    parser.add_argument("--copy-workers", type=int, default=None,
                        help="复制文件时使用的线程数（默认自动选择）")

    args = parser.parse_args()

    count_labels(args.root_dir, args.workers, args.output, args.sample_dir, args.sample_per_class)

    if args.copy_dir:
        copy_dataset_by_label(
            args.root_dir,
            args.copy_dir,
            ignore_labels=['通用-不识别'],
            max_workers=args.copy_workers,
            show_progress=True
        )