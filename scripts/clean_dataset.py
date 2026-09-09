#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键清洗数据集：过滤极端长宽比标注 + 应用类别映射

用途
----
对 LabelMe JSON 格式的数据集（目录结构：<类别目录>/xxx.jpg + xxx.json）做清洗：
  1. 过滤极端长宽比标注：单个标注的 bbox 宽高比 w/h 落在 (min_ratio, max_ratio) 之外
     即视为极端（细长条），删除该 shape；
  2. 应用类别映射：把中文 label / 历史英文残留（如 指示 → she_bei_biao_shi，
     zhi_shi → she_bei_biao_shi）统一映射为目标英文名；
  3. 若某 JSON 的全部 shape 被删空 / 无有效标注，则整个样本（jpg+json）不复制到输出。

输出
----
与输入同构的新数据集目录（类别目录 + jpg + json），原数据集不被修改。
生成清洗统计报告（每类：JSON 总数 / 极端标注删除数 / 删空样本数 / 类名映射数）。

用法
----
python3 scripts/clean_dataset.py \
  --input  /media/industai/data11/SEG_DATA/converted_dataset \
  --output /media/industai/data11/SEG_DATA/converted_dataset_clean \
  --class_mapping class_mapping.txt

依赖
----
仅标准库（json / shutil / os / argparse）。项目约定用 /home/industai/anaconda3/bin/python。
"""

import os
import sys
import json
import shutil
import argparse
from collections import OrderedDict


# ---------- 工具函数 ----------

def load_class_mapping(mapping_file):
    """加载类别映射文件（格式: 中文名:英文名，每行一个映射，支持 # 注释）"""
    mapping = OrderedDict()
    if not mapping_file or not os.path.exists(mapping_file):
        print(f"警告: 类别映射文件不存在或未指定: {mapping_file}，将不应用中文→英文映射")
        return mapping
    with open(mapping_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line:
                src, dst = line.split(':', 1)
                src = src.strip()
                dst = dst.strip()
                if src and dst:
                    mapping[src] = dst
    return mapping


def parse_extra_map(extra_map_str):
    """解析补充映射参数，如 'zhi_shi=she_bei_biao_shi,dian_kang_qi=x' → {旧: 新}"""
    extra = OrderedDict()
    if not extra_map_str:
        return extra
    for item in extra_map_str.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' in item:
            src, dst = item.split('=', 1)
            src, dst = src.strip(), dst.strip()
            if src and dst:
                extra[src] = dst
        else:
            print(f"警告: 忽略无法解析的补充映射项: {item!r}（应为 旧名=新名）")
    return extra


def bbox_from_points(points):
    """从标注点计算 bbox (w, h)；输入 [[x,y], ...] 或 [x1,y1,x2,y2,...]"""
    if not points:
        return 0, 0
    # 归一化为 [[x,y], ...]
    if isinstance(points[0], (list, tuple)):
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
    else:
        xs = [float(points[i]) for i in range(0, len(points), 2)]
        ys = [float(points[i + 1]) for i in range(0, len(points), 2)]
    if not xs or not ys:
        return 0, 0
    w = max(xs) - min(xs) + 1
    h = max(ys) - min(ys) + 1
    return w, h


# ---------- 清洗核心 ----------

class DatasetCleaner:
    """按类别目录遍历 jpg+json，清洗标注并输出新数据集"""

    SUPPORTED_IMG = {'.jpg', '.jpeg', '.png', '.bmp'}

    def __init__(self, input_path, output_path, class_mapping=None,
                 extra_map=None, max_ratio=10.0, min_ratio=0.1):
        self.input_path = os.path.abspath(input_path)
        self.output_path = os.path.abspath(output_path)
        self.class_mapping = class_mapping or {}
        self.extra_map = extra_map or {}
        self.max_ratio = float(max_ratio)
        self.min_ratio = float(min_ratio)
        # 每类统计: {json数, 极端删除, 删空样本, 类名映射数, 错误数}
        self.stats = OrderedDict()

    # ---- 映射 ----

    def map_label(self, label):
        """先查补充旧名映射，再查中文→英文映射，否则返回原 label"""
        if label in self.extra_map:
            return self.extra_map[label]
        return self.class_mapping.get(label, label)

    # ---- 单样本处理 ----

    def process_json(self, json_path):
        """
        清洗单个 JSON。
        返回 (data, kept, mapped_count, orig_count)：
          data        清洗后的完整 JSON（shapes 已更新）
          kept        清洗后应保留的 shapes 列表（空列表表示整样本丢弃）
          mapped_count 本次类名映射次数
          orig_count  原始 shape 数量（用于统计极端删除数）
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        shapes = data.get('shapes', [])
        orig_count = len(shapes)
        kept = []
        mapped_count = 0

        for shape in shapes:
            label = shape.get('label', '')
            points = shape.get('points', [])
            shape_type = shape.get('shape_type', 'polygon')

            # 1) 过滤极端长宽比
            w, h = bbox_from_points(points)
            if w > 0 and h > 0:
                ar = w / h
                if ar > self.max_ratio or ar < self.min_ratio:
                    # 极端标注，删除
                    continue
            # w 或 h 为 0（退化为线/点）：无有效面积，一并过滤
            elif w <= 0 or h <= 0:
                continue

            # 2) 应用类名映射
            new_label = self.map_label(label)
            if new_label != label:
                shape['label'] = new_label
                mapped_count += 1

            # 3) 保留 shape_type 元信息
            if not shape_type:
                shape['shape_type'] = 'polygon'
            kept.append(shape)

        data['shapes'] = kept
        return data, kept, mapped_count, orig_count

    def clean_category(self, cat_name, cat_in_dir, cat_out_dir):
        """清洗单个类别目录"""
        st = self.stats.setdefault(cat_name, {
            'json': 0, 'extreme_removed': 0, 'dropped': 0, 'mapped': 0, 'error': 0,
        })
        os.makedirs(cat_out_dir, exist_ok=True)

        for fn in sorted(os.listdir(cat_in_dir)):
            if not fn.endswith('.json'):
                continue
            json_path = os.path.join(cat_in_dir, fn)
            img_path = os.path.join(cat_in_dir, fn[:-5] + '.jpg')
            if not os.path.exists(img_path):
                # 尝试其它图片后缀
                base = fn[:-5]
                img_path = None
                for suffix in ('.jpg', '.jpeg', '.png', '.bmp'):
                    cand = os.path.join(cat_in_dir, base + suffix)
                    if os.path.exists(cand):
                        img_path = cand
                        break
                if img_path is None:
                    st['error'] += 1
                    print(f"  警告: 跳过缺少图片的标注: {json_path}")
                    continue

            st['json'] += 1
            try:
                data, kept, mapped_count, orig_count = self.process_json(json_path)
            except Exception as e:
                st['error'] += 1
                print(f"  错误: JSON 解析失败 {json_path}: {e}")
                continue

            # 极端删除数 = 原 shape 数 - 保留 shape 数（映射只改 label，不增减数量）
            st['extreme_removed'] += max(0, orig_count - len(kept))
            st['mapped'] += mapped_count

            if not kept:
                # 全部 shape 被删空 → 整样本丢弃
                st['dropped'] += 1
                continue

            # 写清洗后的 JSON
            out_json = os.path.join(cat_out_dir, fn)
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # 复制图片
            out_img = os.path.join(cat_out_dir, os.path.basename(img_path))
            shutil.copy2(img_path, out_img)

    def clean(self):
        if not os.path.isdir(self.input_path):
            print(f"错误: 输入目录不存在: {self.input_path}")
            return False
        os.makedirs(self.output_path, exist_ok=True)

        # 收集所有类别目录
        cats = sorted(
            d for d in os.listdir(self.input_path)
            if os.path.isdir(os.path.join(self.input_path, d))
        )
        if not cats:
            print("警告: 输入目录下没有类别子目录")
            return False

        for cat in cats:
            cat_in = os.path.join(self.input_path, cat)
            cat_out = os.path.join(self.output_path, cat)
            print(f"[{cat}] 处理中 ...")
            self.clean_category(cat, cat_in, cat_out)
        return True

    def generate_report(self):
        """生成清洗统计报告文本"""
        lines = []
        lines.append("=" * 70)
        lines.append("数据集清洗统计报告")
        lines.append("=" * 70)
        lines.append(f"输入目录: {self.input_path}")
        lines.append(f"输出目录: {self.output_path}")
        lines.append(f"极端长宽比判定: w/h > {self.max_ratio} 或 w/h < {self.min_ratio}")
        lines.append(f"补充类名映射: {dict(self.extra_map) or '无'}")
        lines.append("")
        lines.append(f"{'类别':<12}{'JSON数':>8}{'极端删除':>9}{'删空样本':>9}{'类名映射':>9}{'错误':>6}")
        lines.append("-" * 70)
        tot = {'json': 0, 'extreme_removed': 0, 'dropped': 0, 'mapped': 0, 'error': 0}
        for cat, st in self.stats.items():
            lines.append(
                f"{cat:<12}{st['json']:>8}{st['extreme_removed']:>9}"
                f"{st['dropped']:>9}{st['mapped']:>9}{st['error']:>6}"
            )
            for k in tot:
                tot[k] += st[k]
        lines.append("-" * 70)
        lines.append(
            f"{'TOTAL':<12}{tot['json']:>8}{tot['extreme_removed']:>9}"
            f"{tot['dropped']:>9}{tot['mapped']:>9}{tot['error']:>6}"
        )
        lines.append("")
        lines.append("注: 极端删除 = 因宽高比极端被删除的标注数；")
        lines.append("    删空样本 = 标注全被删掉而未写入输出的 jpg+json 样本数；")
        lines.append("    类名映射 = 发生 label 重命名的标注数（如 指示/zhi_shi → she_bei_biao_shi）。")
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="一键清洗数据集：过滤极端长宽比标注 + 应用类别映射")
    parser.add_argument('--input', required=True, help='输入数据集根目录（类别目录/jpg+json）')
    parser.add_argument('--output', default=None,
                        help='输出新数据集目录（默认: 输入同级 converted_dataset_clean）')
    parser.add_argument('--class_mapping', default=None,
                        help='类别映射文件路径（格式: 中文:英文；默认读项目根 class_mapping.txt）')
    parser.add_argument('--max_ratio', type=float, default=10.0, help='宽高比上限，超过视为极端 (默认 10)')
    parser.add_argument('--min_ratio', type=float, default=0.1, help='宽高比下限，低于视为极端 (默认 0.1)')
    parser.add_argument('--extra_map', default='zhi_shi=she_bei_biao_shi',
                        help='补充旧英文名→标准名映射，逗号分隔 旧=新 (默认 zhi_shi=she_bei_biao_shi)')
    parser.add_argument('--report', default=None, help='清洗报告输出路径（默认: 输出目录同级 clean_report.txt）')
    args = parser.parse_args()

    # 默认输出路径
    if not args.output:
        args.output = os.path.join(os.path.dirname(os.path.abspath(args.input)), 'converted_dataset_clean')
    # 默认类映射文件：优先 --class_mapping，其次项目根/scripts 同级 class_mapping.txt
    if not args.class_mapping:
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        for cand in (
            os.path.join(os.path.dirname(scripts_dir), 'class_mapping.txt'),
            os.path.join(scripts_dir, 'class_mapping.txt'),
        ):
            if os.path.exists(cand):
                args.class_mapping = cand
                break

    print("=" * 60)
    print("数据集一键清洗")
    print("=" * 60)
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print(f"类映射文件: {args.class_mapping or '无'}")
    print(f"极端长宽比: w/h > {args.max_ratio} 或 w/h < {args.min_ratio}")
    print(f"补充映射: {args.extra_map}")
    print()

    class_mapping = load_class_mapping(args.class_mapping)
    extra_map = parse_extra_map(args.extra_map)

    cleaner = DatasetCleaner(
        args.input, args.output,
        class_mapping=class_mapping,
        extra_map=extra_map,
        max_ratio=args.max_ratio,
        min_ratio=args.min_ratio,
    )

    ok = cleaner.clean()
    if not ok:
        return 1

    report = cleaner.generate_report()
    print()
    print(report)

    report_path = args.report or os.path.join(
        os.path.dirname(os.path.abspath(args.output)), 'clean_report.txt'
    )
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report + "\n")
    print(f"\n清洗报告已保存: {report_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
