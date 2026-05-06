#!/bin/bash

# ================= 脚本说明 =================
# 用法: ./flat_copy.sh <源文件夹> <目标文件夹> <抽取数量>
# 功能: 递归遍历源文件夹，随机抽取指定数量图片，扁平化输出到单一目标文件夹
# 示例: ./flat_copy.sh /data/images /data/flats 5
# ===========================================

# 1. 检查参数
if [ $# -ne 3 ]; then
    echo "❌ 错误：参数数量错误！"
    echo "用法: $0 <源文件夹> <目标文件夹> <抽取数量>"
    echo "示例: $0 /source /target 3"
    exit 1
fi

SOURCE_DIR="$1"
TARGET_DIR="$2"
NUM_SAMPLES="$3"

# 2. 验证参数
[ ! -d "$SOURCE_DIR" ] && { echo "❌ 错误：源文件夹 '$SOURCE_DIR' 不存在"; exit 1; }
! [[ "$NUM_SAMPLES" =~ ^[0-9]+$ ]] || [ "$NUM_SAMPLES" -lt 1 ] && { 
    echo "❌ 错误：抽取数量必须是正整数（当前值: '$NUM_SAMPLES'）"
    exit 1
}

# 3. 创建目标文件夹
mkdir -p "$TARGET_DIR" || { echo "❌ 错误：无法创建目标文件夹"; exit 1; }

# 4. 主逻辑：收集所有图片路径
mapfile -t all_images < <(
    find "$SOURCE_DIR" -type f \( \
        -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \
        -o -iname "*.bmp" -o -iname "*.gif" -o -iname "*.tiff" \) -print
)

# 5. 检查是否有图片
if [ ${#all_images[@]} -eq 0 ]; then
    echo "❌ 错误：在 '$SOURCE_DIR' 中未找到任何图片文件"
    exit 1
fi

# 6. 计算实际抽取数量（防止总数不足）
actual_samples=$(( ${#all_images[@]} < NUM_SAMPLES ? ${#all_images[@]} : NUM_SAMPLES ))

# 7. 随机抽取并复制（关键：扁平化+重命名）
printf '%s\n' "${all_images[@]}" | shuf -n "$actual_samples" | while read -r src_file; do
    # 生成唯一文件名：[路径哈希]_[原始文件名]
    # 例如：5f3a8c2d_cats_kitty1.jpg
    prefix=$(echo "${src_file#$SOURCE_DIR/}" | tr '/' '_' | sha256sum | cut -c1-8)
    filename=$(basename "$src_file")
    dest_file="${TARGET_DIR}/${prefix}_${filename}"
    
    # 复制并重命名（自动覆盖同名文件）
    cp -v "$src_file" "$dest_file" 2>/dev/null || \
        echo "⚠️ 跳过无法复制的文件: $src_file"
done

echo "✅ 成功！已从 '$SOURCE_DIR' 随机抽取 $actual_samples 个图片到 '$TARGET_DIR'"
echo "   - 所有图片已扁平化输出至同一文件夹"
echo "   - 文件名格式: [路径哈希]_[原始文件名]（避免冲突）"
exit 0