#!/bin/bash
# 脚本功能：将每个子目录下zip压缩包中的JSON文件解压，并将图片和JSON文件拷贝到目标目录
# 用法: ./extract_and_copy.sh [输入目录] [输出目录]
# 默认: 输入目录为当前目录，输出目录为 /tmp/隔离开关

# 默认路径
DEFAULT_INPUT_DIR="$(pwd)"
DEFAULT_OUTPUT_DIR="/tmp/隔离开关"

# 获取输入和输出目录
INPUT_DIR="${1:-$DEFAULT_INPUT_DIR}"
OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录不存在: $INPUT_DIR"
    exit 1
fi

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_DIR"

# 检查输出目录是否可写
if [ ! -w "$OUTPUT_DIR" ]; then
    echo "错误: 输出目录不可写: $OUTPUT_DIR"
    exit 1
fi

echo "=========================================="
echo "图片和JSON文件提取拷贝工具"
echo "=========================================="
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="
echo ""

# 统计变量
total_dirs=0
total_images=0
total_jsons=0

# 遍历所有子目录
for dir in "$INPUT_DIR"/*/; do
    # 检查是否是目录
    [ -d "$dir" ] || continue
    
    # 跳过隐藏目录
    dir_name=$(basename "$dir")
    [[ "$dir_name" == .* ]] && continue
    
    # 查找json.zip文件
    json_zip=""
    for item in "$dir"*json*.zip; do
        [ -e "$item" ] || continue
        json_zip="$item"
        break
    done
    
    # 如果没有找到json.zip，跳过
    if [ -z "$json_zip" ]; then
        echo "跳过 $dir_name: 未找到json.zip文件"
        continue
    fi
    
    # 查找图片文件夹（优先查找包含jpg的文件夹，否则查找数字命名的文件夹）
    img_folder=""
    for item in "$dir"*; do
        [ -e "$item" ] || continue
        basename_item=$(basename "$item")
        
        # 跳过zip文件
        [[ "$basename_item" == *.zip ]] && continue
        
        if [ -d "$item" ]; then
            # 优先查找包含jpg的文件夹
            if [[ "$basename_item" == *"jpg"* ]]; then
                img_folder="$item"
                break
            # 否则查找数字命名的文件夹
            elif [[ "$basename_item" =~ ^[0-9]+$ ]]; then
                img_folder="$item"
            fi
        fi
    done
    
    # 如果没有找到文件夹，跳过
    if [ -z "$img_folder" ]; then
        echo "跳过 $dir_name: 未找到图片文件夹"
        continue
    fi
    
    echo "处理目录: $dir_name"
    echo "  图片文件夹: $(basename "$img_folder")"
    echo "  json压缩包: $(basename "$json_zip")"
    
    # 创建临时目录用于解压
    temp_dir=$(mktemp -d)
    
    # 解压zip文件到临时目录（忽略警告信息）
    unzip -o "$json_zip" -d "$temp_dir" 2>/dev/null
    
    # 检查临时目录是否有json文件
    json_count_in_temp=$(find "$temp_dir" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l)
    
    if [ "$json_count_in_temp" -eq 0 ]; then
        echo "  解压失败或没有JSON文件，跳过此目录"
        rm -rf "$temp_dir"
        continue
    fi
    
    # 统计当前目录的文件数
    dir_images=0
    dir_jsons=0
    
    # 拷贝图片文件到输出目录
    for img_file in "$img_folder"/*.jpg "$img_folder"/*.png "$img_folder"/*.jpeg; do
        [ -e "$img_file" ] || continue
        cp "$img_file" "$OUTPUT_DIR/"
        if [ $? -eq 0 ]; then
            ((dir_images++))
        else
            echo "  警告: 无法拷贝图片 $(basename "$img_file")"
        fi
    done
    
    # 拷贝JSON文件到输出目录
    for json_file in "$temp_dir"/*.json; do
        [ -e "$json_file" ] || continue
        cp "$json_file" "$OUTPUT_DIR/"
        if [ $? -eq 0 ]; then
            ((dir_jsons++))
        else
            echo "  警告: 无法拷贝JSON $(basename "$json_file")"
        fi
    done
    
    # 清理临时目录
    rm -rf "$temp_dir"
    
    echo "  完成: 拷贝 $dir_images 个图片, $dir_jsons 个JSON文件"
    
    # 更新总计
    ((total_dirs++))
    ((total_images += dir_images))
    ((total_jsons += dir_jsons))
    
    echo ""
done

echo "=========================================="
echo "处理完成！"
echo "=========================================="
echo "处理目录数: $total_dirs"
echo "拷贝图片数: $total_images"
echo "拷贝JSON数: $total_jsons"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="
