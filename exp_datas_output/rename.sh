#!/bin/bash

# 根目录
root_dir="你的目录路径"

# 遍历所有子目录
find "$root_dir" -type d -name "100" | while read dir; do
    parent_dir=$(dirname "$dir") # 获取父目录路径
    new_dir="$parent_dir/final" # 新目录路径

    # 重命名文件夹
    mv "$dir" "$new_dir"
    echo "Renamed: $dir -> $new_dir"
done

echo "All '100' directories have been renamed to 'final'."