#!/bin/bash
# 在 HPC 登录节点运行，测试 ImageNet 路径是否可读： ./scripts/check_imagenet_path.sh
set -e
echo "=== 测试 ImageNet 路径 ==="
for path in "/projects/work/public/ml-datasets/imagenet" "/scratch/work/public/ml-datasets/imagenet" "/vast/work/public/ml-datasets/imagenet"; do
  echo ""
  echo "路径: $path"
  if [ ! -d "$path" ]; then
    echo "  -> 不存在或不可访问"
    continue
  fi
  echo "  -> 根目录存在"
  train_dir="$path/train"
  if [ -d "$train_dir" ]; then
    n=$(ls -1 "$train_dir" 2>/dev/null | wc -l)
    echo "  -> train/ 存在，约 $n 个子目录（类别）"
    echo "  -> 可用，请在 config 中设置: data_path: \"$path\""
  else
    echo "  -> train/ 不是目录或不可访问"
    if [ -e "$train_dir" ]; then
      echo "  -> train 存在但类型为: $(file -b "$train_dir" 2>/dev/null || echo 'unknown')"
    fi
    echo "  -> 列出根目录内容:"
    ls -la "$path" 2>/dev/null | head -5
  fi
done
echo ""
echo "若两个都不可用，请查集群文档确认 ImageNet 在本集群的挂载路径。"
