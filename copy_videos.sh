#!/bin/bash
#
# 将源目录中每个子文件夹内的视频文件复制到目标目录，并以子文件夹名重命名。
# 用法: bash copy_videos.sh [SRC] [DST] [VIDEO_FILENAME]
#   SRC            源目录（包含多个子文件夹），默认见下方 DEFAULT_SRC
#   DST            目标目录，默认见下方 DEFAULT_DST
#   VIDEO_FILENAME 子文件夹内视频文件名，默认 gen.mp4
#

# ===== 可修改的默认配置 =====
DEFAULT_SRC="/nfs/yixinyang/code/LongLive/data/minWM-data/videos"
DEFAULT_DST="/nfs/yixinyang/code/LongLive/data/minWM-data/videos_train"
DEFAULT_VIDEO_FILENAME="gen.mp4"
# ============================

SRC="${1:-$DEFAULT_SRC}"
DST="${2:-$DEFAULT_DST}"
VIDEO_FILENAME="${3:-$DEFAULT_VIDEO_FILENAME}"

# 参数检查
if [ ! -d "$SRC" ]; then
    echo "错误: 源目录不存在: $SRC"
    exit 1
fi

if [ -z "$VIDEO_FILENAME" ]; then
    echo "错误: 视频文件名不能为空"
    exit 1
fi

mkdir -p "$DST"

count=0
skipped=0
for d in "$SRC"/*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    if [ -f "$d/$VIDEO_FILENAME" ]; then
        cp "$d/$VIDEO_FILENAME" "$DST/${name}.${VIDEO_FILENAME##*.}"
        count=$((count + 1))
    else
        skipped=$((skipped + 1))
    fi
done

echo "完成。"
echo "  源目录:   $SRC"
echo "  目标目录: $DST"
echo "  视频原名: $VIDEO_FILENAME"
echo "  复制成功: $count 个"
echo "  跳过(无视频): $skipped 个"
