import sys
from collections import Counter

path = "/home/gamma/storage_of_code/limap/outputs/buaa1_v3/alltracks.txt"

with open(path, "r") as f:
    lines = f.read().strip().split("\n")

total = int(lines[0])
print(f"alltracks.txt 总轨迹数: {total}")

# 每条轨迹占5行：header, start, end, img_ids, seg_ids
n_images_list = []
i = 1
while i < len(lines):
    header = lines[i].split()
    if len(header) >= 2:
        n_images = int(header[1])
        n_images_list.append(n_images)
    i += 5

print(f"实际解析轨迹数: {len(n_images_list)}")
print()

# 按观测图像数统计分布
counter = Counter(n_images_list)
print("观测图像数分布（前20档）：")
print(f"{'观测图数':>8}  {'轨迹数':>8}  {'累计占比':>10}")
total_tracks = len(n_images_list)
for k in sorted(counter.keys())[:20]:
    filtered = sum(v for kk, v in counter.items() if kk < k)
    print(f"{k:>8}  {counter[k]:>8}  {filtered:>10} ({filtered/total_tracks*100:.1f}%)")

print()
# 关键门槛统计
for nv in [2, 3, 4, 5]:
    kept = sum(v for k, v in counter.items() if k >= nv)
    filtered = total_tracks - kept
    print(f"n_visible_views >= {nv}: 保留 {kept} 条，过滤掉 {filtered} 条 ({filtered/total_tracks*100:.1f}%)")
