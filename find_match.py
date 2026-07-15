import os
import numpy as np

matches_dir = "/home/gamma/storage_of_code/limap/outputs/buaa1_v2/line_matchings/deeplsd/feats_wireframe/gluestick_n20_top10"
output_path = "/home/gamma/storage_of_code/limap/outputs/buaa1_v2/line_matchings/match_summary.txt"

lines = []
fnames = sorted(os.listdir(matches_dir), key=lambda x: int(x.replace("matches_", "").replace(".npy", "")) if x.endswith(".npy") else -1)
for fname in fnames:
    if not fname.endswith(".npy"):
        continue
    img_id = int(fname.replace("matches_", "").replace(".npy", ""))
    data = np.load(os.path.join(matches_dir, fname), allow_pickle=True).item()

    # 统计每个邻居的匹配数量，取前5
    counts = {neighbor_id: len(matches) for neighbor_id, matches in data.items()}
    top5 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top5_str = ", ".join(f'"{neighbor_id}": {cnt}' for neighbor_id, cnt in top5)
    lines.append(f"{img_id}  {{{top5_str}}}")

with open(output_path, "w") as f:
    f.write("\n".join(lines))

print(f"已保存到 {output_path}")