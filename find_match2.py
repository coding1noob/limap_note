import os
import numpy as np
import cv2

# ========== 配置 ==========
TARGET_IMG_ID = 30
MATCHES_DIR = "/home/gamma/storage_of_code/limap/outputs/buaa1_v5/line_matchings/deeplsd/feats_wireframe/gluestick_n20_top0"
SEGMENTS_DIR = "/home/gamma/storage_of_code/limap/outputs/buaa1_v5/line_detections/deeplsd/segments"
VIS_DIR = "/home/gamma/storage_of_code/limap/outputs/buaa1_v5/line_detections/deeplsd/visualize"
OUTPUT_BASE = "/home/gamma/storage_of_code/limap/outputs/buaa1_v5/line_matchings"
TOP_K = 10          # 只绘制匹配数量最多的前 K 个邻居
LINE_THICKNESS = 2  # 连线粗细
# ==========================


def read_segments(img_id):
    """读取某张图的线段坐标，返回 (N, 4) 数组"""
    path = os.path.join(SEGMENTS_DIR, f"segments_{img_id}.txt")
    segs = []
    with open(path, "r") as f:
        lines = f.read().strip().split("\n")
    for line in lines[1:]:  # 第一行是数量
        vals = list(map(float, line.strip().split()))
        if len(vals) >= 4:
            segs.append(vals[:4])
    return np.array(segs)


def read_image(img_id):
    """读取检测可视化图，转为灰白背景"""
    path = os.path.join(VIS_DIR, f"img_{img_id}_det.png")
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"找不到图像: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return img_gray

# max_draw 表示随机抽取匹配结果的数量
def draw_matches(img_a, segs_a, img_b, segs_b, matches, max_draw=50):
    """
    把两张图横向拼接，在匹配的线段之间连线
    matches: (M, 2) 每行是 [id_in_a, id_in_b]
    """
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    h = max(h_a, h_b)

    # 补齐高度
    canvas_a = np.zeros((h, w_a, 3), dtype=np.uint8)
    canvas_a[:h_a] = img_a
    canvas_b = np.zeros((h, w_b, 3), dtype=np.uint8)
    canvas_b[:h_b] = img_b

    canvas = np.concatenate([canvas_a, canvas_b], axis=1)

    # 随机选最多 max_draw 条匹配连线，避免太密
    idxs = np.arange(len(matches))
    if len(idxs) > max_draw:
        idxs = np.random.choice(idxs, max_draw, replace=False)

    np.random.seed(42)
    for idx in idxs:
        id_a, id_b = matches[idx]
        if id_a >= len(segs_a) or id_b >= len(segs_b):
            continue
        # 取线段中点作为连线端点
        x1a, y1a, x2a, y2a = segs_a[id_a]
        mx_a = int((x1a + x2a) / 2)
        my_a = int((y1a + y2a) / 2)

        x1b, y1b, x2b, y2b = segs_b[id_b]
        mx_b = int((x1b + x2b) / 2) + w_a  # 偏移到右图
        my_b = int((y1b + y2b) / 2)

        color = tuple(np.random.randint(50, 255, 3).tolist())
        cv2.line(canvas, (mx_a, my_a), (mx_b, my_b), color, LINE_THICKNESS)
        cv2.circle(canvas, (mx_a, my_a), LINE_THICKNESS + 1, color, -1)
        cv2.circle(canvas, (mx_b, my_b), LINE_THICKNESS + 1, color, -1)

    return canvas


def main():
    # 读取目标图像的匹配结果
    matches_path = os.path.join(MATCHES_DIR, f"matches_{TARGET_IMG_ID}.npy")
    data = np.load(matches_path, allow_pickle=True).item()

    # 按匹配数量排序，取前 TOP_K
    counts = {nbr: len(m) for nbr, m in data.items()}
    top_neighbors = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:TOP_K]

    print(f"图像 {TARGET_IMG_ID} 匹配数量前{TOP_K}的邻居：")
    for nbr, cnt in top_neighbors:
        print(f"  图像 {nbr}: {cnt} 条匹配")

    # 读取目标图像
    img_a = read_image(TARGET_IMG_ID)
    segs_a = read_segments(TARGET_IMG_ID)

    # 输出文件夹
    output_dir = os.path.join(OUTPUT_BASE, f"match_pic{TARGET_IMG_ID}")
    os.makedirs(output_dir, exist_ok=True)

    # 逐个绘制
    for rank, (nbr_id, cnt) in enumerate(top_neighbors):
        print(f"正在绘制 图{TARGET_IMG_ID} vs 图{nbr_id} ({cnt}条匹配)...")
        img_b = read_image(nbr_id)
        segs_b = read_segments(nbr_id)
        matches = data[nbr_id]

        canvas = draw_matches(img_a, segs_a, img_b, segs_b, matches)

        # 在图上标注信息
        cv2.putText(canvas, f"img_{TARGET_IMG_ID} vs img_{nbr_id}  matches={cnt}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        out_path = os.path.join(output_dir, f"rank{rank+1:02d}_img{TARGET_IMG_ID}_vs_img{nbr_id}.png")
        cv2.imwrite(out_path, canvas)
        print(f"  已保存: {out_path}")

    print(f"\n全部完成，输出目录: {output_dir}")


if __name__ == "__main__":
    main()