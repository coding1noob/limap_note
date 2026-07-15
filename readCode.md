# 代码阅读顺序

| 顺序 | 文件 | 读什么 |
|------|------|--------|
| 1 | `runners/colmap_triangulation.py` | 入口、参数解析、整体串联 |
| 2 | `cfgs/triangulation/default.yaml` | 所有默认参数的含义 |
| 3 | `src/limap/runners/line_triangulation.py` | 六大阶段的完整流程 |
| 4 | `src/limap/runners/functions.py` | A/B/C阶段的具体实现（去畸变、检测、匹配） |
| 5 | `src/limap/pointsfm/` | 步骤1读COLMAP模型的实现 |
| 6 | `src/limap/line2d/` | 检测器(DeepLSD)、提取器(Wireframe)、匹配器(GlueStick)的实现 |
| 7 | `src/limap/triangulation/` | `GlobalLineTriangulator` 三角化核心算法 |
| 8 | `src/limap/merging/` | 线段轨迹合并与过滤 |
| 9 | `src/limap/optimize/` | Bundle Adjustment 优化 |

# 线段提取和描述子实现
line_triangulation.py
runners.compute_2d_segs()

→ src/limap/runners/functions.py 的
extractor = limap.line2d.get_extractor(cfg["line2d"]["extractor"])  # 创建 Wireframe extractor

然后 src/limap/line2d/register_detector.py 中按照配置分发不同的类，这里 wireframe
src/limap/line2d/GlueStick/extractor.py 分到的是 WireframeExtractor 类

→ extractor.extract_all_images(...)
extract_all_images 在 src/limap/line2d/base_detector.py 的 BaseDetector 基类（ src/limap/line2d/base_detector.py ）里，里面有调用 self.extract()，但是被 继承自 BaseDetector 的 WireframeExtractor 重写，其中包含 descinfo = self.compute_descinfo(img, segs) 所以实际只用看：

→ src/limap/line2d/GlueStick/extractor.py
中的 compute_descinfo，但 compute_descinfo 自己没有实现描述子的计算，核心就一行：
kp, scores, dense_desc = self.sp.compute_dense_descriptor(torch_img)
想知道描述子本身怎么算的，要去看 SuperPoint 的实现，在

→ src/limap/point2d/superpoint/superpoint.py

总结：
先用 DeepLSD 提取线段坐标存入 all_2d_segs；然后用 SuperPoint 算整张图 1/8 分辨率的密集描述子，同时检测出特征点；去掉和线段端点距离过近的特征点；最后把线段端点和剩余特征点的坐标及其描述子一起存入 .npz，descinfo_folder 指向这些文件所在的路径。

# 线段匹配

线段匹配有一些问题，用find_match2.py可视化在/home/gamma/storage_of_code/limap/outputs/buaa1_v3/line_matchings/deeplsd/feats_wireframe/gluestick_n20_top0里面的.npy匹配结果，发现线网部分的匹配结果不太好

# 三角化

三角化阶段：把多张图里匹配的2D线段恢复成3D线段
每条3D线段记录了它在哪些图里被观测到（即哪些2D线段支撑了它）
然后BA
最后用 n_visible_views 过滤：被少于 n 张图观测到的3D线段直接丢掉
-nv 2 改的就是 n_visible_views

alltracks.txt 存的是三角化后所有3D线段，包含它们的空间位置和在哪些图里被观测到，只不过经历一次过滤，只保留了 n_visible_views 以上观测到的线段

而 finaltracks/ 则是存的原始的，没有受 n_visible_views 参数过滤的

# 发现

在 -nv 2 之前过滤的
filter_tracks_by_sensitivity   ← 先过滤
    ↓
filter_tracks_by_overlap
    ↓
n_visible_views >= 2           ← 最后才判断

使得线条从17442骤降至8874

filter_tracks_by_sensitivity 是干嘛的

它过滤掉"从侧面看几乎看不出深度"的线段。

具体来说，一条3D线段如果从所有观测它的相机方向看过去，视角都几乎平行于这条线，那么这条线的深度信息就很不可靠——你从侧面看一根竖线，完全判断不了它离你有多远

                        buaa1_v3(topk=10)   buaa1_v4(topk=0, th=60)
三角化后：                  19611               19604   ← 几乎一样
filter_by_reprojection：   19611               19604   ← 没过滤
remerge后：                17442               17424   ← 几乎一样
filter_by_sensitivity：     8874               16283   ← 调松了，保留更多
filter_by_overlap：         8620                8145   ← 反而更少了！
最终：                      8620                8145   ← 反而变少了


filter_tracks_by_overlap 是干嘛的

过滤掉在图像上重叠度太高的线段轨迹，对应参数：


th_overlap: 0.5              # 重叠度阈值
th_overlap_num_supports: 3   # 满足重叠条件的最少支撑数
防护网线段密集、平行、重复，重叠度天然很高，所以大量被过滤