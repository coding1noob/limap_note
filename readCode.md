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

描述子实现：
line_triangulation.py
runners.compute_2d_segs()
→ src/limap/runners/functions.py 的
extractor = limap.line2d.get_extractor(cfg["line2d"]["extractor"])  # 创建 Wireframe extractor
然后 src/limap/line2d/register_detector.py 中按照配置分发不同的类，这里 wireframe
src/limap/line2d/GlueStick/extractor.py 分到的是 WireframeExtractor 类
→ extractor.extract_all_images(...)
extract_all_images 在 src/limap/line2d/base_detector.py 的 BaseDetector 基类（ src/limap/line2d/base_detector.py ）里，里面有调用 self.extract()，但是被 继承自 BaseDetector 的 WireframeExtractor 重写，其中包含 descinfo = self.compute_descinfo(img, segs) 所以实际只用看 src/limap/line2d/GlueStick/extractor.py 中的 compute_descinfo