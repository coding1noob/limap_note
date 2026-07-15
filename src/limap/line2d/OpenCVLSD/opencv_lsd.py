import cv2
import numpy as np

from ..base_detector import (
    BaseDetector,
    DefaultDetectorOptions,
)

params = {
    "_refine"     : 1,     # 默认1     精炼方式（0=不精炼, 1=标准, 2=高级）
    "_scale"      : 2.0,   # 默认0.8   缩放比例。1.0：原图分辨率检测
    "_sigma_scale": 0.4,   # 默认0.6   模糊强度。值越大，抗噪但细线被抹掉
    "_quant"      : 2.0,   # 默认2.0   梯度量化误差上限。值越大 → 对梯度方向要求宽松 → 检测更多线但可能不准
    "_ang_th"     : 22.5,  # 默认22.5  角度容差（度）。值越大 → 方向稍歪的像素也归为同一条线 → 线更容易被检测到
    "_log_eps"    : -1.0,  # 默认0.0   灵敏度，-1.0比默认的0.0更宽松
    "_density_th" : 0.5,   # 默认0.7   线段点密度下限。一条候选线段上，实际对齐的像素点占总像素的比例下限
    "_n_bins"     : 1024,  # 默认1024  梯度排序分箱数
}

class OpenCVLSDDetector(BaseDetector):
    def __init__(self, options=DefaultDetectorOptions):
        super().__init__(options)
        self.lsd = cv2.createLineSegmentDetector(
            params["_refine"],
            params["_scale"],
            params["_sigma_scale"],
            params["_quant"],
            params["_ang_th"],
            params["_log_eps"],
            params["_density_th"],
            params["_n_bins"],
        )

    def get_module_name(self):
        return "opencv_lsd"

    def detect(self, camview):
        img = camview.read_image(set_gray=self.set_gray)
        lines, width, prec, nfa = self.lsd.detect(img)
        if lines is None:
            return np.zeros((0, 5))

        # lines: (N, 4) -> x1, y1, x2, y2
        lines = lines.reshape(-1, 4)

        # 用线段长度作为 score，归一化到 [0, 1]
        lengths = np.sqrt(
            (lines[:, 2] - lines[:, 0]) ** 2 +
            (lines[:, 3] - lines[:, 1]) ** 2
        )
        max_len = lengths.max() if lengths.max() > 0 else 1.0
        scores = lengths / max_len

        segs = np.concatenate([lines, scores[:, None]], axis=1)
        return segs