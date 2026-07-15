# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import os
from pathlib import Path

import torch
from pycolmap import logging
from torch import nn


def simple_nms(scores, nms_radius: int):
    """Fast Non-maximum suppression to remove nearby points"""
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """Removes keypoints too close to the border"""
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor(
        [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
    ).to(keypoints)[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors,
        keypoints.view(b, 1, -1, 2),
        mode="bilinear",
        align_corners=False,
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """

    default_config = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
        "remove_borders": 4,
        "weight_path": None,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5,
            self.config["descriptor_dim"],
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if self.config["weight_path"] is None:
            path = Path(__file__).parent / "weights/superpoint_v1.pth"
        else:
            path = os.path.join(
                self.config["weight_path"],
                "point2d",
                "superpoint",
                "weights/superpoint_v1.pth",
            )
        if not os.path.isfile(path):
            self.download_model(path)
        self.load_state_dict(torch.load(str(path)))

        mk = self.config["max_keypoints"]
        if mk == 0 or mk < -1:
            raise ValueError('"max_keypoints" must be positive or "-1"')

        logging.info("Loaded SuperPoint model")

    def download_model(self, path):
        import subprocess

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        link = "https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/superpoint_v1.pth?raw=true"
        cmd = ["wget", link, "-O", path]
        logging.info("Downloading SuperPoint model...")
        subprocess.run(cmd, check=True)

    # 实际调用的入口
    def compute_dense_descriptor(self, data):
        """Compute keypoints, scores, descriptors for image"""
        # Shared Encoder（编码器：把原始图像压缩成一组更有意义的特征）
        # __init__中有定义： conv1a 为 输入1个通道（灰度图），输出64个通道的卷积层，
        # conv1b 则为 输入64通道，输出64通道 的卷积层
        x = self.relu(self.conv1a(data["image"]))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # x 是一个 4维的 PyTorch 张量（Tensor），形状是：(batch, 通道数, 高, 宽)
        # 把原图的每个 8×8 小块，压缩成了一个 128 维的向量。整张图就变成了 60×80 个这样的向量。
        # 每个向量描述的是"这个区域长什么样"——有没有边缘、角点、纹理等

        # Compute the dense keypoint scores
        # 在编码器理解的基础上，专门判断"哪里是关键点"
        cPa = self.relu(self.convPa(x))                             # 128通道 → 256通道
        scores = self.convPb(cPa)                                   # 256通道 → 65通道
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]     # scores 为 每个像素的关键点概率
        b, _, h, w = scores.shape
        # 还原到原始分辨率
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = simple_nms(scores, self.config["nms_radius"])

        # Extract keypoints
        # 找出得分超过阈值的像素位置，s > 0.005 把得分图变成一张布尔图：
        # 0.001  0.8   0.003        False  True  False
        # 0.02   0.001 0.9    →     True   False True
        # 0.003  0.7   0.001        False  True  False
        # torch.nonzero 找出所有 True 的位置坐标，比如：
        # → [[0, 1],   # 第0行第1列
        # [1, 0],   # 第1行第0列
        # [1, 2],   # 第1行第2列
        # [2, 1]]   # 第2行第1列
        keypoints = [
            torch.nonzero(s > self.config["keypoint_threshold"]) for s in scores
        ]
        # scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        # 图像边缘4个像素以内的关键点直接丢掉
        keypoints, scores = list(
            zip(
                *[
                    remove_borders(
                        k, s, self.config["remove_borders"], h * 8, w * 8
                    )
                    for k, s in zip(keypoints, scores)
                ]
            )
        )

        # Keep the k keypoints with highest score
        # 只保留得分最高的 k 个
        if self.config["max_keypoints"] >= 0:
            keypoints, scores = list(
                zip(
                    *[
                        top_k_keypoints(k, s, self.config["max_keypoints"])
                        for k, s in zip(keypoints, scores)
                    ]
                )
            )

        # Convert (h, w) to (x, y)
        # 坐标从 (行,列) 转成 (x,y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Compute the dense descriptors
        # 计算密集描述子
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
        # keypoints — 关键点坐标列表。keypoints[0]是第一张图的关键点坐标，每行是一个点的 (x, y) 像素坐标。[[312, 48], [107, 203], ...]
        # scores — 每个关键点的置信度得分。和 keypoints 一一对应，每个值是 0~1 之间的数
        # descriptors — 密集描述子。形状是 (1, 256, H/8, W/8)
        return keypoints, scores, descriptors

    def compute_dense_descriptor_and_score(self, data):
        """Compute dense scores and descriptors for an image"""
        # Shared Encoder
        x = self.relu(self.conv1a(data["image"]))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        dense_descriptor = torch.nn.functional.normalize(
            descriptors, p=2, dim=1
        )
        return {"dense_score": scores, "dense_descriptor": dense_descriptor}

    def sample_descriptors(self, data, keypoints):
        _, _, descriptors = self.compute_dense_descriptor(data)

        # Extract descriptors
        descriptors = [
            sample_descriptors(k[None], d[None], 8)[0]
            for k, d in zip(keypoints, descriptors)
        ]

        return {"keypoints": keypoints, "descriptors": descriptors}

    def forward(self, data):
        keypoints, scores, descriptors = self.compute_dense_descriptor(data)

        # Extract descriptors
        descriptors = [
            sample_descriptors(k[None], d[None], 8)[0]
            for k, d in zip(keypoints, descriptors)
        ]

        return {
            "keypoints": keypoints,
            "scores": scores,
            "descriptors": descriptors,
        }
