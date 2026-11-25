# -*- coding: utf-8 -*-
"""21 个手部关键点 -> 定长特征向量。

v2 改动（相对 v1）：
  * 先乘回像素坐标再算（width/height），消除「x 按宽、y 按高」的非等比拉伸。
    同一只手换 16:9 摄像头，特征分布不会漂。
  * 增加 min_hand_px 门控：掌长（腕→中指根）低于阈值直接判不可信。
    归一化空间里手太小会让特征尺度爆炸（v1 那 22 条 |x|>3 就是这么来的）。
  * 不传 width/height 时退化为 v1 行为（老代码不会突然坏掉）。

采集端 collect.py 和训练端必须 import 同一个函数，否则线下准、上线崩。

自检：
    python gesture_features.py
"""
from __future__ import annotations

import sys

import numpy as np

if not sys.stdout.isatty():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PALM_IDX = (0, 5, 9, 13, 17)
WRIST, MIDDLE_MCP = 0, 9


def landmark_array(hand_landmarks) -> np.ndarray:
    """mediapipe 的 hand_landmarks -> (21, 3) float32，归一化图像坐标。"""
    pts = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32)
    if pts.shape != (21, 3):
        raise ValueError(f"期望 21 个关键点，实际 {pts.shape[0]} 个")
    return pts


def to_pixels(points, width: float, height: float) -> np.ndarray:
    """归一化坐标 -> 像素坐标（z 按宽度缩放，保持和 x 同量纲）。"""
    pts = np.asarray(points, dtype=np.float32).copy()
    pts[:, 0] *= float(width)
    pts[:, 1] *= float(height)
    pts[:, 2] *= float(width)
    return pts


def decide_mirror(side, score, train_side="left", min_score=0.80):
    """根据 handedness 决定要不要镜像。

    返回 (mirror, confident)：
      mirror=True      这只手和训练集不是同一只，需要翻转 x
      confident=False  左右手判不准（双手交叉 / 遮挡 / 侧对镜头），
                       调用方应当跳过判定，而不是猜一个方向

    train_side 是训练集用的那只手（MediaPipe 报出来的名字）。
    注意 MediaPipe 的 handedness 假设输入是镜像自拍画面，
    所以这里只比「同不同」，不假设 Left 一定对应你的左手。
    """
    if not side or score < min_score:
        return False, False
    return str(side).strip().lower() != str(train_side).strip().lower(), True


def hand_length(points, width: float | None = None, height: float | None = None) -> float:
    """掌长 = ‖中指根 − 手腕‖。给了 width/height 就是像素长度。"""
    pts = np.asarray(points, dtype=np.float32)
    if width is not None and height is not None:
        pts = to_pixels(pts, width, height)
    return float(np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST]))


def extract(points, width=None, height=None, align=False, use_z=False,
            min_hand_px=0.0, mirror=False):
    """(21,3) -> 1D float32 特征向量。

    归一化流程：
      0. 给了 width/height 就先乘回像素坐标（各向同性）
      0b. mirror=True 时把 x 取反（右手 → 镜像成左手的样子）
      1. min_hand_px 门控：掌长太短说明手太远/检测飘，返回 None
      2. 除以掌长            => 与远近无关
      3. 减去手掌五点质心    => 与画面位置无关
      4. align=True 时旋转到中指朝上 => 与手的倾斜无关
    退化帧返回 None。

    mirror 用在哪：训练集只有一只手时，另一只手做同样的动作，
    21 点在画面里是镜像关系；把 x 取反后就能和训练集同分布。
    要不要镜像由 handedness 决定，见 decide_mirror()。
    """
    pts = np.asarray(points, dtype=np.float32).copy()
    if pts.shape != (21, 3):
        raise ValueError(f"期望 (21,3)，拿到 {pts.shape}")

    if width is not None and height is not None:
        pts = to_pixels(pts, width, height)

    if mirror:
        pts[:, 0] = -pts[:, 0]

    scale = float(np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST]))
    if not np.isfinite(scale) or scale < 1e-6:
        return None
    if min_hand_px > 0 and scale < min_hand_px:
        return None

    pts /= scale
    pts -= pts[list(PALM_IDX)].mean(axis=0)

    if align:
        v = pts[MIDDLE_MCP] - pts[WRIST]
        delta = -np.pi / 2.0 - np.arctan2(v[1], v[0])
        c, s = np.cos(delta), np.sin(delta)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        pts[:, :2] = pts[:, :2] @ rot.T

    feat = pts if use_z else pts[:, :2]
    return np.ascontiguousarray(feat.reshape(-1), dtype=np.float32)


def normalize_from_v1(feat42, width: float, height: float) -> np.ndarray:
    """把 v1（归一化空间）的特征转成 v2（像素空间）特征。

    两个空间只差对角矩阵 D=diag(w,h)：
        feat_px = D·feat / ‖D·(feat9 − feat0)‖
    尺度因子在分子分母里约掉，所以不需要知道原始 hand_size。
    """
    f = np.asarray(feat42, dtype=np.float32).reshape(21, 2).copy()
    f[:, 0] *= float(width)
    f[:, 1] *= float(height)
    denom = float(np.linalg.norm(f[MIDDLE_MCP] - f[WRIST]))
    if denom < 1e-6:
        return None
    f /= denom
    f -= f[list(PALM_IDX)].mean(axis=0)
    return np.ascontiguousarray(f.reshape(-1), dtype=np.float32)


def feature_names(use_z: bool = False):
    dims = "xyz" if use_z else "xy"
    return [f"{d}{i}" for i in range(21) for d in dims]


def _self_test() -> None:
    rng = np.random.default_rng(0)
    pts = rng.random((21, 3)).astype(np.float32)
    pts[9] = pts[0] + np.array([0.0, 0.25, 0.0], np.float32)

    base = extract(pts)
    print("特征维度 :", base.size, "(use_z=False 应为 42)")

    # 1) 正方形画面：像素空间必须和 v1 完全一致
    same = extract(pts, 480, 480)
    print("w=h=480 与 v1 差异 :", round(float(np.linalg.norm(base - same)), 8))

    # 2) 4:3 画面：必须不同（这就是修复的拉伸）
    wide = extract(pts, 640, 480)
    print("w=640,h=480 与 v1 差异 :", round(float(np.linalg.norm(base - wide)), 4))

    # 3) 迁移公式：v1 -> 像素空间，应与直接算一致
    conv = normalize_from_v1(base, 640, 480)
    print("迁移公式 vs 直接算 :", round(float(np.linalg.norm(conv - wide)), 8))

    # 4) 平移+缩放不变
    moved = extract(pts * 0.35 + np.array([0.5, 0.2, 0.0], np.float32), 640, 480)
    print("平移+缩放 L2 差 :", round(float(np.linalg.norm(wide - moved)), 6))

    # 5) 旋转：必须在像素空间里转，才算物理旋转
    ang = np.deg2rad(25.0)
    c, s = np.cos(ang), np.sin(ang)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
    P = to_pixels(pts, 640, 480) @ rot.T
    spun = P.copy()
    spun[:, 0] /= 640.0
    spun[:, 1] /= 480.0
    spun[:, 2] /= 640.0
    print("旋转25° 已对齐 L2 差 :",
          round(float(np.linalg.norm(extract(pts, 640, 480, align=True)
                                     - extract(spun, 640, 480, align=True))), 6))

    # 7) 镜像：右手做同样的动作 = 左手镜像，翻回来应与左手特征一致
    right = pts.copy()
    right[:, 0] = -right[:, 0]
    m_left = extract(pts, 640, 480, align=True)
    m_right = extract(right, 640, 480, align=True, mirror=True)
    print("镜像还原后与左手 L2 差 :", round(float(np.linalg.norm(m_left - m_right)), 6))

    # 8) handedness 门控
    mir, ok = decide_mirror("Right", 0.95)
    print(f"decide_mirror('Right', 0.95) -> mirror={mir}, confident={ok}")
    mir2, ok2 = decide_mirror("Right", 0.55)
    print(f"decide_mirror('Right', 0.55) -> mirror={mir2}, confident={ok2}")

    # 6) min_hand_px 门控
    small = hand_length(pts, 640, 480)
    print(f"掌长 {small:.0f}px, min_hand_px={small + 10:.0f} 时应为 None:",
          extract(pts, 640, 480, min_hand_px=small + 10) is None)

    assert base.size == 42 and base.dtype == np.float32
    assert np.allclose(base, same, atol=1e-6)
    assert np.linalg.norm(base - wide) > 0.05
    assert np.allclose(conv, wide, atol=1e-5)
    assert np.linalg.norm(wide - moved) < 1e-4
    assert extract(pts, 640, 480, min_hand_px=small + 10) is None
    assert np.linalg.norm(m_left - m_right) < 1e-5
    assert decide_mirror("Right", 0.95) == (True, True)
    assert decide_mirror("left", 0.95) == (False, True)
    assert decide_mirror("Right", 0.55) == (False, False)
    assert extract(np.zeros((21, 3), np.float32)) is None
    assert extract(pts, use_z=True).size == 63
    print("自检通过 [OK]")


if __name__ == "__main__":
    _self_test()
