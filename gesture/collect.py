# -*- coding: utf-8 -*-
"""近景手势采集：摄像头 -> MediaPipe 21 点 -> 归一化特征向量 -> CSV

运行：
    python collect.py --preset single          # 单指 4 类
    python collect.py --preset combo           # 组合 10 类
    python collect.py --preset all             # 全部 14 类
    python collect.py --labels 食指,中指 --target 500

键盘：
    0-9 / a d f g   选中该类并开始自动采集（再按同一个键 = 暂停/继续）
    SPACE           暂停 / 继续
    s               补录单帧（用同一套 min-move 过滤，太近会提示不记）
    u               撤销上一条
    q / ESC         保存并退出

设计要点：
  * 自动采集按 --interval 节流（默认 0.12s ≈ 8 帧/秒），不用狂按键盘
  * --min-move 过滤掉「手没动」的重复样本，避免 500 条里 480 条一模一样
  * 兼容 mediapipe 新旧两套 API：0.10.2x 走 legacy solutions，0.10.3x+ 走 tasks
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("缺少 opencv，先跑：pip install -r requirements.txt")

try:
    import mediapipe as mp
except ImportError:
    sys.exit("缺少 mediapipe，先跑：pip install -r requirements.txt")

from gesture_features import extract, feature_names, hand_length

if not sys.stdout.isatty():          # 输出被重定向时统一成 UTF-8，避免中文乱码
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(HERE, "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

CJK_FONTS = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\Deng.ttf",
]

HINT = ("0-9/a d f g 选类   空格 暂停   s 单帧   u 撤销   q 退出",
        "0-9/a d f g class  SPACE pause  s shot  u undo  q quit")

# 类别键位：前 10 类用数字，11 类起用 a/d/f/g（避开 s/u/q 这些功能键）
CLASS_KEYS = "0123456789adfg"

# 内置类表（中文名直接写进 CSV，训练和推理都按这个名字走）
PRESETS = {
    "single": ["食指", "中指", "无名指", "小指"],
    "combo": ["点赞", "OK", "张开", "比耶", "握拳", "六",
              "点踩", "后三指", "中三指", "比心"],
}
PRESETS["all"] = PRESETS["single"] + PRESETS["combo"]


def ensure_model() -> str:
    """tasks 版需要这个 .task 模型文件，只在第一次运行下载一次。"""
    if os.path.exists(MODEL_FILE) and os.path.getsize(MODEL_FILE) > 1_000_000:
        return MODEL_FILE
    print("首次运行：下载手部模型 hand_landmarker.task（约 8MB）…")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    except Exception as exc:  # 网络问题就直接说清楚，别让人猜
        sys.exit(f"下载失败：{exc}\n请手动下载 {MODEL_URL}\n"
                 f"另存为 {MODEL_FILE} 后重新运行")
    return MODEL_FILE


class HandTracker:
    """统一接口：给一帧 RGB，返回 (21,3) 关键点或 None。"""

    def __init__(self):
        self._t0 = time.time()
        if hasattr(mp, "solutions"):                      # 旧 API
            self.kind = "solutions"
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:                                             # 新 API（tasks）
            self.kind = "tasks"
            from mediapipe.tasks import python as mp_py
            from mediapipe.tasks.python import vision

            opts = vision.HandLandmarkerOptions(
                base_options=mp_py.BaseOptions(model_asset_path=ensure_model()),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._hands = vision.HandLandmarker.create_from_options(opts)

    def detect(self, rgb):
        if self.kind == "solutions":
            res = self._hands.process(rgb)
            if not res.multi_hand_landmarks:
                return None
            lms = res.multi_hand_landmarks[0].landmark
        else:
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=np.ascontiguousarray(rgb))
            ts = int((time.time() - self._t0) * 1000)
            res = self._hands.detect_for_video(img, ts)
            if not res.hand_landmarks:
                return None
            lms = res.hand_landmarks[0]
        return np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)

    def close(self):
        try:
            self._hands.close()
        except Exception:
            pass


def draw_hand(frame, pts, w, h):
    px = [(int(x * w), int(y * h)) for x, y, _ in pts]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, px[a], px[b], (0, 200, 0), 2, cv2.LINE_AA)
    for p in px:
        cv2.circle(frame, p, 3, (0, 255, 255), -1, cv2.LINE_AA)


class Overlay:
    """中文 HUD：优先用系统字体渲染，失败就退回英文，别让画面上出现方块。"""

    def __init__(self):
        self.font = None
        try:
            from PIL import Image, ImageDraw, ImageFont

            self._pil = (Image, ImageDraw, ImageFont)
            for path in CJK_FONTS:
                if os.path.exists(path):
                    self.font = ImageFont.truetype(path, 22)
                    break
        except Exception:
            self._pil = None

    def put(self, frame, lines):
        """lines: [(中文, 英文, (B,G,R)), ...]"""
        if self.font is None:
            y = 26
            for _, en, color in lines:
                cv2.putText(frame, en, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, en, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 1, cv2.LINE_AA)
                y += 24
            return

        Image, ImageDraw, _ = self._pil
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        y = 8
        for zh, _, color in lines:
            rgb = (color[2], color[1], color[0])
            draw.text((13, y + 1), zh, font=self.font, fill=(0, 0, 0))
            draw.text((12, y), zh, font=self.font, fill=rgb)
            y += 28
        frame[:] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def parse_args():
    p = argparse.ArgumentParser(description="近景手势数据采集（21 点特征向量）")
    p.add_argument("--preset", choices=["single", "combo", "all"], default=None,
                   help="内置类表：single=单指 4 类, combo=组合 10 类, all=14 类")
    p.add_argument("--labels", default="ok,peace,fist,point",
                   help="逗号分隔的类别名（--preset 优先）；最多 14 个")
    p.add_argument("--out", default=os.path.join(HERE, "data", "gesture_data.csv"),
                   help="输出 CSV 路径")
    p.add_argument("--target", type=int, default=300, help="每类目标样本数")
    p.add_argument("--interval", type=float, default=0.12, help="自动采集最小间隔（秒）")
    p.add_argument("--min-move", type=float, default=0.05, help="特征变化小于此值则丢弃（去冗余）")
    p.add_argument("--cam", type=int, default=0, help="摄像头编号")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--align", action="store_true", help="旋转归一化：让手掌朝向统一（手会歪着摆就开）")
    p.add_argument("--use-z", action="store_true", help="把 z 也存进特征（63 维；z 噪声大，一般不开）")
    p.add_argument("--min-hand-px", type=float, default=80.0,
                   help="掌长（腕→中指根）低于这个像素数就跳过，防尺度爆炸")
    p.add_argument("--no-flip", action="store_true", help="不镜像画面")
    return p.parse_args()


def load_existing(path, dim):
    rows, counts = [], {}
    if not os.path.exists(path):
        return rows, counts
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) == 1 + dim:
                rows.append(row)
                counts[row[0]] = counts.get(row[0], 0) + 1
    return rows, counts


def save(path, rows, names):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + names)
        writer.writerows(rows)
    os.replace(tmp, path)


def main():
    args = parse_args()
    if args.preset:
        labels = list(PRESETS[args.preset])
    else:
        labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    labels = labels[:len(CLASS_KEYS)]
    if not labels:
        sys.exit("--labels 不能为空")
    print(f"类表({len(labels)}): " + "  ".join(
        f"{CLASS_KEYS[i]}={n}" for i, n in enumerate(labels)))

    names = feature_names(args.use_z)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows, counts = load_existing(args.out, len(names))
    if rows:
        print(f"续采：{args.out} 已有 {len(rows)} 条")

    # 追加式审计日志：每写入/撤销/拒绝都记一行，事后能复盘参数与来源
    trace_path = os.path.splitext(os.path.abspath(args.out))[0] + "_trace.csv"
    trace_is_new = (not os.path.exists(trace_path)) or os.path.getsize(trace_path) == 0
    trace_f = open(trace_path, "a", newline="", encoding="utf-8")
    trace_w = csv.writer(trace_f)
    if trace_is_new:
        trace_w.writerow(["time", "event", "label", "dist", "note"])
        trace_f.flush()

    def trace_log(event, label, dist=None, note=""):
        trace_w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), event, label or "",
                          "" if dist is None else f"{dist:.4f}", note])
        trace_f.flush()

    with open(os.path.join(os.path.dirname(os.path.abspath(args.out)), "collect_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "out": os.path.abspath(args.out),
            "align": bool(args.align), "use_z": bool(args.use_z),
            "flip": not args.no_flip, "labels": labels, "dim": len(names),
            # 采集参数也要记，不然事后没法复核“这条为什么没被过滤”
            "interval": args.interval, "min_move": args.min_move,
            "target": args.target, "cam": args.cam,
            "min_hand_px": args.min_hand_px,
            "frame_width": args.width, "frame_height": args.height,
            "feature_version": 2,
            "resume_rows": len(rows),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, ensure_ascii=False, indent=2)

    # 数据文件自己的版本标记：check_data.py 优先读它，不依赖 collect_meta.json 还在不在
    with open(os.path.abspath(args.out) + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"feature_version": 2, "width": args.width, "height": args.height,
                   "rows_at_start": len(rows),
                   "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  f, ensure_ascii=False, indent=2)

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW if os.name == "nt" else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f"打不开摄像头 {args.cam}，换个 --cam 编号试试")

    tracker = HandTracker()
    print(f"mediapipe API: {tracker.kind}")

    hud = Overlay()
    state = {"label": None, "auto": False, "last_vec": None, "last_t": 0.0,
             "fps": 0.0, "msg": "", "msg_until": 0.0}
    dirty = False
    t_prev = time.time()
    win = "gesture collect"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("读帧失败，摄像头被占用？")
                break
            if not args.no_flip:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            pts = tracker.detect(rgb)
            feat = None
            px = 0.0
            if pts is not None:
                draw_hand(frame, pts, w, h)
                px = hand_length(pts, w, h)
                feat = extract(pts, width=w, height=h, align=args.align,
                               use_z=args.use_z, min_hand_px=args.min_hand_px)

            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                state["fps"] = 0.9 * state["fps"] + 0.1 / dt

            label = state["label"]
            if feat is not None and label is not None and state["auto"]:
                if now - state["last_t"] >= args.interval and counts.get(label, 0) < args.target:
                    gap = (float("inf") if state["last_vec"] is None
                           else float(np.linalg.norm(feat - state["last_vec"])))
                    if gap >= args.min_move:
                        rows.append([label] + [f"{v:.5f}" for v in feat])
                        counts[label] = counts.get(label, 0) + 1
                        state["last_vec"] = feat
                        state["last_t"] = now
                        dirty = True
                        trace_log("auto", label, gap, f"px={px:.0f}")

            # ---- HUD ----
            active = counts.get(label, 0) if label else 0
            if label is None:
                head = ("未选类别    ", "no class   ", (200, 200, 200))
            elif state["auto"]:
                head = (f"{label}  {active}/{args.target}  录制中 ●",
                        f"{label}  {active}/{args.target}  REC ●", (60, 220, 60))
            else:
                head = (f"{label}  {active}/{args.target}  已暂停 ‖",
                        f"{label}  {active}/{args.target}  PAUSE ‖", (60, 200, 255))

            cls_lines, chunk = [], []
            for i, n in enumerate(labels):
                chunk.append(f"{CLASS_KEYS[i]}:{n}·{counts.get(n, 0)}")
                if len(chunk) == 7:            # 14 类一行放不下，7 个折一行
                    cls_lines.append("  ".join(chunk))
                    chunk = []
            if chunk:
                cls_lines.append("  ".join(chunk))
            if pts is not None and feat is None:
                info = (f"手太小 {px:.0f}px < {args.min_hand_px:.0f}，跳过",
                        f"too small {px:.0f}px", (60, 120, 255))
            else:
                info = (f"手:{1 if feat is not None else 0}  FPS:{state['fps']:.0f}"
                        f"  掌长:{px:.0f}px  维度:{len(names)}",
                        f"hand:{1 if feat is not None else 0}  fps:{state['fps']:.0f}"
                        f"  palm:{px:.0f}px  dim:{len(names)}",
                        (230, 230, 230))
            hud.put(frame, [head] + [(c, c, (200, 220, 255)) for c in cls_lines]
                    + [info, (HINT[0], HINT[1], (210, 210, 210))])
            if now < state["msg_until"] and state["msg"]:
                cv2.putText(frame, state["msg"], (12, h - 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, state["msg"], (12, h - 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 200, 255), 1, cv2.LINE_AA)
            if feat is None:
                cv2.putText(frame, "NO HAND", (12, h - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2, cv2.LINE_AA)

            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break
            elif key == 32:
                state["auto"] = not state["auto"]
            elif key == ord("s"):
                if feat is not None and label is not None:
                    # 单帧补录也走同一套过滤，否则连按 s 就能塞进一堆重复样本
                    gap = (float("inf") if state["last_vec"] is None
                           else float(np.linalg.norm(feat - state["last_vec"])))
                    if gap >= args.min_move:
                        rows.append([label] + [f"{v:.5f}" for v in feat])
                        counts[label] = counts.get(label, 0) + 1
                        state["last_vec"] = feat
                        dirty = True
                        trace_log("manual", label, gap, f"px={px:.0f}")
                    else:
                        state["msg"] = f"太近了 {gap:.3f} < {args.min_move}，没记"
                        state["msg_until"] = now + 1.5
                        trace_log("manual_reject", label, gap, f"px={px:.0f}")
            elif key == ord("u"):
                if rows:
                    gone = rows.pop()[0]
                    counts[gone] = max(0, counts.get(gone, 0) - 1)
                    # 撤销后基准退回到“现在最后一条”，不是清空——
                    # 清空会让下一帧无条件写入，正好绕过 min-move
                    state["last_vec"] = (np.array([float(v) for v in rows[-1][1:]],
                                                  dtype=np.float32) if rows else None)
                    state["last_t"] = 0.0
                    dirty = True
                    trace_log("undo", gone)
            elif chr(key) in CLASS_KEYS:
                idx = CLASS_KEYS.index(chr(key))
                if idx < len(labels):
                    if state["label"] == labels[idx]:
                        state["auto"] = not state["auto"]
                    else:
                        state["label"] = labels[idx]
                        state["auto"] = True
                    state["last_vec"] = None

            if dirty and len(rows) % 200 == 0:
                save(args.out, rows, names)
                dirty = False

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        try:
            trace_f.close()
        except Exception:
            pass
        if rows:
            save(args.out, rows, names)

    print("\n=== 结果 ===")
    for i, name in enumerate(labels):
        print(f"  {i}: {name:<12} {counts.get(name, 0):>5} 条")
    other = sum(v for k, v in counts.items() if k not in labels)
    if other:
        print(f"  其他标签            {other:>5} 条")
    print(f"总计 {len(rows)} 条 -> {os.path.abspath(args.out)}")
    short = [n for n in labels if counts.get(n, 0) < args.target]
    if short:
        print(f"还差：{', '.join(short)}（再跑一次会自动续采）")


if __name__ == "__main__":
    main()
