# -*- coding: utf-8 -*-
"""实时推理：所有归一化参数从模型包里读，不靠记性。

用法：
    python predict.py                  # 摄像头实时
    python predict.py --check          # 不开摄像头，用 test.csv 验一遍
    python predict.py --model other.joblib --cam 1

模型包由 train.py 生成，里面带 align / min_hand_px / feature_version，
推理端照做即可——这就是「隐性依赖落到文件里」。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

if not sys.stdout.isatty():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "out")

REQUIRED = ("model", "classes", "align", "min_hand_px", "feature_version", "dim")

MIN_SIDE_CONF = 0.80   # 左右手判定低于此值就跳过，因为镜像方向错了必错


def load_bundle(path):
    import joblib

    if not os.path.exists(path):
        sys.exit(f"找不到模型包 {path}，先跑 train.py --align")
    bundle = joblib.load(path)
    missing = [k for k in REQUIRED if k not in bundle]
    if missing:
        sys.exit(f"模型包缺字段 {missing}，重新跑 train.py 生成")
    if bundle["feature_version"] != 2:
        sys.exit(f"模型包是 v{bundle['feature_version']}，当前 gesture_features.py 只出 v2")
    print(f"[model] {len(bundle['classes'])} 类 {bundle['classes']}")
    print(f"[model] align={bundle['align']}  min_hand_px={bundle['min_hand_px']:g}  "
          f"train_side={bundle.get('train_side', 'left')}  "
          f"test_acc={bundle.get('test_acc', '?')}")
    return bundle


def make_feature(bundle, pts, width, height, mirror=False):
    from gesture_features import extract

    feat = extract(pts, width=width, height=height,
                   align=bundle["align"], min_hand_px=bundle["min_hand_px"],
                   mirror=mirror)
    if feat is None:
        return None
    if feat.size != bundle["dim"]:
        sys.exit(f"特征维度 {feat.size} 与模型 {bundle['dim']} 不一致")
    return feat


def predict(bundle, feat):
    proba = bundle["model"].predict_proba(feat[None, :])[0]
    order = np.argsort(proba)[::-1]
    classes = bundle["classes"]
    return classes[order[0]], float(proba[order[0]]), classes[order[1]], float(proba[order[1]])


def check(bundle, path):
    """不开摄像头，用带标签的 CSV 验一遍，确认「训练→落盘→加载→推理」这条链没断。"""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    ok = 0
    conf = {}
    for row in rows:
        feat = np.asarray([float(v) for v in row[1:]], np.float32)
        pred, p, _, _ = predict(bundle, feat)
        if pred == row[0]:
            ok += 1
        else:
            conf[(row[0], pred)] = conf.get((row[0], pred), 0) + 1
    acc = ok / max(1, len(rows))
    print(f"回放 {os.path.basename(path)}: {ok}/{len(rows)} = {acc:.1%}")
    for (t, p), n in sorted(conf.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {t} → {p}: {n}")
    return acc


def run_camera(bundle, cam, width, height, flip=True):
    import cv2

    from collect import HandTracker, draw_hand  # 复用采集端的检测/绘制
    from gesture_features import decide_mirror

    train_side = bundle.get("train_side", "left")

    cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW if os.name == "nt" else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        sys.exit(f"打不开摄像头 {cam}")

    tracker = HandTracker()
    print(f"[camera] mediapipe API: {tracker.kind}")
    win = "gesture predict"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if flip:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            pts, side, side_score = tracker.detect_with_side(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if pts is None:
                cv2.putText(frame, "NO HAND", (16, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (60, 60, 255), 2, cv2.LINE_AA)
            else:
                draw_hand(frame, pts, w, h)
                mirror, side_ok = decide_mirror(side, side_score, train_side, MIN_SIDE_CONF)
                if not side_ok:
                    # 左右手判不准时不做判定：猜错方向比认不出更糟
                    cv2.putText(frame, f"side ? ({side_score:.2f})", (16, 44),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 140, 255), 2, cv2.LINE_AA)
                    feat = None
                else:
                    feat = make_feature(bundle, pts, w, h, mirror=mirror)
                if not side_ok:
                    pass
                elif feat is None:
                    cv2.putText(frame, "too small / too far", (16, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 140, 255), 2, cv2.LINE_AA)
                else:
                    label, p, alt, pa = predict(bundle, feat)
                    cv2.putText(frame, f"{label}  {p:.0%}", (16, 44),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 230, 80), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"2nd: {alt} {pa:.0%}   side: {side}"
                                       f"{'  mirrored' if mirror else ''}",
                                (16, 78),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow(win, frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(DATA_DIR, "gesture_clf.joblib"))
    ap.add_argument("--check", nargs="?", const=os.path.join(OUT_DIR, "test.csv"),
                    help="不开摄像头，用 CSV 回放验证；缺省用 test.csv")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--no-flip", action="store_true")
    args = ap.parse_args()

    bundle = load_bundle(args.model)
    if args.check:
        acc = check(bundle, args.check)
        sys.exit(0 if acc >= 0.8 else 1)
    run_camera(bundle, args.cam, args.width, args.height, not args.no_flip)


if __name__ == "__main__":
    main()
