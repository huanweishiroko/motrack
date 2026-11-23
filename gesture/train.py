# -*- coding: utf-8 -*-
"""过滤 → 抽稀 → 按时间切 → 训练 → 混淆矩阵。

输入是 collect.py 采的 gesture_data.csv（v2 像素空间特征）。
切分按行号（= 采集时间）来，不打乱。

用法：
    python train.py                 # 旋转归一化默认开启（推荐，直接跑这个）
    python train.py --no-align      # 关掉，仅做对照实验
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys
import time

import joblib
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

if not sys.stdout.isatty():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "out")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = []


def say(line=""):
    print(line)
    OUT.append(str(line))


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        labels, feats = [], []
        for row in reader:
            if len(row) != len(header):
                continue
            labels.append(row[0])
            feats.append([float(v) for v in row[1:]])
    return np.asarray(feats, np.float32), np.asarray(labels), header


def align_rows(X):
    """对已存的 42 维特征做旋转归一化：把「腕→中指根」转到朝上。

    采集时没开 --align 也能事后补，因为旋转是作用在特征点上的纯 2D 旋转。
    注意：用这份数据训出来的模型，推理时必须 extract(..., align=True)。
    """
    if X.shape[1] != 42:
        raise ValueError("align_rows 只支持 42 维（use_z=False）的特征")
    pts = X.reshape(-1, 21, 2)
    out = np.empty_like(pts)
    for i in range(len(pts)):
        v = pts[i, 9] - pts[i, 0]
        d = -np.pi / 2 - np.arctan2(v[1], v[0])
        c, s = np.cos(d), np.sin(d)
        rot = np.array([[c, -s], [s, c]], np.float32)
        out[i] = pts[i] @ rot.T
    return out.reshape(X.shape)


def time_split(X, y, classes, train_ratio=0.6, gap_ratio=0.1):
    """每类按行号（= 采集时间）切：前 60% 训练，中间 10% 丢掉，后 30% 测试。

    中间那 10% 是隔离带：不加的话训练集最后一条和测试集第一条只差 0.12 秒，
    等于让模型提前看到了考卷。不许打乱。
    """
    tr, te = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        n = len(idx)
        cut_tr = max(1, int(round(n * train_ratio)))
        cut_gap = max(cut_tr, int(round(n * (train_ratio + gap_ratio))))
        tr.extend(idx[:cut_tr])
        te.extend(idx[cut_gap:])
    return np.array(tr), np.array(te)


def evaluate(name, Xtr, ytr, Xte, yte, classes):
    results = {}
    for model_name, model in (
        # probability=True 才会算 predict_proba（Platt 标定，内部 5 折，训练慢几倍但样本小无所谓）
        ("SVC", make_pipeline(StandardScaler(),
                              SVC(C=10, gamma="scale", probability=True))),
        ("kNN", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))),
    ):
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        acc = accuracy_score(yte, pred)
        results[model_name] = (acc, pred, model)
        say(f"    {name:<22} {model_name:<4} acc={acc:6.1%}  (train={len(Xtr)} test={len(Xte)})")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1,
                    help="抽稀步长；样本少时用 1（不抽稀），样本多且冗余再调大")
    ap.add_argument("--clip", type=float, default=3.0, help="|x| 超过它就当垃圾样本丢掉")
    ap.add_argument("--trim", type=float, default=0.0,
                    help="每类首尾各丢这个比例，去掉换手势时的过渡帧（建议 0.1）")
    ap.add_argument("--input", default=os.path.join(DATA_DIR, "gesture_data.csv"))
    ap.add_argument("--tag", default="", help="输出文件后缀，比如 _px")
    ap.add_argument("--align", action="store_true",
                    help="兼容旧命令；rotate 已是默认行为，这个开关现在什么都不做")
    ap.add_argument("--no-align", action="store_true",
                    help="关掉旋转归一化（不建议；只在做对照实验时用）")
    ap.add_argument("--min-hand-px", type=float, default=80.0,
                    help="写进模型包的掌长门控，推理端照此执行；要和采集时一致")
    args = ap.parse_args()

    X, y, header = load(args.input)
    classes = list(dict.fromkeys(y.tolist()))
    say(f"原始数据: {len(X)} 条, {len(classes)} 类 {classes}")
    # 旋转归一化默认开：忘加参数会让准确率从 94.7% 掉到 86%，
    # 而且模型包里 align=False 会让推理端老实照做——这是最容易踩的坑。
    args.align = not args.no_align
    if args.align:
        X = align_rows(X)
        say("旋转归一化: 已开启（推理时必须 extract(..., align=True)）")
    else:
        say("旋转归一化: 已关闭（--no-align，仅用于对照实验）")

    # ---- 过滤 |x| > clip ----
    bad = np.abs(X).max(axis=1) > args.clip
    say("")
    say(f"第 3 步 · 过滤 |x| > {args.clip}")
    for c in classes:
        m = y == c
        say(f"    {c:<8} 命中 {int((bad & m).sum()):>3} / {int(m.sum())}")
    Xf, yf = X[~bad], y[~bad]
    say(f"    剩余 {len(Xf)} 条（丢弃 {int(bad.sum())} 条）")

    # ---- 抽稀 ----
    keep = []
    say("")
    say(f"第 4 步 · 每类按步长 k={args.k} 抽稀")
    for c in classes:
        idx = np.where(yf == c)[0]
        if args.trim > 0:
            cut = int(len(idx) * args.trim)
            if cut:
                idx = idx[cut:len(idx) - cut]
        picked = idx[::args.k]
        keep.extend(picked)
        say(f"    {c:<8} {len(idx):>3} → {len(picked):>3} 条")
    keep = np.array(keep)
    Xs, ys = Xf[keep], yf[keep]
    say(f"    合计 {len(Xs)} 条")

    # ---- 按时间切 ----
    tr, te = time_split(Xs, ys, classes)
    say("")
    say(f"第 5 步 · 时间切分 60/10/30（不打乱）train={len(tr)} "
        f"gap={len(Xs)-len(tr)-len(te)} test={len(te)}")
    for c in classes:
        n_tr = int(((ys[tr] == c)).sum())
        n_te = int(((ys[te] == c)).sum())
        say(f"    {c:<8} train {n_tr:>3} / test {n_te:>3}")

    # ---- 训练 ----
    say("")
    say("第 6 步 · 训练与评估")
    res = {}
    say("  [C] 过滤 + 抽稀 k=%d（主结果）" % args.k)
    res["slim"] = evaluate("slim", Xs[tr], ys[tr], Xs[te], ys[te], classes)

    say("  [B] 只过滤，不抽稀")
    tr_all, te_all = time_split(Xf, yf, classes)
    res["full"] = evaluate("full", Xf[tr_all], yf[tr_all], Xf[te_all], yf[te_all], classes)

    rng = np.random.default_rng(0)
    ytr_shuf = ys[tr].copy()
    rng.shuffle(ytr_shuf)
    say("  [D] 对照组：训练标签打乱（应≈25%）")
    evaluate("shuffled", Xs[tr], ytr_shuf, Xs[te], ys[te], classes)

    # ---- 混淆矩阵 ----
    say("")
    say("第 7 步 · 混淆矩阵（SVC，行=真实，列=预测）")
    acc, pred, clf = res["slim"]["SVC"]
    cm = confusion_matrix(ys[te], pred, labels=classes)
    width = max(len(c) for c in classes) + 2
    say("    " + " " * (width + 2) + "".join(f"{c:>{width}}" for c in classes))
    for i, c in enumerate(classes):
        say(f"    {c:<{width}}{'':2}" + "".join(f"{v:>{width}}" for v in cm[i]))

    pairs = collections.Counter()
    for t, p in zip(ys[te], pred):
        if t != p:
            pairs[(t, p)] += 1
    say("")
    if pairs:
        say("    误判分布：")
        for (t, p), n in pairs.most_common():
            say(f"      {t} → {p}: {n} 次")
    else:
        say("    测试集零误判——别急着高兴，看跨会话")

    # ---- 落盘 ----
    with open(os.path.join(OUT_DIR, f"gesture_data_slim{args.tag}.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(Xs)):
            w.writerow([ys[i]] + [f"{v:.5f}" for v in Xs[i]])

    for name, idx in (("train", tr), ("test", te)):
        with open(os.path.join(OUT_DIR, f"{name}{args.tag}.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in idx:
                w.writerow([ys[i]] + [f"{v:.5f}" for v in Xs[i]])

    say("")
    say(f"已写出: gesture_data_slim{args.tag}.csv / train{args.tag}.csv / test{args.tag}.csv"
        f"（k={args.k}, |x|<={args.clip}, trim={args.trim:g}）")

    # ---- 落盘模型：推理端只读这个包，不靠记性 ----
    bundle = {
        "model": clf,                       # 已 fit 的 pipeline（StandardScaler + SVC）
        "classes": [str(c) for c in clf.classes_],
        "align": bool(args.align),          # 推理必须照做
        "min_hand_px": float(args.min_hand_px),
        "feature_version": 2,
        "dim": int(Xs.shape[1]),
        "use_z": Xs.shape[1] == 63,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "test_acc": round(float(acc), 4),
        "input": os.path.basename(args.input),
        "sklearn": sklearn.__version__,
    }
    model_path = os.path.join(DATA_DIR, f"gesture_clf{args.tag}.joblib")
    joblib.dump(bundle, model_path)

    # 立刻回读一次，确保推理端拿到的东西是能用的
    chk = joblib.load(model_path)
    proba = chk["model"].predict_proba(Xs[te][:3])
    say(f"模型已存: {model_path}")
    say(f"  回读校验: predict_proba {proba.shape}, "
        f"classes={chk['classes']}, align={chk['align']}, "
        f"min_hand_px={chk['min_hand_px']:g}")

    with open(os.path.join(OUT_DIR, f"train_report{args.tag}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(OUT) + "\n")
    say(f"报告: {os.path.join(OUT_DIR, f'train_report{args.tag}.txt')}")


if __name__ == "__main__":
    main()
