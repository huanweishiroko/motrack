# -*- coding: utf-8 -*-
"""体检采集出来的 gesture_data.csv：条数、重复、离群、类别可分性。

用法：
    python check_data.py [csv路径]
"""
from __future__ import annotations

import collections
import csv
import json
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
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA_DIR, "gesture_data.csv")


def resolve_version(csv_path):
    """特征版本：sidecar > collect_meta.json > 未知。混用 v1/v2 是最隐蔽的掉点原因。"""
    side = csv_path + ".meta.json"
    if os.path.exists(side):
        try:
            with open(side, encoding="utf-8-sig") as f:   # 兼容别人用 PowerShell 写出的 BOM
                return json.load(f).get("feature_version", "?"), "sidecar"
        except Exception:
            pass
    meta = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "collect_meta.json")
    if os.path.exists(meta):
        try:
            with open(meta, encoding="utf-8-sig") as f:
                m = json.load(f)
            if os.path.basename(m.get("out", "")) == os.path.basename(csv_path):
                return m.get("feature_version", "?"), "collect_meta.json"
        except Exception:
            pass
    return "unknown", "未找到元数据"


labels, feats, bad_rows = [], [], 0
with open(path, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    dim = len(header) - 1
    for row in reader:
        if len(row) != dim + 1:
            bad_rows += 1
            continue
        try:
            feats.append([float(v) for v in row[1:]])
        except ValueError:
            bad_rows += 1
            continue
        labels.append(row[0])

X = np.asarray(feats, dtype=np.float32)
y = np.asarray(labels)
classes = list(dict.fromkeys(labels))
counts = collections.Counter(labels)

out = []


def say(line=""):
    print(line)
    out.append(str(line))


say(f"文件      : {path}")
say(f"行数      : {len(X)} 条（丢弃坏行 {bad_rows} 条，表头维度 {dim}）")
say(f"类别      : {len(classes)} 个 → {classes}")
say(f"数值健康  : NaN={int(np.isnan(X).sum())} inf={int(np.isinf(X).sum())} "
    f"|x|max={np.abs(X).max():.2f}")

version, version_src = resolve_version(os.path.abspath(path))
say(f"特征版本  : {version}（{version_src}）")
if version == "unknown":
    say("  ⚠ 认不出是 v1(归一化空间) 还是 v2(像素空间)。两者不能混，"
        "看下面横幅比是否双峰")
elif version == 2:
    say("  v2: 已按像素空间归一化，换分辨率不漂")

say("")
say("类别 | 条数 | 类内离散度 | 重复率 | 离群")
say("---|---|---|---")
dup_total = 0
for c in classes:
    idx = np.where(y == c)[0]
    m = X[idx]
    centroid = m.mean(axis=0)
    spread = float(np.linalg.norm(m - centroid, axis=1).mean())
    d = np.linalg.norm(m[:, None, :] - m[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    dup = float((nn < 0.01).mean())
    dup_total += int((nn < 0.01).sum())
    dist_centroid = np.linalg.norm(m - centroid, axis=1)
    thr = np.percentile(dist_centroid, 99)
    outl = int((dist_centroid > thr * 1.5).sum())
    say(f"{c} | {len(idx)} | {spread:.3f} | {dup:.1%} | {outl}")

say(f"重复样本合计: {dup_total} 条（与同类样本距离 <0.01，等于白采）")

say("")
say("横幅比 = 手横向跨度 / 纵向跨度（同一手势应稳定；双峰 = 混了两种特征空间）")
for c in classes:
    m = X[y == c].reshape(-1, 21, 2)
    ratio = np.ptp(m[:, :, 0], axis=1) / (np.ptp(m[:, :, 1], axis=1) + 1e-6)
    med = float(np.median(ratio))
    far = int((np.abs(ratio - med) > 0.35 * med).sum())
    say(f"  {c:<8} 中位 {med:5.2f}   偏离>35% 的 {far} 条")

# ---- 类别可分性：5 折 kNN ----
rng = np.random.default_rng(0)
order = rng.permutation(len(X))
k = 5
correct = 0
conf = collections.Counter()
for fold in range(5):
    test = order[fold::5]
    train = np.setdiff1d(order, test)
    Xtr, ytr = X[train], y[train]
    d = np.linalg.norm(X[test][:, None, :] - Xtr[None, :, :], axis=2)
    nn = np.argsort(d, axis=1)[:, :k]
    for i, t in enumerate(test):
        votes = collections.Counter(ytr[nn[i]])
        pred = votes.most_common(1)[0][0]
        if pred == y[t]:
            correct += 1
        else:
            conf[(y[t], pred)] += 1

acc = correct / len(X)
say("")
say(f"5 折 kNN(k={k}) 准确率: {acc:.1%}   ← 数据本身可分性，不是最终模型指标")
if conf:
    say("最容易混的类别对:")
    for (a, b), n in conf.most_common(5):
        say(f"  {a} → 误判 {b}: {n} 次")

say("")
say("类间中心距（越大越好，<0.5 说明两类形状太像）:")
centroids = {c: X[y == c].mean(axis=0) for c in classes}
for i, a in enumerate(classes):
    for b in classes[i + 1:]:
        say(f"  {a} ↔ {b}: {np.linalg.norm(centroids[a] - centroids[b]):.2f}")

say("")
say("各维均值（前 6 维，检查归一化有没有把数据压扁）:")
say("  " + " ".join(f"{v:+.3f}" for v in X.mean(axis=0)[:6]))

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "data_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print(f"\n报告已存: {os.path.join(OUT_DIR, 'data_report.txt')}")
