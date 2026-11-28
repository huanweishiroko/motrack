# -*- coding: utf-8 -*-
"""消融实验：把「为什么这么选」变成数字。

四组，都跑同一份数据、同一个时间切分，只改一个变量：
  E1 分类器       SVC-RBF / kNN / 逻辑回归 / 线性 SVC
  E2 镜像增广     无 / 有
  E3 特征子集     42 维(21点) / 32 维(去拇指) / 12 维(仅指尖)
  E4 拒绝阈值     CONF_MIN × MARGIN_MIN 的覆盖率-精度权衡

用法：
    python ablation.py            # 跑全部，输出表到 out/ablation.md
"""
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

import train as T

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "gesture_data.csv")
OUTDIR = os.path.join(HERE, "out")
LINES = []


def say(s=""):
    print(s)
    LINES.append(str(s))


def make_models():
    return [
        ("SVC-RBF", make_pipeline(StandardScaler(),
                                  SVC(C=10, gamma="scale", probability=True))),
        ("kNN-5", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))),
        ("逻辑回归", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
        ("线性SVC", make_pipeline(StandardScaler(), LinearSVC(C=1.0))),
    ]


def subset(X, which):
    """按点索引选特征。X 是 42 维 = 21 点 × (x,y)。"""
    if which == "全量(21点)":
        return X
    if which == "去拇指(16点)":
        idx = [i for p in range(5, 21) for i in (2 * p, 2 * p + 1)]
    elif which == "仅指尖(6点)":
        idx = [i for p in (0, 4, 8, 12, 16, 20) for i in (2 * p, 2 * p + 1)]
    else:
        raise ValueError(which)
    return X[:, idx]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    X, y, _ = T.load(DATA)
    classes = sorted(set(y.tolist()))
    X = T.align_rows(X)                       # 与正式训练保持一致：旋转归一化
    tr, te = T.time_split(X, y, classes)
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    say(f"数据 {len(X)} 条 / {len(classes)} 类，训练 {len(Xtr)}，测试 {len(Xte)}")
    say("切分：每类前 60% 训练 / 中间 10% 丢弃 / 后 30% 测试（时间序，不打乱）")
    say()

    # ---------- E1 分类器 ----------
    say("## E1 · 分类器对比（同一份特征，只换模型）")
    say()
    say("| 模型 | 准确率 |")
    say("|---|---|")
    for name, model in make_models():
        model.fit(Xtr, ytr)
        acc = accuracy_score(yte, model.predict(Xte))
        say(f"| {name} | {acc:.1%} |")
    say()

    # ---------- E2 镜像增广 ----------
    say("## E2 · 训练集镜像增广（SVC-RBF）")
    say()
    say("| 配置 | 原样测试 | 镜像后测试 |")
    say("|---|---|---|")
    Xte_m = T.mirror_features(Xte)
    for tag, aug in (("不加增广", False), ("加增广", True)):
        xa, ya = (np.vstack([Xtr, T.mirror_features(Xtr)]),
                  np.concatenate([ytr, ytr])) if aug else (Xtr, ytr)
        m = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True))
        m.fit(xa, ya)
        a1 = accuracy_score(yte, m.predict(Xte))
        a2 = accuracy_score(yte, m.predict(Xte_m))
        say(f"| {tag} | {a1:.1%} | {a2:.1%} |")
    say()

    # ---------- E3 特征子集 ----------
    say("## E3 · 特征子集（SVC-RBF，加镜像增广）")
    say()
    say("| 特征 | 维度 | 准确率 |")
    say("|---|---|---|")
    for which in ("全量(21点)", "去拇指(16点)", "仅指尖(6点)"):
        A = subset(Xtr, which)
        B = subset(Xte, which)
        xa = np.vstack([A, T.mirror_features(A)])
        ya = np.concatenate([ytr, ytr])
        m = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True))
        m.fit(xa, ya)
        acc = accuracy_score(yte, m.predict(B))
        say(f"| {which} | {A.shape[1]} | {acc:.1%} |")
    say()

    # ---------- E4 拒绝阈值 ----------
    say("## E4 · 拒绝阈值（14 类模型，覆盖 vs 精度）")
    say()
    say("覆盖率 = 被判定的比例；精度 = 被判定的样本里正确的比例")
    say()
    say("| CONF_MIN | MARGIN_MIN | 覆盖率 | 精度 |")
    say("|---|---|---|---|")
    xa = np.vstack([Xtr, T.mirror_features(Xtr)])
    ya = np.concatenate([ytr, ytr])
    m = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True))
    m.fit(xa, ya)
    proba = m.predict_proba(Xte)
    order = np.argsort(proba, axis=1)
    top1 = proba[np.arange(len(proba)), order[:, -1]]
    top2 = proba[np.arange(len(proba)), order[:, -2]]
    pred = m.classes_[order[:, -1]]
    for conf in (0.5, 0.6, 0.7):
        for margin in (0.10, 0.15, 0.20):
            keep = (top1 >= conf) & ((top1 - top2) >= margin)
            cov = keep.mean()
            acc = (pred[keep] == yte[keep]).mean() if keep.any() else float("nan")
            say(f"| {conf:.2f} | {margin:.2f} | {cov:.1%} | {acc:.1%} |")
    say()

    path = os.path.join(OUTDIR, "ablation.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\n已写出: {path}")


if __name__ == "__main__":
    sys.exit(main())
