# 近景手势采集（21 点特征向量）

摄像头 → MediaPipe 21 点 → 归一化特征向量 → CSV。采完直接训 MLP/SVM，不需要图像数据集。

## 1. 环境（已合并成一个）

整个仓库共用 `E:\motrack\.venv`（后端 + 采集 + 训练同一套依赖）。本目录不再有独立 venv。

**本目录所有命令里的 `.\.venv\Scripts\python` 都读作 `..\.venv\Scripts\python`**，例如：

```powershell
cd E:\motrack\gesture
..\.venv\Scripts\python collect.py --preset single
```

重建环境（换机器时）：

```powershell
cd E:\motrack
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

`requirements.txt` 在仓库根，`gesture\requirements.lock.txt` 是当时冻结的完整版本清单。
需要 Python 3.9–3.12。

## 2. 先自检（不用摄像头）

```powershell
.\.venv\Scripts\python gesture_features.py
```

正常输出：

```
特征维度 : 42
平移+缩放后 L2 差 : 1e-06
旋转25° 无对齐 L2 差 : 1.02
旋转25° 已对齐 L2 差 : 0.0
自检通过 [OK]
```

## 3. 采集

```powershell
.\.venv\Scripts\python collect.py --preset single --target 250   # 单指 4 类
.\.venv\Scripts\python collect.py --preset combo  --target 250   # 组合 10 类
.\.venv\Scripts\python collect.py --preset all    --target 250   # 14 类
```

窗口里用键盘：

| 键 | 作用 |
|---|---|
| `0`–`9` / `a` `d` `f` `g` | 选中该类并开始自动采集（再按同一个键 = 暂停/继续） |
| `空格` | 暂停 / 继续 |
| `s` | 补录一帧（和上一条太近会提示「太近了没记」，不写入） |
| `u` | 撤销上一条 |
| `q` / `ESC` | 保存并退出 |

默认 0.12 秒记一帧，`--min-move 0.05` 过滤手没动的重复帧，每类 250 条大概 1–2 分钟。
中途退出不丢数据；再开会自动续采已有 CSV。

### 类表（`--preset`）

| 键 | 单指类 | 键 | 组合类 |
|---|---|---|---|
| `0` | 食指 | `4` | 点赞 |
| `1` | 中指 | `5` | OK |
| `2` | 无名指 | `6` | 张开（五指） |
| `3` | 小指 | `7` | 比耶 |
| | | `8` | 握拳 |
| | | `9` | 六（call） |
| | | `a` | 点踩（反向点赞） |
| | | `d` | 后三指（中·无名·小） |
| | | `f` | 中三指（食·中·无名） |
| | | `g` | 比心（单手） |

`--preset all` 的顺序是 4 个单指在前、10 个组合在后，所以 `0`–`3` 单指、`4`–`9`+`a`/`d`/`f`/`g` 组合。

## 4. 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--labels` | ok,peace,fist,point | 逗号分隔，最多 10 类 |
| `--target` | 300 | 每类目标条数，够了自动停 |
| `--interval` | 0.12 | 自动采集间隔（秒） |
| `--min-move` | 0.05 | 特征差小于它就算重复，丢弃 |
| `--min-hand-px` | 80 | 掌长低于这个像素数就跳过（防尺度爆炸） |
| `--align` | 关 | 手会歪着摆就开，把掌向统一到朝上 |
| `--use-z` | 关 | 存 z（63 维）。z 抖动大，一般不开 |
| `--cam` | 0 | 摄像头编号，打不开就试 1、2 |
| `--no-flip` | 关 | 默认镜像画面（像照镜子） |

## 5. 目录与输出

```
gesture/
  collect.py  gesture_features.py  train.py  predict.py  check_data.py
  README.md  requirements.txt  requirements.lock.txt
  hand_landmarker.task          MediaPipe 模型（首次运行自动下载）
  data/                         ← 原始资产，进版本库
    gesture_data.csv            首列 label，后 42 列 x0,y0,...,x20,y20
    gesture_data.csv.meta.json  特征版本标记，check_data.py 靠它认 v1/v2
    gesture_data_trace.csv      追加式审计日志
    collect_meta.json           采集参数（间隔/门控/摄像头）
    gesture_clf.joblib          训练产物，后端推理用
  out/                          ← 派生，可重生成，已被 .gitignore
    train.csv  test.csv  gesture_data_slim.csv
    train_report.txt  data_report.txt
```

- `gesture_trace.csv`：每写入/撤销/拒绝一行
  `time,event,label,dist,note`；`event` 取值 `auto` / `manual` / `manual_reject` / `undo`，
  `dist` 是相对上一条的距离（`inf` 表示该帧前面没有基准，比如切类后的第一帧）

> 事后有人质疑「数据里怎么有低于 min-move 的相邻行」时，直接查 `gesture_trace.csv`：
> 每条写入都带 `dist`，低于阈值的根本不会出现（`u` 撤销后基准会退回上一条，
> `s` 补录也走同一套过滤）。

## 6. 特征怎么算的

`gesture_features.extract()`（v2，2026-11 起）：

0. 给了 `width/height` 就先乘回**像素坐标** —— 归一化坐标是「x 除以宽、y 除以高」，4:3 画面会把手的纵向拉伸 33%，换 16:9 摄像头分布就漂
1. 除以「腕(0) → 中指根(9)」掌长 → 远近无关
2. 减去手掌五点 (0,5,9,13,17) 质心 → 画面位置无关
3. `--align` 时旋转到中指朝上 → 倾斜无关
4. `--min-hand-px 80`：掌长不足 80px 的点直接跳过（手太远/检测飘，会把特征尺度炸开）

**v1 数据已归档**（`_archive\`），现在的 `gesture_data.csv` 就是 v2。
两个空间只差 `D=diag(w,h)`，`feat_px = D·feat / ‖D·(feat9−feat0)‖`，尺度因子约掉。
**v1 和 v2 的 CSV 不能混**，`check_data.py` 会读 `.meta.json` 认版本，认不出就报警。

## 7. 训练

```powershell
.\.venv\Scripts\python check_data.py          # 先体检
.\.venv\Scripts\python train.py               # 直接用这个（旋转归一是默认）
```

- `--align` **已经是默认行为**，不用加；要关才写 `--no-align`（只用于对照实验）。
  忘加参数会让准确率从 94.7% 掉到 86%，而且模型包里 `align=False` 会让推理端老实照做
- `--k 1` 不抽稀。42 维特征配 100 来个样本学不动，**现在冗余不是问题，样本不够才是**
- `--clip 3.0` 丢掉 |x|>3 的尺度爆炸样本
- `--trim 0.1` 每类首尾各丢 10%，去掉换手势时的过渡帧（默认 0）

切分是按行号（时间）**前 60% 训练 / 中间 10% 丢弃 / 后 30% 测试**，不打乱。
中间那 10% 是隔离带：没有它，训练集最后一条和测试集第一条只差 0.12 秒，等于开卷。

训完会落盘一个 `gesture_clf.joblib`，里面除了模型还带推理端必须知道的东西：

```python
{
  "model": <已 fit 的 StandardScaler + SVC pipeline>,
  "classes": [...],
  "align": True,            # 推理时 extract(..., align=True)
  "min_hand_px": 80.0,      # 推理时的掌长门控
  "feature_version": 2,
  "dim": 42, "use_z": False,
  "test_acc": 0.9467, "n_train": 600, "trained_at": "...", "sklearn": "1.9.1"
}
```

`SVC(probability=True)` 才会算 `predict_proba`（Platt 标定，内部 5 折，训练慢几倍但样本小无所谓）。
代价：`predict_proba` 的 argmax 和 `predict()` 可能有一两例不一致，94.7% → 回放 94.3% 就是这个原因，正常。

## 8. 推理

```powershell
.\.venv\Scripts\python predict.py            # 摄像头实时
.\.venv\Scripts\python predict.py --check    # 不开摄像头，用 test.csv 回放验收
```

参数全部从模型包里读，不靠记性：`align`、`min_hand_px`、`dim`、类别顺序。
`--check` 回放低于 80% 会返回非零退出码，能直接塞进 CI。

## 8. 坑

- mediapipe 有新旧两套 API：`collect.py` 已自动检测，0.10.2x 走 `mp.solutions`，0.10.3x+ 走 `tasks`（本机 0.10.35 走 tasks）
- 摄像头被别的程序占用会读帧失败，关掉再跑
- 背景别太乱、手别出框（出框时关键点是预测的，会污染数据）
- 每类摆 3 个朝向 × 2 个距离 × 2 个手位，比一个姿势摆 300 次值钱
- 采完先画个类别分布图看平不平衡，别一类 300 另一类 40
