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
# 单指 4 类
.\.venv\Scripts\python collect.py --preset single --facing palm --target 250

# 组合 9 类 + 其他 1 类，手心手背分两轮采
.\.venv\Scripts\python collect.py --preset combo --facing palm  --target 150
.\.venv\Scripts\python collect.py --preset combo --facing back  --add 150
.\.venv\Scripts\python collect.py --preset other --facing palm  --target 150
```

`--target` 是**每类总数**，改朝向采第二批时要加 `--add N`（在现有条数上每类再加 N 条），
否则程序会认为所有类都采满了、一帧都不记。

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

| 键 | 单指类 | 键 | 组合类 / 其他 |
|---|---|---|---|
| `0` | 食指 | `4` | 点赞 |
| `1` | 中指 | `5` | OK |
| `2` | 无名指 | `6` | 张开（五指） |
| `3` | 小指 | `7` | 比耶 |
| | | `8` | 握拳 |
| | | `9` | 六（call） |
| | | `a` | 点踩（反向点赞） |
| | | `f` | 中三指（食·中·无名） |
| | | `g` | 比心（单手） |
| | | `h` | 其他 |

**「其他」是分类器的出口**：不属于任何目标手势的姿势都丢这里。没有这一类，
模型对任何输入都必须硬选一个目标类（比个大拇指会被判成中指）。
采的时候别只采一种姿势，要覆盖"接近真类但不对"的变体。

> 原「后三指」已删除：它与 OK 的类间中心距只有 0.93（全场最近），手背朝镜头时
> 拇指食指的"环"被遮住，两类根本分不开。删掉后 OK 的召回从 83% 升到 100%。

## 4. 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--labels` | 无 | 逗号分隔的类别名，`--preset` 优先；最多 10 个 |
| `--target` | 300 | 每类目标**总**条数，够了自动停 |
| `--add` | 0 | 在现有条数上每类**再加** N 条（换朝向采第二批用这个） |
| `--side` | left | 本轮用哪只手；`right` 会把 x 取反，让特征和训练集同分布 |
| `--facing` | any | 本轮手心/手背朝镜头，只记录不参与特征；查朝向平衡用 |
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
5. `mirror=True`：把 x 取反，右手 → 镜像成左手的样子

**镜像归一化**：训练集只有一只手时，另一只手做同样的动作在画面里是镜像关系。
采集时用 `--side right`（或 `--side left`）声明本轮用哪只手，右侧会存成镜像后的特征；
模型包里记 `train_side`，推理端发现是另一只手就自动镜像。

**左右手门控**：`decide_mirror(side, score, train_side, min_score=0.80)`。
分数低于 0.80 时**不镜像、但也照常分类**——模型做过镜像增广，对"翻不翻"不敏感，
没必要为了判不清左右手而丢掉这一帧。

> **朝向陷阱**：采集时如果每类的"手心/手背"比例差别很大，模型会把"拇指在哪一侧"
> 当成类别线索。`check_data.py` 的「朝向平衡」段就是查这个：单侧 >70% 会标 ⚠。
> 训练时加 `--mirror-aug` 可以把每类的两侧自动拉平，代价是 kNN 略降。

**v1 数据已归档**（`_archive\`），现在的 `gesture_data.csv` 就是 v2。
两个空间只差 `D=diag(w,h)`，`feat_px = D·feat / ‖D·(feat9−feat0)‖`，尺度因子约掉。
**v1 和 v2 的 CSV 不能混**，`check_data.py` 会读 `.meta.json` 认版本，认不出就报警。

## 7. 训练

```powershell
.\.venv\Scripts\python check_data.py          # 先体检
.\.venv\Scripts\python train.py --mirror-aug  # 数据里有单朝向的类时必须加
```

当前数据集（14 类 / 5091 条，时间切分 60/10/30）：

| 指标 | 值 |
|---|---|
| 总体 | 88.5% |
| 不含「其他」 | **96.9%** |
| 标签打乱对照 | 15.0%（14 类随机 7.1%） |

- `--align` **已经是默认行为**，不用加；要关才写 `--no-align`（只用于对照实验）。
  忘加参数会让准确率掉一大截，而且模型包里 `align=False` 会让推理端老实照做
- `--mirror-aug`：训练集做镜像增广（x 取反），测试集不动。**测试集翻转等于换答案**
- `--k 1` 不抽稀。42 维特征配 100 来个样本学不动，**现在冗余不是问题，样本不够才是**
- `--clip 3.0` 丢掉 |x|>3 的尺度爆炸样本
- `--trim 0.1` 每类首尾各丢 10%，去掉换手势时的过渡帧（默认 0）

切分是按行号（时间）**前 60% 训练 / 中间 10% 丢弃 / 后 30% 测试**，不打乱。
中间那 10% 是隔离带：没有它，训练集最后一条和测试集第一条只差 0.12 秒，等于开卷。

训完会落盘一个 `gesture_clf.joblib`，里面除了模型还带推理端必须知道的东西：

```python
{
  "model": <已 fit 的 StandardScaler + SVC pipeline>,
  "classes": [...],         # 14 类，顺序即 predict_proba 的下标
  "align": True,            # 推理时 extract(..., align=True)
  "min_hand_px": 80.0,      # 推理时的掌长门控
  "train_side": "left",     # 训练集是哪只手，另一只手推理时镜像
  "mirror_aug": True,       # 训练时是否做了镜像增广
  "feature_version": 2,
  "dim": 42, "use_z": False,
  "test_acc": 0.8847, "n_train": 3055, "trained_at": "...", "sklearn": "1.9.1"
}
```

`SVC(probability=True)` 才会算 `predict_proba`（Platt 标定，内部 5 折，训练慢几倍但样本小无所谓）。
代价：`predict_proba` 的 argmax 和 `predict()` 可能有一两例不一致，属正常现象。

## 7.5 消融实验

```powershell
.\.venv\Scripts\python ablation.py      # 结果写到 out\ablation.md
```

四组，同一份数据、同一个时间切分，每次只改一个变量。

**E1 · 分类器**（同特征，只换模型）

| 模型 | 准确率 |
|---|---|
| SVC-RBF | **87.2%** |
| 线性 SVC | 84.5% |
| kNN-5 | 82.9% |
| 逻辑回归 | 80.8% |

> 差距只有 6 个点，说明**特征比模型重要**。42 维手工特征已经把类分得比较开了，
> 换更复杂的模型边际收益很小。

**E2 · 训练集镜像增广**

| 配置 | 原样测试 | 镜像后测试 |
|---|---|---|
| 不加增广 | 87.2% | **44.3%** |
| 加增广 | 88.5% | **88.5%** |

> 这组是全项目最关键的证据：不加增广时模型严重依赖「手朝哪边」，一旦画面镜像，准确率腰斩。
> 加了增广后原样与镜像**完全一致**——说明模型真的在学形状，不是学朝向。

**E3 · 特征子集**（加镜像增广）

| 特征 | 维度 | 准确率 |
|---|---|---|
| 全量（21 点） | 42 | **88.5%** |
| 仅指尖（腕 + 5 指尖） | 12 | 86.7% |
| 去拇指（16 点） | 32 | 82.1% |

> 12 维只掉 1.8 个点，说明这个任务**信息主要在指尖**；去掉拇指掉 6.4 个点，
> 说明拇指是区分「点赞 / OK / 点踩 / 六」的关键。

**E4 · 拒绝阈值**

| CONF_MIN | MARGIN_MIN | 覆盖率 | 精度 |
|---|---|---|---|
| 0.50 | 0.15 | 93.4% | 92.8% |
| **0.60** | **0.15** | **91.7%** | **93.9%** |
| 0.70 | 0.15 | 89.1% | 95.8% |

> 阈值从 0.5 抬到 0.7，精度 92.8% → 95.8%，代价是覆盖率掉 4 个点。
> 当前线上用 0.60/0.15：牺牲 8% 的帧，换 5 个点的精度。
> 注意 `probability=True` 内部带交叉验证，**同一组配置重跑会有 ±0.5% 的抖动**，
> 不要拿小数点后一位做结论。

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
