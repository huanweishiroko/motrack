# 实时手势交互训练系统

## 项目简介

基于前后端分离架构的实时手势交互训练系统。

用户在浏览器上传图像或开启摄像头，前端按固定频率抽帧并以 Base64 上传；后端调用 MediaPipe HandLandmarker 提取 21 个手部关键点，换算成 42 维特征后交给 SVC 分类器判定手势，并对低置信度、分布外、手部过小的帧返回引导而非判定结果。

## 技术栈

*   **前端**：原生 HTML5 + JavaScript（后续计划升级为 Vue3 + Element Plus）
*   **后端**：Python 3.12.8, FastAPI, Uvicorn
*   **数据校验**：Pydantic
*   **AI 引擎**：MediaPipe 1.2.3（Tasks API · HandLandmarker）, OpenCV 5.1, NumPy 2.5
*   **分类器**：scikit-learn 1.9（SVC · RBF 核 · probability=True）

> 全仓库共用一个 `.venv`（`E:\motrack\.venv`），后端与 `gesture/` 工具链依赖同一套环境。

> mediapipe / numpy / opencv 三者版本互相耦合，改动前请参考 `requirements.txt` 中的注释。

## 当前进度

- [x] 定义前后端 API 接口契约（基于 Pydantic 的输入输出模型）
- [x] 搭建 FastAPI 基础服务，配置跨域 CORS
- [x] 编写前端测试页面，实现图像上传与预览
- [x] 前端通过 Base64 编码传输图像至后端
- [x] 后端接入 MediaPipe HandLandmarker，返回 21 点真实关键点
- [x] 手势判定：关节角度法，计算伸出手指数量
- [x] 手部尺寸门控：尺寸不足时返回引导而非判定
- [x] 摄像头采集（getUserMedia + canvas 抽帧）
- [x] 21 点骨架连线绘制（canvas 固定拓扑）
- [x] 判定改为「21 点特征 + SVC 分类器」，4 类时间切分实测 94.7%
- [x] 置信度 + margin 拒绝、5 帧多数票平滑
- [ ] WebSocket 视频流实时帧传输
- [ ] 业务闭环（会话管理、判定记录、历史趋势）

## 手势判定

> **2026-11-23 更新**：判定已改为「21 点特征 + SVC 分类器」，下文的关节角度规则是历史方案，保留作对照。
> 特征定义、采集与训练流程见 [`gesture/README.md`](gesture/README.md)。

**当前流程**：21 点 → `gesture_features.extract()` 得 42 维（像素空间归一化 + 掌长门控，可选旋转对齐）→ SVC `predict_proba` → 低于 `CONF_MIN=0.60` 或 top1−top2 < `MARGIN_MIN=0.15` 判为「不在训练集内」→ 5 帧多数票平滑。

**历史方案（角度法）**：

**关键点**：MediaPipe HandLandmarker 输出 21 个手部关键点，索引顺序固定（0 手腕，1–4 拇指，5–8 食指，9–12 中指，13–16 无名指，17–20 小指）。

**判据**：以 PIP 关节为顶点，计算 `MCP → PIP → TIP` 三点夹角。

| 部位 | 顶点 | 伸直判定 |
|---|---|---|
| 四指 | PIP | θ ≥ 165° |
| 拇指 | IP | θ ≥ 155° |

**坐标系**：所有角度在像素坐标下计算（`x·宽`、`y·高`）。MediaPipe 输出为归一化坐标，x/y 按不同比例缩放（1920×1080 下相差 1.78 倍），直接使用会使角度产生系统性偏差。

**质量门控**：

手部尺寸比例 = (手腕 ↔ 中指 MCP 的像素距离) / 画面短边

比例低于 0.18 时不做手势判定，返回 `state = "low_confidence"` 与靠近提示。依据：手部过小时，蜷曲手指的 `PIP → TIP` 向量极短，关键点定位误差会大幅偏转其方向，导致判定失效（实测同姿势由 30.5° 漂移至 140.7°）。

## 已知限制

*   手指指向镜头或手掌侧对镜头时，两点投影缩短，判据可能失效
*   拇指判据余量较薄（约 10°），需补充更多负样本
*   门控阈值 0.18 基于 4 张样本估计，待接入摄像头后用真实分布重新标定
*   当前为单图 HTTP 模式，无时序信息，`cooldown` / `success` 等状态尚未启用
*   多手场景未充分验证（左右手标签、双手排序）

**分类器阶段新增**：

*   只训练了 4 类单指手势（食指/中指/无名指/小指），其它手势一律返回「姿势不在训练集内」
*   数据来自单人单场录制，跨会话、跨光照、跨人未验证
*   拒绝阈值 `CONF_MIN=0.60` / `MARGIN_MIN=0.15` 在 1000 条样本上估的，扩类后需重标
*   投票状态 `_hist` 是进程内全局变量，多标签页/多用户会互相干扰，上线前需按 session 隔离

## 目录结构

```text
motrack/
├── app/
│   └── index.html              # 前端页面（原生 HTML/JS，摄像头抽帧 + 骨架绘制）
├── backend/
│   └── main.py                 # FastAPI 后端主程序
├── gesture/                    # 21 点特征：采集 → 训练 → 推理
│   ├── collect.py              # 摄像头采集，出 CSV
│   ├── gesture_features.py     # 21 点 → 42 维特征（v2）
│   ├── train.py                # 训练并落盘 gesture_clf.joblib
│   ├── predict.py              # 实时推理 / --check 回放验收
│   ├── check_data.py           # 数据体检
│   ├── data/                   # 原始数据与模型（进版本库）
│   └── out/                    # 派生产物（已忽略，可重生成）
├── .venv/                      # 全仓库共用虚拟环境（已忽略）
├── test_hands_tasks.py         # MediaPipe 冒烟 / 压测脚本
├── hand_landmarker.task        # 模型文件 7.8MB（已忽略，首次运行自动下载）
├── test*.jpg                   # 手势测试样本
├── 项目日志.md                  # 开发决策与踩坑记录
├── 规划.txt                    # 8-10 周开发规划
├── requirements.txt
└── README.md
```

## 快速开始

```powershell
cd E:\motrack
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt

cd backend
..\.venv\Scripts\python -m uvicorn main:app --reload --port 8000
```

首次运行会自动下载 `hand_landmarker.task`（约 7.8 MB）。启动后访问 `/docs` 可直接调用接口。

> 必须在 `backend/` 目录下启动，否则要写 `backend.main:app`；`--app-dir backend` 也可以。从仓库根启动 `.venv` 会让 watchfiles 监控整个虚拟环境，启动明显变慢。