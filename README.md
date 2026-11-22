# 实时手势交互训练系统

## 项目简介

基于前后端分离架构的实时手势交互训练系统。

用户在浏览器上传图像（后续支持摄像头实时采集），后端调用 MediaPipe HandLandmarker 提取 21 个手部关键点，通过关节角度判定伸出手指数量，并对质量不足的帧返回引导而非判定结果。

## 技术栈

*   **前端**：原生 HTML5 + JavaScript（后续计划升级为 Vue3 + Element Plus）
*   **后端**：Python 3.12.8, FastAPI, Uvicorn
*   **数据校验**：Pydantic
*   **AI 引擎**：MediaPipe 1.2.3（Tasks API · HandLandmarker）, OpenCV 5.1, NumPy 2.5

> mediapipe / numpy / opencv 三者版本互相耦合，改动前请参考 `requirements.txt` 中的注释。

## 当前进度

- [x] 定义前后端 API 接口契约（基于 Pydantic 的输入输出模型）
- [x] 搭建 FastAPI 基础服务，配置跨域 CORS
- [x] 编写前端测试页面，实现图像上传与预览
- [x] 前端通过 Base64 编码传输图像至后端
- [x] 后端接入 MediaPipe HandLandmarker，返回 21 点真实关键点
- [x] 手势判定：关节角度法，计算伸出手指数量
- [x] 手部尺寸门控：尺寸不足时返回引导而非判定
- [ ] 摄像头采集（getUserMedia + canvas 抽帧）
- [ ] WebSocket 视频流实时帧传输
- [ ] 业务闭环（会话管理、判定记录、历史趋势）

## 手势判定

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

## 目录结构

```text
motrack/
├── app/
│   └── index.html              # 前端测试页面（原生 HTML/JS）
├── backend/
│   ├── main.py                 # FastAPI 后端主程序
│   └── .venv/                  # 虚拟环境（已忽略）
├── test_hands_tasks.py         # MediaPipe 冒烟 / 压测脚本
├── hand_landmarker.task        # 模型文件 7.8MB（已忽略，首次运行自动下载）
├── test*.jpg                   # 手势测试样本
├── 项目日志.md                  # 开发决策与踩坑记录
├── 规划.txt                    # 8-10 周开发规划
├── requirements.txt
└── README.md
```

## 快速开始

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

首次运行会自动下载 `hand_landmarker.task`（约 7.8 MB）。启动后访问 `/docs` 可直接调用接口。