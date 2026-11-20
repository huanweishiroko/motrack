# 实时姿态交互训练系统

## 项目简介
本项目是一个基于前后端分离架构的实时姿态交互训练系统。用户可以通过浏览器上传图像（或通过摄像头采集帧），系统在后台调用 AI 模型（MediaPipe）进行人体姿态识别与关键点提取，并实时返回姿态反馈与判定结果。

## 技术栈
*   **前端**：原生 HTML5 + JavaScript (后续计划升级为 Vue3 + Element Plus)
*   **后端**：Python 3.12.8, FastAPI, Uvicorn
*   **数据校验**：Pydantic
*   **AI 引擎**：MediaPipe (规划中), OpenCV, Numpy

## 当前进度 (MVP 阶段)
- [x] 定义前后端 API 接口契约（基于 Pydantic 的输入输出模型）。
- [x] 搭建 FastAPI 基础服务，配置跨域 CORS。
- [x] 编写前端测试页面，实现图像上传与预览。
- [x] 前端通过 Base64 编码传输图像至后端，后端成功接收并返回模拟的 JSON 结果。
- [ ] 后端接入 MediaPipe，实现真实的图像解码与姿态识别。
- [ ] 实现 WebSocket 视频流实时帧传输。

## 目录结构
```text
├── main.py          # FastAPI 后端主程序
├── index.html       # 前端测试页面 (原生 HTML/JS)
├── README.md        # 项目说明文档
└── .gitignore       # Git 忽略文件配置