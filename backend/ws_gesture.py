# -*- coding: utf-8 -*-
"""WebSocket 版手势接口。

和 HTTP 路径的关系：**完全并行**。main.py 里的 /api/Gestureinput 一行没动，
这个文件坏了直接删掉、把 main.py 末尾那两行 include_router 注掉就回退了。

为什么要单独一个文件、还要重写一遍判定流程：
  main.py 的 detect_gesture 用的是进程级 _hist（所有请求共享），
  这里要的是「每条连接一份历史」。共用那份全局状态就做不到隔离，
  所以这里自带一个 VoteState。代价是组装响应的逻辑有两份——
  改动时序逻辑时两边都要看一眼（文件头这句就是提醒）。
"""
from __future__ import annotations

import asyncio
import time
from collections import Counter, deque

import cv2
import numpy
from anyio import to_thread
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import main

router = APIRouter()


# ---------------------------------------------------------------- 指标
class Stats:
    """进程级计数。只在事件循环线程里被改动，所以不用加锁。"""

    def __init__(self, cap: int = 600):
        self.connections = 0          # 当前连接数
        self.peak = 0                 # 峰值连接数
        self.frames_in = 0            # 收到的帧
        self.dropped = 0              # 被新帧覆盖掉的旧帧
        self.served = 0               # 处理完并回出去的帧
        self.errors = 0               # 解码失败
        self.lat = deque(maxlen=cap)  # 最近 N 帧的服务端处理耗时（ms）
        self.times = deque(maxlen=cap)
        self.started = time.time()

    def snapshot(self):
        lat = sorted(self.lat)
        p50 = lat[len(lat) // 2] if lat else 0.0
        p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else 0.0
        now = time.time()
        recent = sum(1 for t in self.times if now - t <= 2.0)
        return {
            "connections": self.connections,
            "connections_peak": self.peak,
            "frames_in": self.frames_in,
            "frames_served": self.served,
            "frames_dropped": self.dropped,
            "errors": self.errors,
            "latency_ms": {"p50": round(p50, 1), "p95": round(p95, 1),
                           "max": round(lat[-1], 1) if lat else 0.0},
            "fps": round(recent / 2.0, 1),
            "uptime_s": round(now - self.started, 1),
        }


_stats = Stats()


@router.get("/api/stats")
def ws_stats():
    """WS 通道的运行指标：连接数、吞吐、丢帧、p50/p95 延迟、FPS。"""
    return _stats.snapshot()



# ---------------------------------------------------------------- 投票状态
class VoteState:
    """每连接一份的时序投票（对应 main.py 里的 _hist/_last + _vote）。"""

    def __init__(self, frames: int | None = None, vote_min: int | None = None):
        self.frames = frames or main.VOTE_FRAMES
        self.vote_min = vote_min or main.VOTE_MIN
        self.hist: dict[str, deque] = {}
        self.last: dict[str, dict] = {}

    def clear(self):
        self.hist.clear()
        self.last.clear()

    def vote(self, side, res):
        if res is None:
            self.hist.pop(side, None)
            self.last.pop(side, None)
            return None
        d = self.hist.setdefault(side, deque(maxlen=self.frames))
        d.append(res[0])
        self.last.setdefault(side, {})[res[0]] = res[1]
        if len(d) < self.vote_min:
            return res
        label, n = Counter(d).most_common(1)[0]
        if n < self.vote_min:
            return ("?", res[1])
        return (label, self.last[side].get(label, res[1]))


# ---------------------------------------------------------------- 同步处理
def _decode(blob: bytes):
    """二进制帧（JPEG/PNG 字节）→ RGB ndarray；坏数据返回 None。"""
    try:
        arr = numpy.frombuffer(blob, numpy.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR_RGB)
    except Exception:
        return None


# 每连接的 VIDEO 模式检测器上限：一个约 20–40MB，超了就退回共享的 IMAGE 检测器
MAX_VIDEO_DETECTORS = 4
_video_slots = asyncio.Semaphore(MAX_VIDEO_DETECTORS)


class VideoDetector:
    """每连接一个 VIDEO 模式检测器。跨帧追踪，比 IMAGE 快约 1.6 倍。

    两条硬约束：
      1. 时间戳必须严格递增，同一毫秒发两帧会抛异常
      2. 实例有状态，同一实例只能被一条连接串行使用
    """

    def __init__(self, detector):
        self.det = detector
        self._ts = 0

    def _next_ts(self):
        self._ts = max(int(time.time() * 1000), self._ts + 1)
        return self._ts

    def detect(self, frame):
        return self.det.detect_for_video(main.to_mp_image(frame), self._next_ts())

    def close(self):
        try:
            self.det.close()
        except Exception:
            pass


async def _make_video_detector():
    """在线程池里建（建一次 1–2 秒，不能在事件循环里干）。拿不到名额返回 None。"""
    if _video_slots.locked() and _video_slots._value <= 0:
        return None
    await _video_slots.acquire()
    try:
        det = await to_thread.run_sync(
            main.make_detector, main.ensure_model(main.DEFAULT_MODEL), 2, True)
    except Exception as e:               # 建不起来就退回共享检测器，别让连接挂掉
        print(f"[ws] VIDEO 检测器创建失败，退回共享 IMAGE 检测器：{e}")
        _video_slots.release()
        return None
    return VideoDetector(det)


def _classify(hands, labels, w, h, state: VoteState, ts):
    """拿到检测结果之后的分类与组装（IMAGE / VIDEO 两条路共用）。"""
    if not hands:
        return {"type": "result", "status": "success", "is_success": False,
                "gesture_name": "", "confidence": 0.0,
                "feedback": "没有检测到手", "state": "no_person",
                "keypoints": None, "ts": ts}

    keypoints = {}
    scores = []
    counts = []
    for i, landmarks in enumerate(hands):
        cat = labels[i][0] if i < len(labels) and labels[i] else None
        side = getattr(cat, "category_name", "").lower() or f"hand{i}"
        score = getattr(cat, "score", 0.0)
        scores.append(score)
        for idx, name in enumerate(main.HAND_KEYS):
            lm = landmarks[idx]
            keypoints[f"{side}_{name}"] = [lm.x, lm.y, lm.z]
        counts.append(state.vote(side, main.classify_hand(
            landmarks, w, h,
            side != main.TRAIN_SIDE if score >= main.MIN_SIDE_CONF else False)))

    avg = sum(scores) / len(scores) if scores else 0.0
    low = avg < main.LOW_CONF
    side_cn = ("双手" if len(hands) > 1 else
               ("左手" if getattr(labels[0][0], "category_name", "") == "Left" else "右手"))
    picked = [c for c in counts if c is not None]
    gated = len(picked) < len(counts)
    unknown = any(c[0] in ("?", main.OTHER_LABEL) for c in picked)
    bad = gated or low or unknown

    if gated:
        gesture_name = f"{side_cn} 距离过远，未判定"
        feedback = "手离镜头太远，请靠近一点"
        conf = 0.0
    elif unknown:
        gesture_name = f"{side_cn}：{main.OTHER_LABEL}"
        feedback = "未识别的姿势，按「其他」处理"
        conf = min(c[1] for c in picked)
    else:
        names = " + ".join(c[0] for c in picked)
        conf = min(c[1] for c in picked)
        gesture_name = f"{side_cn}：{names}"
        feedback = "识别置信度偏低，调整姿势" if conf < 0.6 else "识别成功"

    bad = bad or conf < 0.6
    return {"type": "result", "status": "success", "is_success": not bad,
            "gesture_name": gesture_name, "confidence": round(conf, 2),
            "feedback": feedback, "state": "low_confidence" if bad else "ready",
            "keypoints": keypoints, "ts": ts}


def _process(frame, state: VoteState, ts=0, video: VideoDetector | None = None):
    """检测 + 分类。跑在线程池里，别在事件循环里直接调。

    video 为 None 时走共享的 IMAGE 检测器（和 HTTP 接口同一个），
    否则用这条连接自己的 VIDEO 检测器。
    """
    detector = video or main.get_detector()
    result = detector.detect(frame)
    h, w = frame.shape[:2]
    return _classify(result.hand_landmarks, result.handedness, w, h, state, ts)


# 检测器不是线程安全的：全局串行。单用户无感，多用户会排队（P2 再优化）
_detector_lock = asyncio.Lock()


@router.websocket("/ws/gesture")
async def ws_gesture(ws: WebSocket):
    await ws.accept()
    state = VoteState()
    slot = {"blob": None}      # 只留最新一帧，旧的被覆盖 = 天然背压
    stop = asyncio.Event()

    async def receiver():
        """专门收帧。处理慢的时候，缓冲区里的旧帧会被新帧覆盖掉。"""
        try:
            while not stop.is_set():
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    stop.set()
                    return
                text = msg.get("text")
                if text is not None:
                    if "ping" in text:
                        await ws.send_json({"type": "pong"})
                    continue
                blob = msg.get("bytes")
                if blob:
                    if slot["blob"] is not None:
                        _stats.dropped += 1     # 上一帧还没处理就被覆盖了
                    _stats.frames_in += 1
                    slot["blob"] = blob
        except (WebSocketDisconnect, RuntimeError):
            stop.set()

    recv_task = asyncio.create_task(receiver())
    _stats.connections += 1
    _stats.peak = max(_stats.peak, _stats.connections)
    # 懒建：连接建立时不建，等第一帧真正到来才建。
    # 否则打开页面不录像的连接也会白占 20–40MB。
    video = None
    video_tried = False
    try:
        while not stop.is_set():
            blob = slot["blob"]
            if blob is None:
                await asyncio.sleep(0.005)
                continue
            slot["blob"] = None
            frame = _decode(blob)
            if frame is None:
                _stats.errors += 1
                await ws.send_json({"type": "error", "code": "decode_failed",
                                    "msg": "无法解码图像"})
                continue
            ts = int(time.time() * 1000)
            if video is None and not video_tried:
                video_tried = True
                video = await _make_video_detector()
            t0 = time.perf_counter()
            if video is not None:
                # 独占的 VIDEO 检测器，不需要和别的连接抢锁
                result = await to_thread.run_sync(_process, frame, state, ts, video)
            else:
                async with _detector_lock:
                    result = await to_thread.run_sync(_process, frame, state, ts, None)
            _stats.served += 1
            _stats.lat.append((time.perf_counter() - t0) * 1000)
            _stats.times.append(time.time())
            await ws.send_json(result)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        stop.set()
        recv_task.cancel()
        state.clear()
        _stats.connections = max(0, _stats.connections - 1)
        if video is not None:
            video.close()
            _video_slots.release()
