import base64
import logging
import os
import sys
import urllib.request
from collections import Counter, deque

import cv2
import joblib
import mediapipe
import numpy
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pydantic import BaseModel

# uvicorn 的访问日志每帧两行（OPTIONS + POST），5 FPS 下会把有用信息刷掉。
# 需要排查请求时设 MOTRACK_ACCESS_LOG=1。
if os.environ.get("MOTRACK_ACCESS_LOG") != "1":
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "gesture"))

from gesture_features import extract

app = FastAPI(title="手势交互系统")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    # 浏览器对 application/json 的 POST 会先发 OPTIONS 预检；不设这个就默认 5 秒一次，
    # 5 FPS 下等于每帧多一个请求。设成 1 天，预检只剩每天一次。
    max_age=86400,
)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL = "hand_landmarker.task"
HAND_KEYS=[
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip"
]
LOW_CONF=0.5

CONF_MIN = 0.60   # 最高概率低于它就判为不认识
MARGIN_MIN = 0.15 # top1 与 top2 差距太小也算不认识
VOTE_FRAMES = 5   # 时序投票窗口
VOTE_MIN = 3      # 窗口内至少几票才认

class Gestureinput(BaseModel):
    img_64:str

class Gestureoutput(BaseModel):
    status: str                 # success / error
    is_success: bool
    gesture_name: str
    confidence: float           # 置信度
    feedback: str
    state: str
    keypoints: dict[str, list[float]] | None = None

def decode_base64_image(img_str):
    if "," in img_str:
        img_str=img_str.split(",")[1]
    try:
        img_bytes=base64.b64decode(img_str)
        numpy_arr=numpy.frombuffer(img_bytes,numpy.uint8)
        frame=cv2.imdecode(numpy_arr,cv2.IMREAD_COLOR_RGB)
    except Exception:
        # 空串 / 半截数据 / 非图片：cv2 会直接抛 cv2.error，不该让整个请求 500
        return None
    return frame

def ensure_model(path: str) -> str:
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path

    print(f"[model] 本地没有模型，下载中 → {path}")
    urllib.request.urlretrieve(MODEL_URL, path)
    print(f"[model] 完成 {os.path.getsize(path) / 1e6:.1f} MB")
    return path

def make_detector(model_path: str, num_hands: int, video_mode: bool):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=(
            vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE
        ),
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = make_detector(ensure_model(DEFAULT_MODEL), num_hands=2, video_mode=False)
    return _detector

CLF_PATH = os.path.join(_HERE, "..", "gesture", "data", "gesture_clf.joblib")
_B = joblib.load(CLF_PATH)
clf = _B["model"]
CLF_CLASSES = _B["classes"]
CLF_ALIGN = _B["align"]
MIN_HAND_PX = _B["min_hand_px"]
TRAIN_SIDE = _B.get("train_side", "left")   # 训练集是哪只手；另一只手推理时镜像
MIN_SIDE_CONF = 0.80                        # 左右手判定低于此值 → 不镜像（翻错也无所谓）
OTHER_LABEL = "其他"                         # 四不像统一兜到这一类，不强行判真手势
print(f"[clf] {len(CLF_CLASSES)} 类 {CLF_CLASSES}, align={CLF_ALIGN}, "
      f"min_hand_px={MIN_HAND_PX:g}, train_side={TRAIN_SIDE}, test_acc={_B['test_acc']}")
def to_mp_image(rgb: numpy.ndarray) -> mediapipe.Image:
    return mediapipe.Image(
        image_format=mediapipe.ImageFormat.SRGB,
        data=numpy.ascontiguousarray(rgb),
    )

def classify_hand(landmarks, width, height, mirror=False):
    """返回 (label, proba) 或 None（手太小 / 退化帧）。"""
    pts = numpy.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=numpy.float32)
    feat = extract(pts, width, height, align=CLF_ALIGN, min_hand_px=MIN_HAND_PX,
                   mirror=mirror)
    if feat is None:
        return None
    proba = clf.predict_proba(feat.reshape(1, -1))[0]
    order = numpy.argsort(proba)[::-1]
    top1, top2 = float(proba[order[0]]), float(proba[order[1]])
    if top1 < CONF_MIN or (top1 - top2) < MARGIN_MIN:
        return (OTHER_LABEL, top1)   # 四不像：兜进「其他」，不强行选真手势
    return CLF_CLASSES[order[0]], top1


_hist = {}
_last = {}

def _vote(side, res):
    """同一只手 5 帧内至少 3 票才认；够票就输出多数票，不够返回不确定。"""
    if res is None:
        _hist.pop(side, None)
        _last.pop(side, None)
        return None
    d = _hist.setdefault(side, deque(maxlen=VOTE_FRAMES))
    d.append(res[0])
    _last.setdefault(side, {})[res[0]] = res[1]
    if len(d) < VOTE_MIN:
        return res
    label, n = Counter(d).most_common(1)[0]
    if n < VOTE_MIN:
        return ("?", res[1])
    return (label, _last[side].get(label, res[1]))

@app.post("/api/Gestureinput")
def detect_gesture(data:Gestureinput):
    frame=decode_base64_image(data.img_64)
    if frame is None:
        return Gestureoutput(
            status="error",
            is_success=False,
            gesture_name="",
            confidence=0.0,
            feedback="无法解码图像",
            state="no_person",
            keypoints=None
        )
    height, width = frame.shape[:2]
    result=get_detector().detect(to_mp_image(frame))
    hands=result.hand_landmarks
    labels=result.handedness

    if not hands:
        return Gestureoutput(
            status="success",
            is_success=False,
            gesture_name="",
            confidence=0.0,
            feedback="没有检测到手",
            state="no_person",
            keypoints=None
        )
    keypoints={}
    scores=[]
    counts=[]

    for i, landmarks in enumerate(hands):
        cat = labels[i][0] if i < len(labels) and labels[i] else None
        side = getattr(cat, "category_name", "").lower() or f"hand{i}"
        score = getattr(cat, "score", 0.0)
        scores.append(score)
        for idx, name in enumerate(HAND_KEYS):
            lm = landmarks[idx]
            keypoints[f"{side}_{name}"] = [lm.x, lm.y, lm.z]
        # 左右手判不准就不镜像：模型做过镜像增广，翻错方向也无所谓，别为此丢帧
        counts.append(_vote(side, classify_hand(
            landmarks, width, height,
            side != TRAIN_SIDE if score >= MIN_SIDE_CONF else False)))
    avg = sum(scores) / len(scores) if scores else 0.0
    low = avg < LOW_CONF
    side_cn = ("双手" if len(hands) > 1 else
               ("左手" if getattr(labels[0][0], "category_name", "") == "Left" else "右手"))
    picked = [c for c in counts if c is not None]
    gated = len(picked) < len(counts)
    unknown = any(c[0] in ("?", OTHER_LABEL) for c in picked)
    bad = gated or low or unknown

    if gated:
        gesture_name = f"{side_cn} 距离过远，未判定"
        feedback = "手离镜头太远，请靠近一点"
        conf = 0.0
    elif unknown:
        gesture_name = f"{side_cn}：{OTHER_LABEL}"
        feedback = "未识别的姿势，按「其他」处理"
        conf = min(c[1] for c in picked)
    else:
        names = " + ".join(c[0] for c in picked)
        conf = min(c[1] for c in picked)
        gesture_name = f"{side_cn}：{names}"
        feedback = "识别置信度偏低，调整姿势" if conf < 0.6 else "识别成功"

    bad = bad or conf < 0.6
    return Gestureoutput(status="success", is_success=not bad,
                         gesture_name=gesture_name,
                         confidence=round(conf, 2),
                         feedback=feedback,
                         state="low_confidence" if bad else "ready",
                         keypoints=keypoints)


# WebSocket 版本：逻辑全在 ws_gesture.py，这里只挂上去。
# 想退回纯 HTTP，把下面两行注释掉即可（HTTP 接口完全不受影响）。
from ws_gesture import router as _ws_router
app.include_router(_ws_router)
