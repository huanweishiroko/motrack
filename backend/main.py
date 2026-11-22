from fastapi import FastAPI
from pydantic import BaseModel
import mediapipe
import base64
import numpy
import cv2
import os
import math
import urllib.request
from fastapi.middleware.cors import CORSMiddleware

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
app=FastAPI(title="手势交互系统")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL = "hand_landmarker.task"
INDEX_FINGER_TIP = 8

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

FINGERS = [
    ("index_finger_mcp",  "index_finger_pip",  "index_finger_tip"),
    ("middle_finger_mcp", "middle_finger_pip", "middle_finger_tip"),
    ("ring_finger_mcp",   "ring_finger_pip",   "ring_finger_tip"),
    ("pinky_mcp",         "pinky_pip",         "pinky_tip"),
]
FINGER_ANGLE_THRESHOLD = 165.0
THUMB_ANGLE_THRESHOLD = 155.0
MIN_HAND_SIZE_RATIO = 0.18   # 手腕↔中指MCP / 画面短边

class Gestureinput(BaseModel):
    img_64:str

class Gestureoutput(BaseModel):
    status:str #success or error
    is_success:bool 
    gesture_name:str
    confidence:float #置信度
    feedback:str
    state:str
    keypoints: dict[str,list[float]]|None=None

def decode_base64_image(img_str):
    if "," in img_str:
        img_str=img_str.split(",")[1]
    img_bytes=base64.b64decode(img_str)
    numpy_arr=numpy.frombuffer(img_bytes,numpy.uint8)
    frame=cv2.imdecode(numpy_arr,cv2.IMREAD_COLOR_RGB)
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



detector=make_detector(ensure_model(DEFAULT_MODEL), num_hands=2, video_mode=False)

def to_mp_image(rgb: numpy.ndarray) -> mediapipe.Image:
    return mediapipe.Image(
        image_format=mediapipe.ImageFormat.SRGB,
        data=numpy.ascontiguousarray(rgb),
    )

def dist(a, b, width: int, height: int) -> float:
    ax, ay = a.x * width, a.y * height
    bx, by = b.x * width, b.y * height
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

def point_abs(pt,w,h):
    return (pt.x*w,pt.y*h) #未处理z

def angle_between(p_mcp, p_pip, p_tip, width: int, height: int) -> float:

    mcp = point_abs(p_mcp, width, height)
    pip = point_abs(p_pip, width, height)
    tip = point_abs(p_tip, width, height)

    v1 = (mcp[0] - pip[0], mcp[1] - pip[1])
    v2 = (tip[0] - pip[0], tip[1] - pip[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    len1 = math.hypot(v1[0], v1[1])
    len2 = math.hypot(v2[0], v2[1])

    if len1 < 1e-6 or len2 < 1e-6:
        return 0.0
    cos_ang = max(min(dot / (len1 * len2), 1.0), -1.0)
    rad = math.acos(cos_ang)
    return math.degrees(rad)

def count_extended_fingers(landmarks, width: int, height: int) -> int:
    by_name = dict(zip(HAND_KEYS, landmarks))
    count = 0

    for mcp_key, pip_key, tip_key in FINGERS:
        mcp = by_name[mcp_key]
        pip = by_name[pip_key]
        tip = by_name[tip_key]
        ang = angle_between(mcp, pip, tip, width, height)
        print(f"[finger] {mcp_key:<18} angle={ang:.1f}")
        if ang >= FINGER_ANGLE_THRESHOLD:
            count += 1

    th_mcp = by_name["thumb_mcp"]
    th_ip = by_name["thumb_ip"]
    th_tip = by_name["thumb_tip"]
    th_ang = angle_between(th_mcp, th_ip, th_tip, width, height)
    print(f"[finger] {'thumb_mcp':<18} angle={th_ang:.1f}")
    if th_ang >= THUMB_ANGLE_THRESHOLD:
        count += 1

    return count

def hand_size_ratio(landmarks, width: int, height: int) -> float:
    by_name = dict(zip(HAND_KEYS, landmarks))
    wrist = by_name["wrist"]
    middle_mcp = by_name["middle_finger_mcp"]
    return dist(wrist, middle_mcp, width, height) / min(width, height)

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
            state="no_person", #no_person仅占位
            keypoints=None
        )
    height, width = frame.shape[:2]
    result=detector.detect(to_mp_image(frame))
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
        size = hand_size_ratio(landmarks, width, height)
        print(f"[size] {size:.3f}")
        if size < MIN_HAND_SIZE_RATIO:
            counts.append(None)
        else:
            counts.append(count_extended_fingers(landmarks, width, height))
    avg = sum(scores) / len(scores) if scores else 0.0
    low = avg < LOW_CONF
    side_cn = "双手" if len(hands) > 1 else ("左手" if labels[0][0].category_name == "Left" else "右手")
    finger_cn = " + ".join("?" if c is None else str(c) for c in counts)
    gated = any(c is None for c in counts)
    bad = gated or low

    if gated:
        gesture_name = f"{side_cn} 距离过远，未判定"
        feedback = "手离镜头太远，请靠近一点"
    else:
        gesture_name = f"{side_cn} 伸出 {finger_cn} 指"
        feedback = "识别置信度偏低，调整光照和距离" if low else "手部关键点已获取"

    return Gestureoutput(status="success", is_success=not bad,
                         gesture_name=gesture_name,
                         confidence=round(avg, 2),
                         feedback=feedback,
                         state="low_confidence" if bad else "ready",
                         keypoints=keypoints)