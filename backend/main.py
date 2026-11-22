from fastapi import FastAPI
from pydantic import BaseModel
import mediapipe
import base64
import numpy
import cv2
import os
import urllib.request
from fastapi.middleware.cors import CORSMiddleware

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
app=FastAPI(title="姿态交互系统")
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

def to_mp_image(rgb: numpy.ndarray) -> mediapipe.Image:
    return mediapipe.Image(
        image_format=mediapipe.ImageFormat.SRGB,
        data=numpy.ascontiguousarray(rgb),
    )

detector=make_detector(ensure_model(DEFAULT_MODEL), num_hands=2, video_mode=False)



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

    for i, landmarks in enumerate(hands):
        cat = labels[i][0] if i < len(labels) and labels[i] else None
        side = getattr(cat, "category_name", "").lower() or f"hand{i}"
        score = getattr(cat, "score", 0.0)
        scores.append(score)
        for idx, name in enumerate(HAND_KEYS):
            lm = landmarks[idx]
            keypoints[f"{side}_{name}"] = [lm.x, lm.y, lm.z]

    avg = sum(scores) / len(scores) if scores else 0.0
    low = avg < LOW_CONF
    side_cn = "双手" if len(hands) > 1 else ("左手" if labels[0][0].category_name == "Left" else "右手")

    return Gestureoutput(status="success", is_success=not low,
                         gesture_name=f"检测到{side_cn}（{len(hands)}只）",
                         confidence=round(avg, 2),
                         feedback="手部关键点已获取" if not low else "识别置信度偏低，调整光照和距离",
                         state="low_confidence" if low else "ready",
                         keypoints=keypoints)
