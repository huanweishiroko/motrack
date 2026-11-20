from fastapi import FastAPI
from pydantic import BaseModel
import mediapipe
import base64
import numpy
import cv2
from fastapi.middleware.cors import CORSMiddleware
app=FastAPI(title="姿态交互系统")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Gestureinput(BaseModel):
    img_64:str

class Gestureoutput(BaseModel):
    status:str #success or error
    is_success:bool 
    gesture_name:str
    confidence:float #置信度
    feedback:str #反馈，动作微调
    state:str
    keypoints: dict[str,list[float]]|None=None


def decode_base64_image(img_str):
    if "," in img_str:
        img_str=img_str.split(",")[1]
    img_bytes=base64.b64decode(img_str)
    numpy_arr=numpy.frombuffer(img_bytes,numpy.uint8)
    frame=cv2.imdecode(numpy_arr,cv2.IMREAD_COLOR_RGB)
    print(frame.shape)
    return frame

@app.post("/api/Gestureinput")
def detect_gesture(data:Gestureinput):
    frame=decode_base64_image(data.img_64)
    #mediapipe推理

    return Gestureoutput(
        status="success",
        is_success=True,
        gesture_name="双手平举",
        confidence=0.95,
        feedback="动作标准",
        state="ready",
        keypoints={"left_shoulder": [0.5, 0.6], "right_shoulder": [0.5, 0.4]}
    )