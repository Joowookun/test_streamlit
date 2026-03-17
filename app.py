import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase # 클래스명 변경
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os
import urllib.request
import math
import av # 추가: PyAV 라이브러리 필요

# 1. 모델 파일 확인 및 다운로드
MODEL_PATH = "pose_landmarker.task"
if not os.path.exists(MODEL_PATH):
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

# VideoTransformerBase -> VideoProcessorBase로 변경
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.timestamp = 0
        self.ANGLE_THRESHOLD = 65

    # transform -> recv로 변경
    def recv(self, frame):
        # frame은 av.VideoFrame 객체입니다.
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.timestamp += 1
        results = self.detector.detect_for_video(mp_image, self.timestamp)
        
        if results.pose_landmarks:
            for landmarks in results.pose_landmarks:
                nose = landmarks[0]
                l_sh = landmarks[11]
                r_sh = landmarks[12]

                mid_sh_y = (l_sh.y + r_sh.y) / 2
                mid_sh_z = (l_sh.z + r_sh.z) / 2

                dz = abs(nose.z - mid_sh_z)
                dy = abs(mid_sh_y - nose.y)

                tilt_angle = math.degrees(math.atan2(dz, dy + 1e-6))

                if tilt_angle > self.ANGLE_THRESHOLD:
                    color = (0, 0, 255)
                    status = f"WARNING: Forward Lean {int(tilt_angle)} deg"
                else:
                    color = (0, 255, 0)
                    status = f"Good: Lean {int(tilt_angle)} deg"

                cv2.line(img, (int(l_sh.x * w), int(l_sh.y * h)), (int(r_sh.x * w), int(r_sh.y * h)), color, 2)
                cv2.circle(img, (int(nose.x * w), int(nose.y * h)), 6, color, -1)
                cv2.putText(img, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
        # 결과물을 다시 av.VideoFrame 형태로 변환하여 반환해야 합니다.
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI 부분 ---
st.set_page_config(page_title="Front-view Neck Angle Analyzer")
st.title("📏 정면 기반 목 기울기 분석기")
st.markdown("""
정면을 바라본 상태에서 **목이 어깨보다 얼마나 앞으로 나왔는지**를 3D 깊이 값으로 분석하여 각도를 산출합니다.
- **Good:** 수직에 가까운 상태
- **Warning:** 목이 앞으로 65도 이상 기울어진 상태
""")

webrtc_streamer(
    key="neck-angle-analysis",
    # video_transformer_factory -> video_processor_factory로 변경
    video_processor_factory=PoseProcessor,
    rtc_configuration={ # 배포 환경을 위한 STUN 서버 설정 추가
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)
