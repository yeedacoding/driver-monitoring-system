import cv2
import time
import pygame
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template, Response
from flask_socketio import SocketIO

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app, async_mode="threading")  # 비동기 모드 설정

# ✅ ONNX 모델 로드
onnx_model_path_1 = "weights/yolov11n_20250226_075134_e50b32_dataset_face_class_only/weights/best.onnx"
onnx_model_path_2 = "weights/yolov11n_20250226_01_41_20_e50b32_dataset_calling_drinking_only/weights/best.onnx"

session_1 = ort.InferenceSession(onnx_model_path_1, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
session_2 = ort.InferenceSession(onnx_model_path_2, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# ✅ 클래스 정의
class_names_1 = ["Distracted", "SafeDriving", "SleepyDriving", "Yawn"]
class_names_2 = ["Calling", "Drinking"]

# ✅ pygame 초기화 및 알람 파일 로드
pygame.mixer.init()
alarms = {
    "SleepyDriving": pygame.mixer.Sound("asset/sleepy.mp3"),
    "Distracted": pygame.mixer.Sound("asset/distract.mp3"),
    "Yawn": pygame.mixer.Sound("asset/yawn.mp3"),
    "Calling": pygame.mixer.Sound("asset/calling.mp3"),
    "Drinking": pygame.mixer.Sound("asset/drinking.mp3"),
}

# ✅ 감지 상태 변수 초기화
state_flags = {
    "Distracted": {"start_time": None, "detected": False},
    "SafeDriving": {"start_time": None, "detected": False},  # 알람 없음
    "SleepyDriving": {"start_time": None, "detected": False},
    "Yawn": {"start_time": None, "detected": False},
    "Calling": {"start_time": None, "detected": False},
    "Drinking": {"start_time": None, "detected": False},
}

@app.route('/')
def index():
    return render_template('index.html')

def detect_objects(session, frame, class_names, color):
    """ ONNX 모델을 사용하여 객체 감지 및 바운딩 박스 그리기 """
    input_size = (640, 640)
    height, width, _ = frame.shape  # 원본 영상 크기
    scale_x = width / input_size[0]  # 가로 스케일링 비율
    scale_y = height / input_size[1]  # 세로 스케일링 비율

    # YOLO ONNX 입력 전처리
    image_resized = cv2.resize(frame, input_size)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_input = image_rgb.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    image_input = image_input[np.newaxis, :, :, :].astype(np.float32) / 255.0

    # 모델 실행
    outputs = session.run(None, {"images": image_input})
    detections = outputs[0][0]  # (1, N, 6) → (N, 6)

    detected_classes = []
    bounding_boxes = []  # 바운딩 박스 리스트

    for det in detections:
        x1, y1, x2, y2, confidence, class_id = det[:6]

        # 좌표를 원본 프레임 크기에 맞게 변환
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        if confidence > 0.3 and 0 <= class_id < len(class_names):  # 신뢰도 필터링 및 클래스 범위 체크
            class_name = class_names[int(class_id)]
            detected_classes.append(class_name)
            bounding_boxes.append((x1, y1, x2, y2, class_name, confidence))

    return detected_classes, bounding_boxes

def process_alerts(detected_classes):
    """ 감지된 행동에 따라 경고 알람 재생 (SafeDriving 제외) """
    current_time = time.time()

    for class_name in state_flags:
        if class_name == "SafeDriving":  # ✅ SafeDriving은 무시
            continue  

        state = state_flags[class_name]

        if class_name in detected_classes:  # ✅ 감지됨
            if not state["detected"]:  # 처음 감지된 경우
                state["start_time"] = current_time
                state["detected"] = True
            else:
                # SleepyDriving은 2초 후, 나머지는 4초 후 알람
                required_time = 2 if class_name == "SleepyDriving" else 4
                
                if state["start_time"] is not None and current_time - state["start_time"] >= required_time:
                    if class_name in alarms and alarms[class_name].get_num_channels() == 0:  # ✅ KeyError 방지
                        alarms[class_name].play()
                        state["start_time"] = current_time  # ✅ 알람 재생 후 시간 리셋

        else:  # ✅ 감지가 안 되면 상태 초기화
            if state["detected"]:  # ✅ 기존에 감지되었다가 사라진 경우에만 상태 초기화
                state["start_time"] = None
                state["detected"] = False

def gen_frames():
    """ 웹캠에서 실시간 영상 받아오기 & YOLO ONNX 추론 """
    cap = cv2.VideoCapture(0)  # 웹캠 활성화
    # cap.set(cv2.CAP_PROP_FPS, 30)  # FPS 제한

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 두 번째 모델 (Calling, Drinking) 먼저 실행
        detected_classes_2, bounding_boxes_2 = detect_objects(session_2, frame, class_names_2, (0, 0, 255))

        if detected_classes_2:  # Calling, Drinking이 감지되면 다른 감지 차단
            detected_classes = detected_classes_2
            bounding_boxes = bounding_boxes_2
        else:
            # Calling, Drinking이 감지되지 않으면 첫 번째 모델 실행
            detected_classes_1, bounding_boxes_1 = detect_objects(session_1, frame, class_names_1, (0, 255, 0))
            detected_classes = detected_classes_1
            bounding_boxes = bounding_boxes_1  

        # 감지된 객체에 따른 알람 처리
        process_alerts(detected_classes)

        # 감지된 데이터 확인 (터미널 출력)
        print(f"📌 감지된 행동: {detected_classes}")

        # 감지된 행동을 웹으로 전송
        socketio.emit("detected_actions", {"actions": detected_classes})

        # 바운딩 박스 & 라벨 표시 (해당 모델의 감지만)
        for x1, y1, x2, y2, class_name, confidence in bounding_boxes:
            color = (0, 0, 255) if class_name in class_names_2 else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 웹 스트리밍을 위한 프레임 변환
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8000, debug=False)
