import cv2
import time
import pygame
from ultralytics import YOLO
from flask import Flask, render_template, Response
from flask_socketio import SocketIO

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app)

# YOLO 모델 로드
model_1 = YOLO("weights/yolov11n_20250226_075134_e50b32_dataset_face_class_only/weights/best.pt")
model_2 = YOLO("weights/yolov11n_20250226_01_41_20_e50b32_dataset_calling_drinking_only/weights/best.pt")

# 클래스 이름 정의
class_names_1 = ["Distract","Safety Driving", "Sleepy Driving", "Yawn"]
class_names_2 = ["Calling", "Drinking"]

# pygame 초기화 및 mp3 파일 로드
pygame.mixer.init()
sleepy_alarm = pygame.mixer.Sound("asset/sleepy.mp3")
distract_alarm = pygame.mixer.Sound("asset/distract.mp3")
yawn_alarm = pygame.mixer.Sound("asset/yawn.mp3")
calling_alarm = pygame.mixer.Sound("asset/calling.mp3")
drinking_alarm = pygame.mixer.Sound("asset/drinking.mp3")

# 이상행동 감지 관련 변수
sleepy_start_time = None  # 졸음운전 감지 시작 시간
sleepy_detected = False   # 졸음운전 감지 상태 플래그
distract_start_time = None  # 주의산만 감지 시작 시간
distract_detected = False   # 주의산만 감지 상태 플래그
yawn_start_time = None  # 하품 감지 시작 시간
yawn_detected = False   # 하품 감지 상태 플래그
calling_start_time = None  # 통화 감지 시작 시간
calling_detected = False   # 통화 감지 상태 플래그
drinking_start_time = None  # 음료섭취 감지 시작 시간
drinking_detected = False   # 음료섭취 감지 상태 플래그

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    global sleepy_detected, sleepy_start_time, distract_detected, distract_start_time, yawn_start_time, yawn_detected
    global call_start_time, call_detected, drink_start_time, drink_detected
    cap = cv2.VideoCapture(0)  # 웹캠으로부터 영상 캡처
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Distract, Safe Driving, Sleepy Driving, Yawn 탐지
        results_1 = model_1.predict(source=frame, conf=0.1)
        sleepy_present = False  # 이번 프레임에서 졸음운전 감지 여부
        distract_present = False  # 이번 프레임에서 주의산만 감지 여부
        yawn_present = False  # 이번 프레임에서 하품 감지 여부
        yawn_count = 0

        for result in results_1:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                label = f"{class_names_1[class_id]}: {box.conf[0]:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # "Sleepy Driving" 감지 여부 체크
                if class_id == class_names_1.index("Sleepy Driving"):
                    sleepy_present = True
                elif class_id == class_names_1.index("Distract") :
                    distract_present = True
                elif class_id == class_names_1.index("Yawn") :
                    yawn_present = True
                    yawn_count += 1
                    

        # 졸음운전이 감지되었을 경우
        if sleepy_present:
            if not sleepy_detected:  # 처음 감지된 경우
                sleepy_start_time = time.time()
                sleepy_detected = True
            else:
                # 감지가 2초 이상 지속된 경우 mp3 재생
                if time.time() - sleepy_start_time >= 2:
                    if not pygame.mixer.get_busy():  # 이미 재생 중이 아닐 경우에만 실행
                        sleepy_alarm.play()
        else:
            # 졸음운전이 감지되지 않으면 타이머 및 플래그 리셋
            sleepy_start_time = None
            sleepy_detected = False

        # 주의산만 감지되었을 경우
        if distract_present :
            if not distract_detected:  # 처음 감지된 경우
                distract_start_time = time.time()
                distract_detected = True
            else:
                # 감지가 4초 이상 지속된 경우 mp3 재생
                if time.time() - distract_start_time >= 4:
                    if not pygame.mixer.get_busy():  # 이미 재생 중이 아닐 경우에만 실행
                        distract_alarm.play()

        else:
            # 주의산만 감지되지 않으면 타이머 및 플래그 리셋
            distract_start_time = None
            distract_detected = False

        # 하품 감지되었을 경우
        if yawn_present :
            if not yawn_detected:  # 처음 감지된 경우
                yawn_start_time = time.time()
                yawn_detected = True
            else:
                # 감지가 4초 이상 지속된 경우 mp3 재생
                if time.time() - yawn_start_time >=2:
                    if not pygame.mixer.get_busy():  # 이미 재생 중이 아닐 경우에만 실행
                        yawn_alarm.play()

        else:
            # 하품이 감지되지 않으면 타이머 및 플래그 리셋
            yawn_start_time = None
            yawn_detected = False


        ## Calling, Drinking 탐지
        results_2 = model_2.predict(source=frame, conf=0.25)
        calling_present = False
        drinking_present = False

        for result in results_2:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                label = f"{class_names_2[class_id]}: {box.conf[0]:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if class_id == class_names_2.index("Calling"):
                    calling_present = True
                elif class_id == class_names_2.index("Drinking") :
                    drinking_present = True

        # 통화 감지되었을 경우
        if calling_present :
            if not calling_detected:  # 처음 감지된 경우
                calling_start_time = time.time()
                calling_detected = True
            else:
                # 감지가 4초 이상 지속된 경우 mp3 재생
                if time.time() - calling_start_time >= 4:
                    if not pygame.mixer.get_busy():  # 이미 재생 중이 아닐 경우에만 실행
                        calling_alarm.play()

        else:
            # 주의산만 감지되지 않으면 타이머 및 플래그 리셋
            calling_start_time = None
            calling_detected = False

        # 음료섭취 감지되었을 경우
        if drinking_present :
            if not drinking_detected:  # 처음 감지된 경우
                drinking_start_time = time.time()
                drinking_detected = True
            else:
                # 감지가 4초 이상 지속된 경우 mp3 재생
                if time.time() - drinking_start_time >= 4:
                    if not pygame.mixer.get_busy():  # 이미 재생 중이 아닐 경우에만 실행
                        drinking_alarm.play()

        else:
            # 주의산만 감지되지 않으면 타이머 및 플래그 리셋
            drinking_start_time = None
            drinking_detected = False

        # 결과 프레임을 웹 브라우저로 전송
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)