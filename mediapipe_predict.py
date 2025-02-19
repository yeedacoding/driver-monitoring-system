from mp_method import *
import cv2

#  웹캠 설정
cap = cv2.VideoCapture(0)  # 0번 카메라 사용 (내장 웹캠)

# 프레임 크기 조정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 모델 초기화
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 스트리밍 시작
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("웹캠을 열 수 없습니다.")
        break

    img_height, img_width, _ = frame.shape
    # detect_objects(frame)
    detected_classes = detect_objects(frame)
    print(detected_classes)

    # BGR → RGB 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 얼굴 및 손 랜드마크 검출
    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)
    pose_results = pose.process(rgb_frame)

    # 초기값 설정
    landmarks = {}
    landmarks_pose = {}
    ear = mouth_open = horizontal_movement = vertical_movement = rotation_angle = 1000
    mouth_distance = left_ear_distance = right_ear_distance = 1000


    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                x, y = int(landmark.x * img_width), int(landmark.y * img_height)

                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # 랜드마크 점 찍기

                if idx in [159, 145, 386, 374, 13, 14, 308, 78, 132, 361]:
                    cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 1)
            
            # 얼굴 랜드마크 좌표 추출
            landmarks = {idx: [landmark.x, landmark.y] for idx, landmark in enumerate(face_landmarks.landmark)}
            
            # EAR 및 입 개방도 계산
            ear = calc_ear(landmarks)
            mouth_open = calc_yawn(landmarks)

            
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x, y = int(landmark.x * img_width), int(landmark.y * img_height)

                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # 랜드마크 점 찍기

                if idx in [0]:
                    cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                # print(f"Hand detected: {hand_results.multi_hand_landmarks is not None}")

            hand_positions = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
            mouth_distance = hand_near_mouth(hand_positions, landmarks)
            right_ear_distance, left_ear_distance = hand_near_ear(hand_positions, landmarks, img_width, img_height)

            # print(near_mouth, mouth_distance, right_ear_distance, left_ear_distance)

    
    if pose_results.pose_landmarks:
        landmarks_pose = {idx: [landmark.x, landmark.y] for idx, landmark in enumerate(pose_results.pose_landmarks.landmark)}

    if landmarks and landmarks_pose:
        horizontal_movement, vertical_movement, rotation_angle = get_head_pose(landmarks, landmarks_pose)

    # 수치 테스트
    cv2.putText(frame, f"Object Detected : {detected_classes}", (1300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Mouth open : {round(mouth_open, 2)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Eyes open : {round(ear,2)}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Hand-Left ear dist : {round(left_ear_distance, 1)}", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Hand-Right ear dist : {round(right_ear_distance, 1)}", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Hand-Mouth dist : {round(mouth_distance, 1)}", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Head Move H: {round(horizontal_movement, 2)}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Head Move V: {round(vertical_movement, 2)}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Rotation Angle: {round(rotation_angle, 2)}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 행동 분류
    behavior = classify_behavior(ear, mouth_open, mouth_distance, left_ear_distance, right_ear_distance, horizontal_movement, vertical_movement, rotation_angle, detected_classes)

    # 결과 출력
    cv2.putText(frame, behavior, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # 화면 출력
    cv2.imshow("Driver Monitoring", frame)

    # 종료 키: ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 종료 후 리소스 해제
cap.release()
cv2.destroyAllWindows()
