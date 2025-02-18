import mediapipe as mp
import numpy as np
import time


### 1. EAR(눈 감음) 계산 (SleepyDriving)
def calc_ear(landmarks) :
    left_eye = np.linalg.norm(np.array(landmarks[159]) - np.array(landmarks[145]))
    right_eye = np.linalg.norm(np.array(landmarks[386]) - np.array(landmarks[374]))

    ear = (left_eye + right_eye) / 2.0

    return ear

# Yawn(입 개방 정도) 계산
def calc_yawn(landmarks) :
    upper_lip = np.linalg.norm(np.array(landmarks[13]) - np.array(landmarks[14]))
    lower_lip = np.linalg.norm(np.array(landmarks[308]) - np.array(landmarks[78]))

    mouth_open = upper_lip / lower_lip

    return mouth_open


### 2. 손 위치 검출 (Calling) : 손이 얼글 측면(귀 부근)에 가까운지 판단
def hand_near_ear(hand_landmarks, face_landmarks, img_width, img_height):
    hand_x, hand_y = hand_landmarks[0][:2]  # 손 중심 좌표 (정규화 값)
    right_ear_x, right_ear_y = face_landmarks[132][:2]  # 오른쪽 귀 좌표
    left_ear_x, left_ear_y = face_landmarks[361][:2]  # 왼쪽 귀 좌표

    # 픽셀 좌표로 변환
    right_ear_distance = np.linalg.norm([hand_x - right_ear_x, hand_y - right_ear_y])
    left_ear_distance = np.linalg.norm([hand_x - left_ear_x, hand_y - left_ear_y])

    return right_ear_distance, left_ear_distance


### 3. 손 위치 검출 (Drinking) : 손이 입 근처에 위치한지 판단
def hand_near_mouth(hand_landmarks, face_landmarks):
    hand_x, hand_y = hand_landmarks[0][:2]  # 손 중심 좌표
    mouth_x, mouth_y = face_landmarks[13][:2]  # 입 좌표

    mouth_distance = np.linalg.norm([hand_x - mouth_x, hand_y - mouth_y])

    return mouth_distance

### 4. 정면 주시 (SafetyDriving)
import numpy as np

def get_head_pose(landmarks_face, landmarks_pose):
    """
    얼굴과 자세 랜드마크를 기반으로 머리의 움직임과 회전을 감지하여 정면 주시 여부를 판단
    - landmarks_face: Mediapipe FaceMesh에서 추출한 얼굴 랜드마크
    - landmarks_pose: Mediapipe Pose에서 추출한 신체 랜드마크
    - 반환값: (horizontal_movement, vertical_movement, rotation_angle)
    """
    nose_tip = np.array(landmarks_face[1])      # 코 끝
    chin = np.array(landmarks_face[199])        # 턱 끝

    # 어깨 중앙 위치 계산
    left_shoulder = np.array(landmarks_pose[11])   # 왼쪽 어깨
    right_shoulder = np.array(landmarks_pose[12])  # 오른쪽 어깨
    shoulder_mid = (left_shoulder + right_shoulder) / 2

    # 1️⃣ **머리의 좌우(horizontal) 이동 감지** (코 기준)
    horizontal_movement = np.linalg.norm(nose_tip[:2] - shoulder_mid[:2])  

    # 2️⃣ **머리의 상하(vertical) 이동 감지** (코 기준)
    vertical_movement = np.linalg.norm(nose_tip[1] - shoulder_mid[1])

    # 3️⃣ **머리 회전 감지** (코와 턱을 연결한 벡터의 기울기)
    head_vector = nose_tip - chin
    rotation_angle = np.arctan2(head_vector[1], head_vector[0])  # 회전 각도 계산

    return horizontal_movement, vertical_movement, rotation_angle


### 6. 행동 감지 알림
def classify_behavior(ear, mouth_open, mouth_distance, left_ear_distance, right_ear_distance, horizontal_movement, vertical_movement, rotation_angle):
    if right_ear_distance < 0.1 or left_ear_distance < 0.1:
        behavior = "Calling"
    elif mouth_distance < 0.15:
        behavior = "Drinking"
    elif mouth_open > 0.6:
        behavior = "Yawn"
    elif horizontal_movement > 0.15 or abs(rotation_angle) > np.radians(15):  # 고개를 좌우로 돌리거나 머리가 많이 이동
        behavior = "Distracted"
    elif horizontal_movement <= 0.05 and vertical_movement <= 0.05 and abs(rotation_angle) <= np.radians(10):
        behavior = "SafeDriving"
    elif ear < 0.015:
        behavior = "SleepyDriving"
    
    return behavior
