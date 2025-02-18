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

    mouth_open = lower_lip / upper_lip

    return mouth_open


### 2. 손 위치 검출 (Calling) : 손이 얼글 측면(귀 부근)에 가까운지 판단
def hand_near_ear(hand_landmarks, face_landmarks, img_width, img_height):
    """
    손이 얼굴의 귀 근처에 있는지 확인하는 함수
    Mediapipe 얼굴 랜드마크 기준:
    - 오른쪽 귀: 132번
    - 왼쪽 귀: 361번
    """
    hand_x, hand_y = hand_landmarks[0][:2]  # 손 중심 좌표 (정규화 값)
    right_ear_x, right_ear_y = face_landmarks[132][:2]  # 오른쪽 귀 좌표
    left_ear_x, left_ear_y = face_landmarks[361][:2]  # 왼쪽 귀 좌표

    # 픽셀 좌표로 변환
    hand_x, hand_y = hand_x * img_width, hand_y * img_height
    right_ear_x, right_ear_y = right_ear_x * img_width, right_ear_y * img_height
    left_ear_x, left_ear_y = left_ear_x * img_width, left_ear_y * img_height

    right_ear_distance = np.linalg.norm([hand_x - right_ear_x, hand_y - right_ear_y])
    left_ear_distance = np.linalg.norm([hand_x - left_ear_x, hand_y - left_ear_y])

    # 픽셀 단위 거리 기준으로 변경 (30~50픽셀 기준)
    if right_ear_distance < 50 or left_ear_distance < 50:
        return True  
    
    return False


### 3. 손 위치 검출 (Drinking) : 손이 입 근처에 위치한지 판단
def hand_near_mouth(hand_landmarks, face_landmarks):
    hand_x, hand_y = hand_landmarks[0][:2]  # 손 중심 좌표
    mouth_x, mouth_y = face_landmarks[13][:2]  # 입 좌표

    mouth_distance = np.linalg.norm([hand_x - mouth_x, hand_y - mouth_y])

    if mouth_distance < 0.15 :
        return True  # 손이 입 근처에 있는 경우
    
    return False

### 4. 정면 주시 (SafetyDriving)
def get_head_pose(landmarks):
    nose_tip = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    # 코를 기준으로 머리 중심이 정면인지 확인
    horizontal_angle = np.linalg.norm(left_eye - right_eye)

    return horizontal_angle

### 5. 주의 분산 (Distracted)
def is_head_moving(landmarks, prev_landmarks):
    movement = np.linalg.norm(landmarks[1] - prev_landmarks[1])
    
    if movement > 2 :
        return True # 머리가 빠르게 움직이면 주의 분산
    
    return False 

### 6. 행동 감지 알림
def classify_behavior(ear, mouth_open, is_hand_near_face, distracted, is_hand_near_mouth, is_hand_near_ear):
    behavior = "SafeDriving"
    
    if is_hand_near_ear:
        behavior = "Calling"
    elif is_hand_near_mouth:
        behavior = "Drinking"
    elif mouth_open > 1.5:
        behavior = "Yawn"
    elif distracted:
        behavior = "Distracted"
    elif ear < 0.2:
        behavior = "SleepyDriving"
    
    return behavior
