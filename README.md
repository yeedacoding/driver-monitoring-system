# 🚗 운전자 이상 행동 감지 및 알람 프로그램 개발
![Image](https://github.com/user-attachments/assets/b413a67e-2fc1-424a-bc99-9894c1a47d5f)
<br>


## 💡 소개
### 주제
- 운전자 이상 행동 감지 및 경고 시스템
### 목표
- 딥러닝을 활용하여 운전자의 졸음운전, 전방 미주시, 통화 등 이상 행동 탐지 및 즉각적인 경고/알람을 제공하는 운전자 모니터링 시스템 개발
### 기대효과
- 졸음 운전, 음료 섭취 등 위험한 운전 행동을 탐지
- 이상 행동 감지 시 즉각 경고음 또는 시각적 알림 제공
- 운전자 행동 데이터를 분석하여 ITS 같은 교통 시스템을 통하여 지속적인 안전 운전을 유도
<br>

## 👥 팀 구성
| 허승회 | 최선호 | 이정현 | 김태헌 |
|:---:|:---:|:---:|:---:|
| 모델 훈련 및 테스트 <br>WebApp 개발 <br>PPT 작성 및 발표 | EDA <br>모델 훈련 및 테스트 <br> 하이퍼파라미터 최적화 <br>실 운전 환경에서 테스트 진행 | 모델 훈련 및 테스트 <br>하이퍼파라미터 최적화 | EDA <br>모델 훈련 및 테스트 <br>하이퍼파라미터 최적화 <br>WebApp 개발 |
<br>

## ⏰ 개발 기간
- 25.02.12 ~ 25.03.04
#### Timeline
<img width="900" alt="image" src="https://github.com/user-attachments/assets/f64e90b0-c6e3-462a-ae33-037821304d80" /><br>


## 🏔️ 개발 과정
#### 1️⃣ 1차 : EDA(이미지 데이터셋 분석) - [상세 설명](https://github.com/yeedacoding/driver-monitoring-system/wiki)
- YOLO 모델 학습에 사용될 이미지 데이터셋 분석
- ***적용 기술*** : `Pandas`, `Matplotlib`
#### 2️⃣ 2차 : 실시간 운전자 얼굴 및 행동 감지 - [상세 설명](https://github.com/yeedacoding/driver-monitoring-system/wiki/YOLO-%EB%AA%A8%EB%8D%B8-%ED%9B%88%EB%A0%A8-%EB%B0%8F-%ED%85%8C%EC%8A%A4%ED%8A%B8)
- 운전 중 졸음, 스마트폰 사용, 주의 산만 행동 등을 분류하는 딥러닝 모델 개발
- ***적용 기술*** : `YOLOv11 nano`, `Optuna`(하이퍼파라미터 최적화)
#### 3️⃣ 3차 : 운전자 이상 행동 분류에 따른 알람 WebApp 개발 - [상세 설명](https://github.com/yeedacoding/driver-monitoring-system/wiki/%EC%9A%B4%EC%A0%84%EC%9E%90-%EC%9D%B4%EC%83%81-%ED%96%89%EB%8F%99-%EB%B6%84%EB%A5%98%EC%97%90-%EB%94%B0%EB%A5%B8-%EC%95%8C%EB%9E%8C-WebApp-%EA%B0%9C%EB%B0%9C)
- 실제 스트리밍 운전 상황 영상을 테스트하여 이상 행동 감지 시 알람을 제공할 수 있는 WebApp 개발
- ***적용 기술*** : `Flask`, `WebSocket`
#### ➕ 부가 목표 : 얼굴 Landmark 검출을 통한 "졸음" 분석 알고리즘 개발 - [CODE](https://github.com/yeedacoding/driver-monitoring-system/tree/master/mediapipe_test)
- 운전자의 연속적인 행동 패턴 (졸음, 고개 돌림 등)을 얼굴 landmark로부터 학습하여 보다 정확한 졸음 행동 탐지
- ***적용 기술*** : `MediaPipe`
