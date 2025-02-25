# server.py
import argparse
import asyncio
import json
import os

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from concurrent.futures import ThreadPoolExecutor

pcs = set()

# 전역 변수 (서버 측 data channel)
global_data_channel = None

# 커스텀 모델 로드 (모델 파일 경로 수정)
try:
    model_path = "weights/yolov11n_20250224_00_19_09_e50b32_dataset_all_calling_drinking_aug/weights/best.pt"
    model = YOLO(model_path)
    print("YOLO 모델 로드 성공!", flush=True)
except Exception as e:
    print("모델 로드 중 에러 발생:", e)
    model = None

# 모델 웜업: dummy 이미지를 이용해 예비 인퍼런스 수행
if model is not None:
    dummy_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    try:
        _ = model.predict(dummy_img, verbose=False)
        print("모델 웜업 완료!", flush=True)
    except Exception as e:
        print("모델 웜업 중 에러:", e)

class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track, data_channel=None, inference_interval=5):
        super().__init__()
        self.track = track
        self.data_channel = data_channel  # 초기에는 None일 수 있음
        self.inference_interval = inference_interval
        self.frame_count = 0
        self.last_detection_result = None  # (detection_data, names) 튜플
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def run_inference(self, rgb_img):
        loop = asyncio.get_event_loop()
        if model is None:
            raise Exception("모델이 로드되지 않았습니다. 모델 파일 경로를 확인하세요.")
        return await loop.run_in_executor(self.executor, lambda: model.predict(rgb_img, verbose=False))

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # 매 inference_interval 프레임마다 인퍼런스 수행
        if self.frame_count % self.inference_interval == 0:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                results = await self.run_inference(rgb_img)
                res = results[0] if isinstance(results, list) else results

                if res.boxes is not None and len(res.boxes) > 0:
                    data = res.boxes.data
                    data = data.cpu().numpy() if data.is_cuda else data.numpy()
                    self.last_detection_result = (data, res.names)
                else:
                    self.last_detection_result = None
                print("인퍼런스 수행:", self.last_detection_result)
            except Exception as e:
                print("YOLO 인퍼런스 중 에러 발생:", e)
                self.last_detection_result = None

        # 만약 data_channel이 아직 할당되지 않았다면 전역 변수에서 가져옴
        global global_data_channel
        if self.data_channel is None and global_data_channel is not None:
            self.data_channel = global_data_channel

        annotated_img = img.copy()
        highest_label = ""
        highest_confidence = 0.0
        if self.last_detection_result is not None:
            data, names = self.last_detection_result
            if len(data) > 0:
                highest_idx = int(np.argmax(data[:, 4]))
                highest_confidence = data[highest_idx, 4]
                highest_label = names[int(data[highest_idx, 5])]
                print("Highest detection:", highest_label, "with confidence:", highest_confidence)
                for box in data:
                    x1, y1, x2, y2, conf, cls_id = box
                    label = names[int(cls_id)]
                    cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(annotated_img, label, (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # data_channel을 통해 최고 detection 정보 전송
        if self.data_channel and self.data_channel.readyState == "open":
            try:
                self.data_channel.send(json.dumps({
                    "label": highest_label,
                    "confidence": float(highest_confidence)
                }))
            except Exception as e:
                print("데이터 채널 전송 에러:", e)

        new_frame = VideoFrame.from_ndarray(annotated_img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)
    print("PeerConnection 생성됨")

    # 서버에서 data channel을 직접 생성하여 SDP에 포함시킵니다.
    data_channel = pc.createDataChannel("detection")
    data_channel.onopen = lambda: print("서버 데이터 채널 열림")
    global global_data_channel
    global_data_channel = data_channel

    @pc.on("track")
    def on_track(track):
        print("수신된 트랙:", track.kind)
        if track.kind == "video":
            local_video = VideoTransformTrack(track, global_data_channel)
            pc.addTrack(local_video)

        @track.on("ended")
        async def on_ended():
            print("트랙 종료됨")

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    response = web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }),
    )
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)

@web.middleware
async def cors_middleware(request, handler):
    if request.method == 'OPTIONS':
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        }
        return web.Response(status=200, headers=headers)
    response = await handler(request)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC YOLO 객체 탐지 서버")
    parser.add_argument("--host", default="0.0.0.0", help="서버 호스트")
    parser.add_argument("--port", type=int, default=8080, help="서버 포트")
    args = parser.parse_args()

    app = web.Application(middlewares=[cors_middleware])
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/offer", offer)
    web.run_app(app, host=args.host, port=args.port)
