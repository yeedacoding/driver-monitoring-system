import argparse
import asyncio
import json
import os

import cv2
import numpy as np
import torch
import onnxruntime as ort
from ultralytics import YOLO

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from concurrent.futures import ThreadPoolExecutor

pcs = set()
global_data_channel = None  # 서버 측 데이터 채널 전역 변수

# ✅ CORS 미들웨어 추가
@web.middleware
async def cors_middleware(request, handler):
    if request.method == 'OPTIONS':  # Preflight 요청 처리
        return web.Response(headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        })
    
    response = await handler(request)
    response.headers["Access-Control-Allow-Origin"] = "*"  # 모든 도메인에서 접근 허용
    return response

# 두 개의 YOLO 모델 로드
try:
    model_path_1 = "/Users/taeheon/driver_monitoring/weights/yolov11n_20250226_01_41_20_e50b32_dataset_calling_drinking_only/weights/best.pt"
    model_path_2 = "/Users/taeheon/driver_monitoring/weights/yolov11n_20250226_01_41_20_e50b32_dataset_calling_drinking_only/weights/best.pt"
    model_1 = YOLO(model_path_1)
    model_2 = YOLO(model_path_2)
    # model_1 = ort.InferenceSession("/Users/taeheon/driver_monitoring/weights/yolov11n_20250226_01_41_20_e50b32_dataset_calling_drinking_only/weights/best.onnx")
    # model_2 = ort.InferenceSession("/Users/taeheon/driver_monitoring/weights/yolov11n_20250226_075134_e50b32_dataset_face_class_only/weights/best.onnx")
    print("✅ 두 개의 YOLO 모델 로드 성공!", flush=True)
except Exception as e:
    print("❌ 모델 로드 중 에러 발생:", e)
    model_1, model_2 = None, None

# ✅ 모델 웜업: dummy 이미지를 이용해 예비 인퍼런스 수행
if model_1 and model_2:
    # dummy_img = np.ones((1, 3, 320, 320), dtype=np.float32)  # ONNX 입력 형식
    dummy_img = np.ones((480, 640, 3), dtype=np.uint8) * 255  # pt 입력 형식
    try:
        _ = model_1.predict(dummy_img, verbose=False)
        _ = model_2.predict(dummy_img, verbose=False)
        # _ = model_1.run(None, {'images' : dummy_img})
        # _ = model_2.run(None, {'images' : dummy_img})
        print("✅ 모델 웜업 완료!", flush=True)
    except Exception as e:
        print("❌ 모델 웜업 중 에러:", e)

class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track, data_channel=None, inference_interval=10):
        super().__init__()
        self.track = track
        self.data_channel = data_channel
        self.inference_interval = inference_interval
        self.frame_count = 0
        self.executor = ThreadPoolExecutor(max_workers=2)

    async def run_inference(self, rgb_img):
        loop = asyncio.get_event_loop()

        # ONNX 모델 입력 전처리 (정규화 및 차원 변환)
        img_resized = cv2.resize(rgb_img, (640, 640))  # ONNX 입력 크기로 리사이즈
        img_float = img_resized.astype(np.float32) / 255.0  # 정규화
        img_onnx = np.expand_dims(img_float.transpose(2, 0, 1), axis=0)  # (H, W, C) → (1, C, H, W)

        results_1, results_2 = await asyncio.gather(
            loop.run_in_executor(self.executor, lambda: model_1.predict(rgb_img, verbose=False)),
            loop.run_in_executor(self.executor, lambda: model_2.predict(rgb_img, verbose=False))
            # loop.run_in_executor(self.executor, lambda: model_1.run(None, {'images' : img_onnx})),
            # loop.run_in_executor(self.executor, lambda: model_2.run(None, {'images' : img_onnx}))
        )
        print("results_1 shape : ", results_1.shape, "results_2 shape : ", results_2.shape)
        return results_1[0], results_2[0]

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % self.inference_interval == 0:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                results_1, results_2 = await self.run_inference(rgb_img)

                def extract_detection_data(results):
                    if results.boxes is not None and len(results.boxes) > 0:
                        data = results.boxes.data.cpu().numpy()
                        highest_idx = int(np.argmax(data[:, 4]))
                        highest_confidence = data[highest_idx, 4]
                        highest_label = results.names[int(data[highest_idx, 5])]
                        return highest_label, highest_confidence
                    return "None", 0.0

                highest_label_1, highest_confidence_1 = extract_detection_data(results_1)
                highest_label_2, highest_confidence_2 = extract_detection_data(results_2)

                print(f"✅ Model 1: {highest_label_1} ({highest_confidence_1:.2f})")
                print(f"✅ Model 2: {highest_label_2} ({highest_confidence_2:.2f})")

            except Exception as e:
                print("❌ YOLO 인퍼런스 중 에러 발생:", e)
                highest_label_1, highest_confidence_1 = "None", 0.0
                highest_label_2, highest_confidence_2 = "None", 0.0

            global global_data_channel
            if self.data_channel is None and global_data_channel is not None:
                self.data_channel = global_data_channel

            if self.data_channel and self.data_channel.readyState == "open":
                try:
                    self.data_channel.send(json.dumps({
                        "model_1_label": highest_label_1,
                        "model_1_confidence": float(highest_confidence_1),
                        "model_2_label": highest_label_2,
                        "model_2_confidence": float(highest_confidence_2)
                    }))
                except Exception as e:
                    print("❌ 데이터 채널 전송 에러:", e)

        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)
    print("✅ PeerConnection 생성됨")

    global global_data_channel
    data_channel = pc.createDataChannel("detection")
    global_data_channel = data_channel
    print("✅ 서버 데이터 채널 생성 완료!")

    @pc.on("track")
    def on_track(track):
        print("🎥 수신된 트랙:", track.kind)
        if track.kind == "video":
            local_video = VideoTransformTrack(track, global_data_channel)
            pc.addTrack(local_video)

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
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

# ✅ 서버 종료 시 PeerConnection 정리
async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC YOLO 객체 탐지 서버 (멀티 모델)")
    parser.add_argument("--host", default="0.0.0.0", help="서버 호스트")
    parser.add_argument("--port", type=int, default=8080, help="서버 포트")
    args = parser.parse_args()

    app = web.Application(middlewares=[cors_middleware])  # ✅ CORS 미들웨어 추가
    app.on_shutdown.append(on_shutdown)  # ✅ 서버 종료 시 PeerConnection 정리
    app.router.add_post("/offer", offer)

    print("🚀 멀티 모델 YOLO 서버 시작...")
    web.run_app(app, host=args.host, port=args.port)
