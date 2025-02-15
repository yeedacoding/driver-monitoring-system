from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("weights/yolov11n_20250214_e50b32/weights/best.pt")

# Single stream with batch-size 1 inference
# source = "udp://127.0.0.1:9090"  # RTSP, RTMP, TCP, or IP streaming address

# Run inference on the source
## 녹화 영상
# results = model.predict(source="test.mov", show=True)

## 스트리밍
results = model.predict(source=0, conf=0.1, show=True)

print(results)