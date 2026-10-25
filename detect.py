from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # auto-downloads (about 6 MB)
model.predict(source=0, show=True, device=0)  # webcam, GPU