import cv2, numpy as np
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO

app = FastAPI(title="Object Detection API")
model = YOLO("yolov8n.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    data = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    res = model(img, device=0, verbose=False)[0]
    return {"detections": [
        {"label": model.names[int(c)],
         "confidence": round(float(p), 3),
         "box": [round(float(v), 1) for v in b]}
        for c, p, b in zip(res.boxes.cls, res.boxes.conf, res.boxes.xyxy)
    ]}