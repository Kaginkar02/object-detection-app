import cv2
from collections import Counter
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
n = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    res = model(frame, device=0, verbose=False)[0]
    frame = res.plot()
    counts = Counter(model.names[int(c)] for c in res.boxes.cls)
    for i, (name, k) in enumerate(counts.items()):
        cv2.putText(frame, f"{name}: {k}", (10, 30 + 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Detector", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        cv2.imwrite(f"snap_{n}.jpg", frame); n += 1
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()