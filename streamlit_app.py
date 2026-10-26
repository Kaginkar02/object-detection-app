import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO

@st.cache_resource
def load():
    return YOLO("yolov8n.pt")

model = load()
st.title("Object Detector")
conf = st.slider("Min confidence", 0.0, 1.0, 0.4)
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    res = model(img, conf=conf, verbose=False)[0]
    draw = ImageDraw.Draw(img)
    for b, c, p in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
        x1, y1, x2, y2 = map(float, b)
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=4)
        draw.text((x1 + 5, y1 + 5), f"{model.names[int(c)]} {float(p):.2f}", fill="lime")
    st.image(img, width="stretch")
    st.write(f"{len(res.boxes)} objects found")
