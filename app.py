import io, requests, streamlit as st
from PIL import Image, ImageDraw

st.title("Object Detector")
conf = st.slider("Min confidence", 0.0, 1.0, 0.4)
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    r = requests.post("http://127.0.0.1:8000/detect",
                      files={"file": ("img.jpg", buf.getvalue(), "image/jpeg")})
    dets = [d for d in r.json()["detections"] if d["confidence"] >= conf]
    draw = ImageDraw.Draw(img)
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=5)
        draw.text((x1 + 6, y1 + 6), f'{d["label"]} {d["confidence"]:.2f}', fill="lime")
    st.image(img, width="stretch")
    st.write(f"{len(dets)} objects found")
    st.json(dets)