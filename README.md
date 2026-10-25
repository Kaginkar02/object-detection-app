\# Real-Time Object Detection App



YOLOv8 detector with a FastAPI backend and a Streamlit UI. Runs on GPU.







!\[demo](assets/demo.gif)







\## Features

\- Live webcam detection with object counts (`detect\_count.py`)

\- REST API: `POST /detect` returns labels, confidence, boxes

\- Web UI: upload an image, adjust the confidence slider



\## Tech stack

Python, PyTorch, Ultralytics YOLOv8, FastAPI, Streamlit, OpenCV



\## Setup

1\. `py -3.12 -m venv .venv` then `.venv\\Scripts\\activate`

2\. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`

3\. `pip install -r requirements.txt`



\## Run

\- API: `uvicorn api:app`

\- UI (new terminal): `streamlit run app.py`

\- Webcam: `python detect\_count.py`



\## Future work

\- Custom-trained model, ONNX/TensorRT speedup, Docker, deployment

