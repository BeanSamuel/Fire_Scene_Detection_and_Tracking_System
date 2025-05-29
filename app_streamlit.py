import streamlit as st
import cv2
import numpy as np
import torch
import time
import tempfile
from ultralytics import YOLO
from tracker.tracker import Tracker

st.sidebar.title("Fire MOT Demo")

weights_path = st.sidebar.selectbox("Yolo Weights (.pt)", ["models/test.pt"])
source_type = st.sidebar.selectbox("Video Source", ["Webcam", "Video File"])
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

video_file = None
if source_type == "Video File":
    video_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

start_btn = st.sidebar.button("Start")

tracker = Tracker(name="stream")

def load_model(w_path):
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(w_path).to(device)
    return model

def get_yolo_bboxes(results, conf=0.25, target_cls=None):
    dets = []
    if hasattr(results, "boxes"):
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy.cpu().numpy().flatten()
            score = float(box.conf)
            cls = int(box.cls)
            if score < conf:
                continue
            if (target_cls is not None) and (cls != target_cls):
                continue
            dets.append([x1, y1, x2, y2, score])
    dets = np.array(dets, dtype=np.float32)
    if dets.ndim != 2 or dets.shape[1] != 5:
        dets = np.empty((0,5), dtype=np.float32)
    return dets

def draw_tracks(img, tracks):
    img = img.copy()
    for row in tracks:
        x1, y1, x2, y2, tid, conf = row
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        tid = int(tid)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(img, f"ID:{tid}", (x1, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,80,255), 2)
    return img

def run_stream(source, model, conf):
    cap = cv2.VideoCapture(source)
    fps = 0
    prev = time.time()
    frame_show = st.empty()

    tracker.reset(name="stream")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("No more frames to read.")
            break

        with torch.no_grad():
            yolo_results = model(frame, conf=conf, verbose=False)[0]

        dets = get_yolo_bboxes(yolo_results, conf=conf, target_cls=0)
        tracks = tracker.track(frame, dets)  # (M,6) [x1,y1,x2,y2,id,conf]
        frame_out = draw_tracks(frame, tracks)
        fps = fps*0.9 + 0.1*(1/(time.time()-prev+1e-8))
        prev = time.time()
        cv2.putText(frame_out, f"FPS {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50,255,50), 2)
        frame_show.image(frame_out, channels="BGR")
    cap.release()

if start_btn:
    model = load_model(weights_path)
    if source_type == "Webcam":
        run_stream(0, model, confidence)
    else:
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            tfile.flush()
            run_stream(tfile.name, model, confidence)
        else:
            st.sidebar.warning("Please upload a video file before starting.")

