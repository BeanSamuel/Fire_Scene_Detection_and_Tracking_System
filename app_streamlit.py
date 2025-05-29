import streamlit as st
import cv2
import numpy as np
import torch
import time
import tempfile
from ultralytics import YOLO
from tracker.tracker import Tracker

st.sidebar.title("Fire MOT Demo")

weights_path = st.sidebar.selectbox("Yolo Weights (.pt)", ["models/test.pt","models/best.pt"])
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

def get_yolo_bboxes(results, conf=0.25):
    dets = []
    all_boxes = []
    if hasattr(results, "boxes"):
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy.cpu().numpy().flatten()
            score = float(box.conf)
            cls = int(box.cls)
            if score < conf:
                continue
            all_boxes.append([x1, y1, x2, y2, score, cls])
            if cls == 2:
                dets.append([x1, y1, x2, y2, score])
    dets = np.array(dets, dtype=np.float32)
    if dets.ndim != 2 or dets.shape[1] != 5:
        dets = np.empty((0, 5), dtype=np.float32)
    all_boxes = np.array(all_boxes, dtype=np.float32)
    if all_boxes.ndim != 2 or all_boxes.shape[1] != 6:
        all_boxes = np.empty((0, 6), dtype=np.float32)
    return dets, all_boxes

def draw_tracks_and_dets(img, tracks, dets_all, class_names=None):
    img = img.copy()
    for row in dets_all:
        x1, y1, x2, y2, conf, cls = row
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls = int(cls)
        label = f"{class_names[cls] if class_names else cls}:{conf:.2f}"
        color = (0, 180, 255) if cls != 0 else (0, 120, 255)  # orange for others, blue-ish for person
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    for row in tracks:
        x1, y1, x2, y2, tid, conf = row
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        tid = int(tid)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"ID:{tid}", (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return img

def run_stream(source, model, conf):
    class_names = model.model.names if hasattr(model.model, 'names') else [str(i) for i in range(80)]
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

        dets, dets_all = get_yolo_bboxes(yolo_results, conf=conf)
        tracks = tracker.track(frame, dets)  # (M,6) [x1,y1,x2,y2,id,conf]
        frame_out = draw_tracks_and_dets(frame, tracks, dets_all, class_names=class_names)
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
