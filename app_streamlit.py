import streamlit as st
import cv2
import numpy as np
import torch
import time
import tempfile
from ultralytics import YOLO
from tracker.tracker import Tracker
from utils import load_model, extract_yolo_bboxes, draw_tracks_and_detections

st.sidebar.title( "Fire MOT Demo" )

yolo_weights_path = st.sidebar.selectbox(
    "Yolo Weights (.pt)", 
    [
        "models/test.py",
        "models/v4.pt"
    ]
)

source_type = st.sidebar.selectbox( 
    "Video Source",
    [
        "Webcam",
        "Video File"
    ]
)

conf_threshold = st.sidebar.slider( "Confidence Threshold", 0.1, 1.0, 0.25, 0.05 )

video_file = None
if source_type == "Video File":
    video_file = st.sidebar.file_uploader( "Upload a video file", type=["mp4", "avi", "mov"] )

start_btn = st.sidebar.button( "Start" )

def run_video_stream( source, model, conf ):
    class_names = model.model.names if hasattr(model.model, "names") else [str(i) for i in range(80)]
    cap = cv2.VideoCapture(source)
    fps = 0
    prev = time.time()
    frame_show = st.empty()
    tracker = Tracker( name="stream" )
    tracker.reset( name="stream" )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning( "No more frames to read" )
            break
    
        with torch.no_grad():
            yolo_results = model( frame, conf=conf, verbose=False )[0]

        fire_dets, all_dets = extract_yolo_bboxes( yolo_results, conf=conf )
        tracks = tracker.track( frame, fire_dets )
        frame_out = draw_tracks_and_detections( frame, tracks, all_dets, class_names=class_names )
        fps = fps * 0.9 + 0.1 * (1 / (time.time() - prev + 1e-8))
        prev = time.time()
        cv2.putText( frame_out, f"FPS {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, (50, 255, 50), 2 )
        frame_show.image( frame_out, channels="RGR" )
    cap.release()

if __name__ == "__main__":
    if start_btn:
        model = load_model( yolo_weights_path )
        if source_type == "Webcam":
            run_video_stream( 0, model, conf_threshold )
        else:
            if video_file is not None:
                tfile = tempfile.NamedTemporaryFile( delete = False )
                tfile.write( video_file.read() )
                tfile.flush()
                run_video_stream( tfile.name, model, conf_threshold )
            else:
                st.sidebar.warning( "Please upload a video file before starting." )