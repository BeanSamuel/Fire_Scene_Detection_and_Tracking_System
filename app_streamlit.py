import streamlit as st
import cv2, numpy as np, torch, time, tempfile
from ultralytics import YOLO

st.sidebar.title( "Fire MOT Demo" )
weights_path = st.sidebar.selectbox( "Yolov5n Weights (.pt)", ["models/test.pt"] )
source_type = st.sidebar.selectbox( "Video Source", ["Webcam", "Video File"] )
confidence   = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
start_btn = st.sidebar.button("Start")

def load_model(w_path):
    device = 0 if torch.cuda.is_available() else "cpu"
    model  = YOLO(w_path).to(device)
    return model

def run_stream(source, model, conf ):
    cap = cv2.VideoCapture( source )
    fps = 0
    prev = time.time()

    frame_show = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning( "No more frames to read." )
            break
        # yolov5n
        with torch.no_grad():
            results = model(frame, conf=conf, verbose=False)[0]

        annotated = results.plot() 

        # compare iuo mpa rmpa ... 
        # boosttrack
        # compare ... 
        fps = fps*0.9 + 0.1*(1/(time.time()-prev))
        prev = time.time()
        cv2.putText( annotated, f"FPS {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50,255,50), 2 )
        frame_show.image( annotated, channels="BGR" )
    cap.release()


if start_btn:
    model = load_model("models/test.pt")
    if source_type == "Webcam":
        run_stream(0, model, confidence)
    else:
        file = st.file_uploader( "Upload a video file", type=["mp4", "avi", "mov"] )
        if file is not None:
            tfile = tempfile.NamedTemporaryFile( delete=False )
            tfile.write( file.read() )
            run_stream(tfile.name, model, confidence)
        else:
            st.sidebar.warning( "Please upload a video file." )