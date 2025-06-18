import numpy as np
import cv2
from ultralytics import YOLO

PERSON_CLASS_ID = 2

def load_model( w_path: str ):
    device = "cpu"
    model = YOLO( w_path ).to( device )
    return model

def extract_yolo_bboxes( results, conf: float = 0.25 ):
    person_dets = []
    all_boxes = []

    if hasattr( results, "boxes" ):
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy.cpu().numpy().flatten()
            score = float( box.conf )
            cls = int( box.cls )
            if score < conf:
                continue
            all_boxes.append( [x1, y1, x2, y2, score, cls] ) #FIXME
            if cls == PERSON_CLASS_ID:
                person_dets.append( [x1, y1, x2, y2, score] ) #FIXME

    person_dets = np.array( person_dets, dtype=np.float32 )
    if person_dets.ndim != 2 or person_dets.shape[1] != 5:
        person_dets = np.empty( (0, 5), dtype=np.float32 )
    all_boxes = np.array( all_boxes, dtype=np.float32 )
    if all_boxes.ndim != 2 or all_boxes.shape[1] != 6:
        all_boxes = np.empty( (0, 6), dtype=np.float32 )
    return person_dets, all_boxes

def draw_tracks_and_detections( img, tracks, dets_all, class_names=None ):
    img = img.copy()
    for row in dets_all:
        x1, y1, x2, y2, conf, cls = row
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls = int(cls)
        label = f"{class_names[cls] if class_names else cls}:{conf:.2f}"
        color = (0, 180, 0) if cls == PERSON_CLASS_ID else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    for row in tracks:
        x1, y1, x2, y2, tid, conf = row
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        tid = int(tid)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"ID:{tid}", (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return img