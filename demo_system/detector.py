from ultralytics import YOLO

class Detector:
    def __init__( self, weights_path: str, conf: int=0.25, device: str="cpu" ):
        self.model = YOLO(weights_path).to(device)
        self.conf = conf
    def __call__( self, frame ):
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        bboxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        return bboxes, scores
    def __str__( self ):
        return f"Detector: {self.model.model.yaml['nc']} classes, {self.conf} confidence"