import numpy as np
from tracker.algorithms.boost import BoostTrack

class Tracker:
    def __init__(self, name="sequence"):
        self.name = name
        self.frame_id = 1
        self.tracker = BoostTrack(video_name=self.name)
    
    def reset(self, name=None):
        if name:
            self.name = name
        self.frame_id = 1
        self.tracker = BoostTrack(video_name=self.name)

    def track(self, img, dets):
        if not isinstance(img, np.ndarray) or img.ndim != 3:
            raise ValueError("img 必須是 (H, W, 3) 的 numpy 圖片。")
        if not isinstance(dets, np.ndarray) or dets.shape[1] != 5:
            raise ValueError("dets 必須是 (N, 5) 的 numpy 陣列。")
        
        img_tensor = np.zeros((1, 3, img.shape[0], img.shape[1]), dtype=np.float32)
        tag = f"{self.name}:{self.frame_id}"
        result = self.tracker.update(dets, img_tensor, img, tag)
        self.frame_id += 1
        return result
