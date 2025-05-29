# tracker/algorithms/ecc.py

import numpy as np
import cv2
from copy import deepcopy
from typing import Optional, Dict
def ecc(src, dst, warp_mode=cv2.MOTION_EUCLIDEAN, eps=1e-5, max_iter=100, scale=0.15):
    assert src.shape == dst.shape
    if src.ndim == 3:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    if scale != 1:
        src_r = cv2.resize(src, (0, 0), fx=scale, fy=scale)
        dst_r = cv2.resize(dst, (0, 0), fx=scale, fy=scale)
    else:
        src_r, dst_r = src, dst
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
    (cc, warp_matrix) = cv2.findTransformECC(src_r, dst_r, warp_matrix, warp_mode, criteria, None, 1)
    if scale != 1:
        warp_matrix[0, 2] /= scale
        warp_matrix[1, 2] /= scale
    return warp_matrix

class ECC:
    def __init__(self, warp_mode=cv2.MOTION_EUCLIDEAN, eps=1e-4,
                 max_iter=100, scale=0.15, align=False,
                 video_name: Optional[str] = None, use_cache: bool = True):
        self.warp_mode = warp_mode
        self.eps = eps
        self.max_iter = max_iter
        self.scale = scale
        self.align = align
        self.prev_image: Optional[np.ndarray] = None
        self.video_name = video_name
        self.use_cache = use_cache
        self.cache: Dict[str, np.ndarray] = dict()
    def __call__(self, np_image, frame_id, video=""):
        if frame_id == 1:
            self.prev_image = deepcopy(np_image)
            return np.eye(3, dtype=float)
        result = ecc(self.prev_image, np_image, self.warp_mode, self.eps, self.max_iter, self.scale)
        self.prev_image = deepcopy(np_image)
        if result.shape == (2, 3):
            result = np.vstack((result, np.array([[0, 0, 1]], dtype=float)))
        return result
