import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from utils import iou_batch, mahalanobis_batch

class Track:
    _next_id = 0
    def __init__( self, bbox, score ):
        self.kf = self._init_kf( bbox )
        self.time_since_update = 0
        self.hits = 1
        self.id = Track._next_id; Track._next_id += 1
        self.last_score = score
        self.age = 1
    
    @staticmethod
    def _init_kf( bbox ):
        kf = KalmanFilter( dim_x=8, dim_z=4 )
        kf.F = np.block( [np.eye(4),np.eye(4)], [np.zeros(4),np.eye(4)] )
        kf.H = np.hstack( [np.eye(4), np.zeros(4)] )
        
        x1, x2, y1, y2 = bbox
        u = (x1+x2)/2 # Coordinate of the object center x
        v = (y1+y2)/2 # Coordinate of the object center y
        h = y2 - y1 # Height of the object
        r = (x2-x1)/(y2-y1) # Ratio of the object’s width
        # kf.P *= 10; kf.R *= 1; kf.Q *= 0.01 # FIXME
        kf.x[:4,0] = [u, v, h, r]
        return kf

    def predict( self ):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self
    
    def update( self, bbox, score ):
        x1, y1, x2, y2 = bbox
        x1, x2, y1, y2 = bbox
        self.kf.update( [(x1+x2)/2, (y1+y2)/2, y2 - y1, (x2-x1)/(y2-y1)] )
        self.time_since_update = 0
        self.last_score = score
        self.hits += 1
    
    @property
    def state( self ):
        u, v, h, r = self.kf.x[:4,0]
        return np.array([u, v, h, r])

class BoostTrack:
    def __init__(self, iou_thr=0.3, lambda_iou=.5, lambda_mhd=.25,
                 lambda_shape=.25, boost_thr=0.1):
        self.tracks = []
        self.iou_thr = iou_thr
        self.l_iou, self.l_mhd, self.l_shape = lambda_iou, lambda_mhd, lambda_shape
        self.boost_thr = boost_thr  # 分數低於此值就嘗試 boost

    # ---------- 2-1 偵測信心分數 Boost ----------
    def _boost_scores(self, dets, scores):
        if len(self.tracks) == 0: return scores
        trks = np.array([t.state for t in self.tracks])
        ious = iou_batch(dets, trks)
        max_iou = ious.max(1)
        boosted = np.maximum(scores, max_iou)  # 式(4)
        scores[scores < self.boost_thr] = boosted[scores < self.boost_thr]
        return scores

    # ---------- 2-2 建立多元相似度矩陣 ----------
    def _build_cost(self, dets, scores):
        trks = np.array([t.state for t in self.tracks])
        iou = iou_batch(dets, trks)
        conf = scores.reshape(-1,1) * np.array([t.last_score for t in self.tracks])
        cost = iou + self.l_iou * conf * iou       # IoU 與信心
        mhd = mahalanobis_batch(dets, trks)
        cost += self.l_mhd * mhd                   # Mahalanobis
        # Shape 相似度
        dw = (dets[:,2]-dets[:,0]).reshape(-1,1); dh = (dets[:,3]-dets[:,1]).reshape(-1,1)
        tw = (trks[:,2]-trks[:,0]).reshape(1,-1); th = (trks[:,3]-trks[:,1]).reshape(1,-1)
        shape_sim = np.exp(-(np.abs(dw-tw)/np.maximum(dw,tw)+
                             np.abs(dh-th)/np.maximum(dw,tw)))
        cost += self.l_shape * conf * shape_sim
        return cost, iou

    # ---------- 2-3 一次性匈牙利配對 ----------
    def update(self, detections, scores):
        if len(detections)==0: 
            detections=np.empty((0,4))
            scores=np.empty((0,))
        scores = self._boost_scores(detections, scores)
        # predict step
        for t in self.tracks: t.predict()

        if len(self.tracks)==0 or len(detections)==0:
            unmatched_det = list(range(len(detections)))
            matches = []
        else:
            cost, iou = self._build_cost(detections, scores)
            row_ind, col_ind = linear_sum_assignment(-cost)
            matches, unmatched_det = [], []
            for r,c in zip(row_ind, col_ind):
                if iou[r,c] < self.iou_thr: unmatched_det.append(r)
                else: matches.append((r,c))
            unmatched_det += [d for d in range(len(detections)) if d not in row_ind]

        # 更新 matched tracks
        for d,t_idx in matches:
            self.tracks[t_idx].update(detections[d], scores[d])

        # 建立新軌跡
        for d in unmatched_det:
            self.tracks.append(Track(detections[d], scores[d]))

        # 刪除失蹤太久的軌跡
        self.tracks = [t for t in self.tracks if t.time_since_update < 30]
        return [(t.id, *t.state) for t in self.tracks]

