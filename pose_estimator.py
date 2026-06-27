from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

NUM_KEYPOINTS = 17

SKELETON_CONNECTIONS = [

    (0, 1), (0, 2), (1, 3), (2, 4),

    (5, 6), (5, 11), (6, 12), (11, 12),

    (5, 7), (7, 9),

    (6, 8), (8, 10),

    (11, 13), (13, 15),

    (12, 14), (14, 16),

    (0, 5), (0, 6),

    (3, 5),
]

KP_COLORS = {
    "head": (255, 200, 0),
    "torso": (0, 255, 0),
    "left_arm": (0, 200, 255),
    "right_arm": (255, 100, 100),
    "left_leg": (255, 0, 200),
    "right_leg": (200, 0, 255),
}

_CONNECTION_GROUP = {
    (0, 1): "head", (0, 2): "head", (1, 3): "head", (2, 4): "head",
    (5, 6): "torso", (5, 11): "torso", (6, 12): "torso", (11, 12): "torso",
    (5, 7): "left_arm", (7, 9): "left_arm",
    (6, 8): "right_arm", (8, 10): "right_arm",
    (11, 13): "left_leg", (13, 15): "left_leg",
    (12, 14): "right_leg", (14, 16): "right_leg",
    (0, 5): "head", (0, 6): "head", (3, 5): "head",
}

@dataclass
class PoseDetection:
    bbox: Tuple[float, float, float, float]
    keypoints: np.ndarray
    confidence: float

    def visible_keypoints_count(self, conf_thresh: float = 0.5) -> int:
        return int((self.keypoints[:, 2] >= conf_thresh).sum())

    def get_keypoint(self, idx: int,
                      conf_thresh: float = 0.5
                      ) -> Optional[Tuple[float, float]]:
        if self.keypoints[idx, 2] < conf_thresh:
            return None
        return (float(self.keypoints[idx, 0]),
                float(self.keypoints[idx, 1]))

class PoseEstimator:

    def __init__(self, model_path: str = "yolo11l-pose.pt",
                 device: Optional[str] = None,
                 conf: float = 0.25,
                 iou: float = 0.5,
                 imgsz: int = 640,
                 verbose: bool = False):

        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.verbose = verbose

        print(f"[PoseEstimator] Загружена модель: {model_path}")

    def predict_frame(self, frame: np.ndarray) -> List[PoseDetection]:
        kwargs = dict(
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=self.verbose,
            classes=[0],

        )
        if self.device is not None:
            kwargs["device"] = self.device

        results = self.model.predict(frame, **kwargs)

        if not results or len(results) == 0:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        if result.keypoints is None:
            return []

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        boxes_conf = result.boxes.conf.cpu().numpy()

        keypoints_data = result.keypoints.data.cpu().numpy()

        detections: List[PoseDetection] = []
        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = boxes_xyxy[i]
            detections.append(PoseDetection(
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                keypoints=keypoints_data[i].astype(np.float32),
                confidence=float(boxes_conf[i]),
            ))

        return detections

def _bbox_iou(box_a: Tuple[float, float, float, float],
              box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0

    return inter / union

def match_poses_to_tracks(
    tracked_objects: np.ndarray,
    poses: List[PoseDetection],
    iou_thresh: float = 0.4,
) -> Dict[int, Optional[PoseDetection]]:
    result: Dict[int, Optional[PoseDetection]] = {}

    if len(tracked_objects) == 0:
        return result

    track_ids = [int(obj[4]) for obj in tracked_objects]
    track_boxes = [
        (float(obj[0]), float(obj[1]), float(obj[2]), float(obj[3]))
        for obj in tracked_objects
    ]

    if not poses:
        return {tid: None for tid in track_ids}

    n_tracks = len(track_boxes)
    n_poses = len(poses)
    iou_matrix = np.zeros((n_tracks, n_poses), dtype=np.float32)
    for i, tb in enumerate(track_boxes):
        for j, p in enumerate(poses):
            iou_matrix[i, j] = _bbox_iou(tb, p.bbox)

    cost = 1.0 - iou_matrix
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)
    except ImportError:

        pairs = []
        for i in range(n_tracks):
            for j in range(n_poses):
                pairs.append((iou_matrix[i, j], i, j))
        pairs.sort(reverse=True)
        used_tracks = set()
        used_poses = set()
        row_ind, col_ind = [], []
        for iou_val, i, j in pairs:
            if i in used_tracks or j in used_poses:
                continue
            if iou_val < iou_thresh:
                break
            row_ind.append(i)
            col_ind.append(j)
            used_tracks.add(i)
            used_poses.add(j)
        row_ind = np.array(row_ind)
        col_ind = np.array(col_ind)

    matched_tracks = set()
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_thresh:
            result[track_ids[r]] = poses[c]
            matched_tracks.add(r)

    for i, tid in enumerate(track_ids):
        if i not in matched_tracks:
            result[tid] = None

    return result

def draw_skeleton(frame: np.ndarray,
                  pose: PoseDetection,
                  color: Optional[Tuple[int, int, int]] = None,
                  conf_thresh: float = 0.5,
                  line_thickness: int = 2,
                  point_radius: int = 3,
                  use_part_colors: bool = True) -> np.ndarray:
    if pose is None or pose.keypoints is None:
        return frame

    kpts = pose.keypoints

    for (a, b) in SKELETON_CONNECTIONS:
        if kpts[a, 2] < conf_thresh or kpts[b, 2] < conf_thresh:
            continue

        line_color = color
        if line_color is None:
            if use_part_colors:
                group = _CONNECTION_GROUP.get((a, b), "torso")
                line_color = KP_COLORS[group]
            else:
                line_color = (0, 255, 0)

        pa = (int(kpts[a, 0]), int(kpts[a, 1]))
        pb = (int(kpts[b, 0]), int(kpts[b, 1]))
        cv2.line(frame, pa, pb, line_color, line_thickness, cv2.LINE_AA)

    for i in range(NUM_KEYPOINTS):
        if kpts[i, 2] < conf_thresh:
            continue

        pt = (int(kpts[i, 0]), int(kpts[i, 1]))

        cv2.circle(frame, pt, point_radius + 1, (0, 0, 0), -1)

        pt_color = color if color is not None else (255, 255, 255)
        cv2.circle(frame, pt, point_radius, pt_color, -1)

    return frame

def draw_keypoints_only(frame: np.ndarray,
                        pose: PoseDetection,
                        color: Optional[Tuple[int, int, int]] = None,
                        conf_thresh: float = 0.5,
                        point_radius: int = 3) -> np.ndarray:
    if pose is None or pose.keypoints is None:
        return frame
    kpts = pose.keypoints
    pt_color = color if color is not None else (0, 255, 255)

    for i in range(NUM_KEYPOINTS):
        if kpts[i, 2] < conf_thresh:
            continue
        pt = (int(kpts[i, 0]), int(kpts[i, 1]))
        cv2.circle(frame, pt, point_radius + 1, (0, 0, 0), -1)
        cv2.circle(frame, pt, point_radius, pt_color, -1)
    return frame

class PoseSmoother:

    def __init__(self,
                 alpha: float = 0.5,
                 min_conf: float = 0.3,
                 max_inactive_frames: int = 60):
        self.alpha = float(alpha)
        self.min_conf = float(min_conf)
        self.max_inactive_frames = max_inactive_frames

        self._state: Dict[int, np.ndarray] = {}

        self._last_seen: Dict[int, int] = {}

    def smooth(self, frame_num: int,
               track_poses: Dict[int, Optional["PoseDetection"]]
               ) -> Dict[int, Optional["PoseDetection"]]:
        smoothed: Dict[int, Optional[PoseDetection]] = {}

        for tid, pose in track_poses.items():
            if pose is None:
                smoothed[tid] = None
                continue
            smoothed[tid] = self._smooth_pose(tid, pose, frame_num)

        self._cleanup(frame_num)

        return smoothed

    def _smooth_pose(self, tid: int, pose: "PoseDetection",
                     frame_num: int) -> "PoseDetection":
        kp_in = pose.keypoints
        kp_out = kp_in.copy()

        prev_xy = self._state.get(tid)

        if prev_xy is None:
            prev_xy = np.full((NUM_KEYPOINTS, 2), np.nan, dtype=np.float32)

        new_state = prev_xy.copy()

        for i in range(NUM_KEYPOINTS):
            x_cur, y_cur, c_cur = kp_in[i]
            if c_cur < self.min_conf:

                if not np.isnan(prev_xy[i, 0]):
                    kp_out[i, 0] = prev_xy[i, 0]
                    kp_out[i, 1] = prev_xy[i, 1]

                continue

            if np.isnan(prev_xy[i, 0]):

                new_state[i, 0] = x_cur
                new_state[i, 1] = y_cur
            else:
                new_state[i, 0] = self.alpha * x_cur + \
                                  (1 - self.alpha) * prev_xy[i, 0]
                new_state[i, 1] = self.alpha * y_cur + \
                                  (1 - self.alpha) * prev_xy[i, 1]

            kp_out[i, 0] = new_state[i, 0]
            kp_out[i, 1] = new_state[i, 1]

        self._state[tid] = new_state
        self._last_seen[tid] = frame_num

        return PoseDetection(
            bbox=pose.bbox,
            keypoints=kp_out,
            confidence=pose.confidence,
        )

    def _cleanup(self, frame_num: int) -> None:
        to_drop = [
            tid for tid, last in self._last_seen.items()
            if frame_num - last > self.max_inactive_frames
        ]
        for tid in to_drop:
            del self._state[tid]
            del self._last_seen[tid]

    def reset(self) -> None:
        self._state.clear()
        self._last_seen.clear()
