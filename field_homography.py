from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np

class FieldModel:
    LENGTH = 105.0
    WIDTH = 68.0

    HALF_L = LENGTH / 2
    HALF_W = WIDTH / 2

    PENALTY_DEPTH = 16.5
    PENALTY_HALF_W = 40.32 / 2

    GOAL_AREA_DEPTH = 5.5
    GOAL_AREA_HALF_W = 18.32 / 2

    CENTER_CIRCLE_RADIUS = 9.15

    PENALTY_SPOT_DIST = 11.0

    @classmethod
    def keypoints(cls) -> Dict[str, Tuple[float, float]]:
        L, W = cls.HALF_L, cls.HALF_W
        pd, pw = cls.PENALTY_DEPTH, cls.PENALTY_HALF_W
        return {

            "corner_TL": (-L, -W),
            "corner_TR": (+L, -W),
            "corner_BL": (-L, +W),
            "corner_BR": (+L, +W),

            "center": (0.0, 0.0),

            "halfway_top": (0.0, -W),
            "halfway_bot": (0.0, +W),

            "penalty_L_TL": (-L, -pw),
            "penalty_L_TR": (-L + pd, -pw),
            "penalty_L_BL": (-L, +pw),
            "penalty_L_BR": (-L + pd, +pw),
            "penalty_R_TL": (+L - pd, -pw),
            "penalty_R_TR": (+L, -pw),
            "penalty_R_BL": (+L - pd, +pw),
            "penalty_R_BR": (+L, +pw),

            "penalty_spot_L": (-L + cls.PENALTY_SPOT_DIST, 0.0),
            "penalty_spot_R": (+L - cls.PENALTY_SPOT_DIST, 0.0),
        }

    @classmethod
    def is_inside(cls, X: float, Y: float, margin: float = 5.0) -> bool:
        return (-cls.HALF_L - margin <= X <= cls.HALF_L + margin and
                -cls.HALF_W - margin <= Y <= cls.HALF_W + margin)

@dataclass
class HomographyResult:
    matrix: np.ndarray
    matrix_inv: np.ndarray
    confidence: float
    frame_num: int
    method: str = "unknown"
    n_points_used: int = 0

    @classmethod
    def from_image_to_world(cls, H: np.ndarray, frame_num: int,
                            confidence: float = 1.0,
                            method: str = "unknown",
                            n_points_used: int = 0) -> "HomographyResult":
        H = np.asarray(H, dtype=np.float64)
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            raise ValueError("Singular homography matrix")
        return cls(matrix=H, matrix_inv=H_inv, confidence=confidence,
                   frame_num=frame_num, method=method,
                   n_points_used=n_points_used)

    def project_to_world(self, points_xy: np.ndarray) -> np.ndarray:
        return _apply_homography(self.matrix, points_xy)

    def project_to_image(self, points_XY: np.ndarray) -> np.ndarray:
        return _apply_homography(self.matrix_inv, points_XY)

    def is_valid(self) -> bool:
        try:
            corners_world = np.array([
                [-FieldModel.HALF_L, -FieldModel.HALF_W],
                [+FieldModel.HALF_L, -FieldModel.HALF_W],
                [+FieldModel.HALF_L, +FieldModel.HALF_W],
                [-FieldModel.HALF_L, +FieldModel.HALF_W],
            ])
            corners_img = self.project_to_image(corners_world)

            if not np.isfinite(corners_img).all():
                return False

            if np.abs(corners_img).max() > 50000:
                return False

            area = _polygon_area(corners_img)
            if area < 100:
                return False

            test_world = np.array([[0.0, 0.0]])
            test_img = self.project_to_image(test_world)
            if not np.isfinite(test_img).all():
                return False
            test_back = self.project_to_world(test_img)
            err = np.linalg.norm(test_world - test_back)
            if err > 1.0:
                return False

            return True
        except Exception:
            return False

def _apply_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = len(points)
    homo = np.hstack([points, np.ones((n, 1))])
    proj = (H @ homo.T).T

    w = proj[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return proj[:, :2] / w

def _polygon_area(pts: np.ndarray) -> float:
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def bbox_to_world_position(
    bbox_xyxy,
    homography_result,
    off_pitch_margin: float = 8.0,
) -> Optional[Tuple[float, float]]:
    if homography_result is None:
        return None

    x1, y1, x2, y2 = float(bbox_xyxy[0]), float(bbox_xyxy[1]), \
                     float(bbox_xyxy[2]), float(bbox_xyxy[3])

    ground = np.array([[(x1 + x2) / 2.0, y2]], dtype=np.float32)

    try:
        world = homography_result.project_to_world(ground)
    except Exception:
        return None

    Xw, Yw = float(world[0, 0]), float(world[0, 1])

    if not (np.isfinite(Xw) and np.isfinite(Yw)):
        return None

    if not FieldModel.is_inside(Xw, Yw, margin=off_pitch_margin):
        return None

    return (Xw, Yw)

def world_distance_meters(
    pos_a: Optional[Tuple[float, float]],
    pos_b: Optional[Tuple[float, float]],
) -> Optional[float]:
    if pos_a is None or pos_b is None:
        return None
    dx = pos_a[0] - pos_b[0]
    dy = pos_a[1] - pos_b[1]
    return float(np.hypot(dx, dy))

class FieldHomographyEstimator(ABC):

    @abstractmethod
    def estimate(self, frame: np.ndarray,
                 frame_num: int) -> Optional[HomographyResult]:
        ...

class HomographyTracker:

    def __init__(self,
                 estimator: FieldHomographyEstimator,
                 keyframe_interval: int = 30,
                 smooth_window: int = 5,
                 max_extrapolation: int = 60):
        self.estimator = estimator
        self.keyframe_interval = max(1, keyframe_interval)
        self.smooth_window = max(1, smooth_window)
        self.max_extrapolation = max_extrapolation

        self._history: deque[HomographyResult] = deque(
            maxlen=max(self.smooth_window, 10)
        )

        self._cache: Dict[int, HomographyResult] = {}

        self._last_valid: Optional[HomographyResult] = None

        self.stats = {
            "total_calls": 0,
            "keyframes_computed": 0,
            "keyframes_failed": 0,
            "interpolated": 0,
            "extrapolated": 0,
            "no_homography": 0,
        }

    def get(self, frame: np.ndarray,
            frame_num: int) -> Optional[HomographyResult]:
        self.stats["total_calls"] += 1

        if frame_num in self._cache:
            return self._cache[frame_num]

        is_keyframe = (frame_num % self.keyframe_interval) == 0

        if is_keyframe:
            result = self._compute_keyframe(frame, frame_num)
            if result is not None:
                self._cache[frame_num] = result
                return result

        return self._fallback(frame_num)

    def _compute_keyframe(self, frame: np.ndarray,
                           frame_num: int) -> Optional[HomographyResult]:
        result = self.estimator.estimate(frame, frame_num)

        if result is None or not result.is_valid():
            self.stats["keyframes_failed"] += 1
            return None

        self.stats["keyframes_computed"] += 1
        self._history.append(result)
        self._last_valid = result

        if self.smooth_window > 1 and len(self._history) >= 2:
            return self._smooth_recent(frame_num)

        return result

    def _smooth_recent(self, frame_num: int) -> HomographyResult:
        recent = list(self._history)[-self.smooth_window:]
        normed = []
        for r in recent:
            H = r.matrix.copy()
            if abs(H[2, 2]) > 1e-9:
                H = H / H[2, 2]
            normed.append(H)
        H_avg = np.mean(normed, axis=0)

        conf_avg = np.mean([r.confidence for r in recent]) * 0.95

        smoothed = HomographyResult.from_image_to_world(
            H_avg, frame_num,
            confidence=conf_avg,
            method=f"{recent[-1].method}+smoothed{len(recent)}",
        )
        if smoothed.is_valid():
            return smoothed

        return recent[-1]

    def _fallback(self, frame_num: int) -> Optional[HomographyResult]:
        if self._last_valid is None:
            self.stats["no_homography"] += 1
            return None

        gap = frame_num - self._last_valid.frame_num
        if gap > self.max_extrapolation:
            self.stats["no_homography"] += 1
            return None

        self.stats["extrapolated"] += 1
        return HomographyResult(
            matrix=self._last_valid.matrix,
            matrix_inv=self._last_valid.matrix_inv,
            confidence=self._last_valid.confidence * (1.0 - gap / 100.0),
            frame_num=frame_num,
            method=f"{self._last_valid.method}+extrap{gap}",
            n_points_used=self._last_valid.n_points_used,
        )

    def get_stats(self) -> Dict:
        return dict(self.stats)

class CachedHomography(FieldHomographyEstimator):

    def __init__(self, cache_path: str,
                 interpolate: bool = True,
                 max_gap_for_interp: int = 90):
        cache_path = Path(cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache not found: {cache_path}")

        data = np.load(str(cache_path), allow_pickle=True)
        self.frames = np.asarray(data["frames"], dtype=np.int64)
        self.matrices = np.asarray(data["matrices"], dtype=np.float64)
        self.confidences = np.asarray(data["confidences"], dtype=np.float32)

        try:
            self.method = str(data["method"])
        except (KeyError, ValueError):
            self.method = "cached"
        try:
            self.source_video = str(data["video_path"])
        except (KeyError, ValueError):
            self.source_video = ""

        order = np.argsort(self.frames)
        self.frames = self.frames[order]
        self.matrices = self.matrices[order]
        self.confidences = self.confidences[order]

        self.interpolate = interpolate
        self.max_gap_for_interp = max_gap_for_interp

        if len(self.frames) == 0:
            raise ValueError(f"Cache {cache_path} is empty")

        print(f"[CachedHomography] Загружено {len(self.frames)} ключевых кадров "
              f"(метод: {self.method})")
        print(f"  Диапазон кадров: {int(self.frames[0])} - "
              f"{int(self.frames[-1])}")
        avg_conf = float(self.confidences.mean())
        print(f"  Средняя уверенность: {avg_conf:.2f}")

    def estimate(self, frame: np.ndarray,
                 frame_num: int) -> Optional[HomographyResult]:

        idx_right = int(np.searchsorted(self.frames, frame_num, side="left"))

        if idx_right < len(self.frames) and \
           int(self.frames[idx_right]) == frame_num:
            H = self.matrices[idx_right]
            try:
                return HomographyResult.from_image_to_world(
                    H=H,
                    frame_num=frame_num,
                    confidence=float(self.confidences[idx_right]),
                    method=self.method,
                )
            except ValueError:
                return None

        if idx_right == 0:

            return self._make_result(0, frame_num, suffix="before_first")
        if idx_right >= len(self.frames):

            return self._make_result(
                len(self.frames) - 1, frame_num, suffix="after_last"
            )

        idx_left = idx_right - 1
        f_left = int(self.frames[idx_left])
        f_right = int(self.frames[idx_right])
        gap = f_right - f_left

        if gap > self.max_gap_for_interp or not self.interpolate:

            if (frame_num - f_left) < (f_right - frame_num):
                return self._make_result(
                    idx_left, frame_num, suffix=f"nearest_left_gap{gap}"
                )
            else:
                return self._make_result(
                    idx_right, frame_num, suffix=f"nearest_right_gap{gap}"
                )

        t = (frame_num - f_left) / max(1, gap)
        H_left = self._normalize_H(self.matrices[idx_left])
        H_right = self._normalize_H(self.matrices[idx_right])
        H_interp = (1 - t) * H_left + t * H_right
        conf = (1 - t) * float(self.confidences[idx_left]) + \
               t * float(self.confidences[idx_right])

        conf *= 0.95

        try:
            return HomographyResult.from_image_to_world(
                H=H_interp,
                frame_num=frame_num,
                confidence=conf,
                method=f"{self.method}+interp",
            )
        except ValueError:
            return None

    def _make_result(self, idx: int, frame_num: int,
                      suffix: str = "") -> Optional[HomographyResult]:
        H = self.matrices[idx]
        method = self.method
        if suffix:
            method = f"{method}+{suffix}"
        try:
            return HomographyResult.from_image_to_world(
                H=H,
                frame_num=frame_num,
                confidence=float(self.confidences[idx]),
                method=method,
            )
        except ValueError:
            return None

    @staticmethod
    def _normalize_H(H: np.ndarray) -> np.ndarray:
        H = np.asarray(H, dtype=np.float64)
        if abs(H[2, 2]) > 1e-9:
            return H / H[2, 2]
        return H
