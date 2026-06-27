from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from interaction_detector import FoulSignal, FoulEvent
from pose_estimator import PoseDetection, NUM_KEYPOINTS

FOUL_COLORS = {

    "slide_tackle":         (0, 50, 255),
    "slide_tackle_delayed": (50, 80, 200),

    "arm_strike":           (100, 0, 255),
    "arm_strike_delayed":   (130, 50, 200),

    "high_kick":     (0, 100, 255),
    "elbow_to_head": (0, 0, 255),
    "head_butt":     (50, 0, 220),
}

ATTACKER_COLOR = (0, 0, 255)
VICTIM_COLOR = (0, 150, 255)

def _draw_dashed_line(img, p1, p2, color, thickness=2, gap=10):
    x1, y1 = p1
    x2, y2 = p2
    dist = int(np.hypot(x2 - x1, y2 - y1))
    if dist < 1:
        return
    n_seg = max(1, dist // gap)
    for i in range(0, n_seg, 2):
        t1 = i / n_seg
        t2 = min((i + 1) / n_seg, 1.0)
        s = (int(x1 + (x2 - x1) * t1), int(y1 + (y2 - y1) * t1))
        e = (int(x1 + (x2 - x1) * t2), int(y1 + (y2 - y1) * t2))
        cv2.line(img, s, e, color, thickness, cv2.LINE_AA)

def draw_foul_visualization(
    frame: np.ndarray,
    signal: FoulSignal,
    pose_attacker: Optional[PoseDetection],
    pose_victim: Optional[PoseDetection],
) -> np.ndarray:
    color_foul = FOUL_COLORS.get(signal.kind, (0, 0, 255))

    if pose_attacker is not None:
        ax1, ay1, ax2, ay2 = map(int, pose_attacker.bbox)
        cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), ATTACKER_COLOR, 4)
    if pose_victim is not None:
        vx1, vy1, vx2, vy2 = map(int, pose_victim.bbox)
        cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), VICTIM_COLOR, 3)

    if pose_attacker is not None:
        for kp_idx in signal.attacker_kpts:
            if kp_idx >= NUM_KEYPOINTS:
                continue
            x, y, c = pose_attacker.keypoints[kp_idx]
            if c < 0.2:
                continue
            cv2.circle(frame, (int(x), int(y)), 12, (0, 0, 0), -1)
            cv2.circle(frame, (int(x), int(y)), 9, (0, 0, 255), -1)
            cv2.circle(frame, (int(x), int(y)), 9, (255, 255, 255), 1)

    if pose_victim is not None:
        for kp_idx in signal.victim_kpts:
            if kp_idx >= NUM_KEYPOINTS:
                continue
            x, y, c = pose_victim.keypoints[kp_idx]
            if c < 0.2:
                continue
            cv2.circle(frame, (int(x), int(y)), 12, (0, 0, 0), -1)
            cv2.circle(frame, (int(x), int(y)), 9, (0, 200, 255), -1)
            cv2.circle(frame, (int(x), int(y)), 9, (255, 255, 255), 1)

    if pose_attacker is not None and pose_victim is not None:
        for (a_idx, v_idx) in signal.cross_lines:
            if a_idx >= NUM_KEYPOINTS or v_idx >= NUM_KEYPOINTS:
                continue
            ax, ay, ac = pose_attacker.keypoints[a_idx]
            vx, vy, vc = pose_victim.keypoints[v_idx]
            if ac < 0.2 or vc < 0.2:
                continue
            _draw_dashed_line(frame, (int(ax), int(ay)),
                               (int(vx), int(vy)), color_foul,
                               thickness=3, gap=8)

    if pose_attacker is not None:
        ax1, ay1, ax2, ay2 = map(int, pose_attacker.bbox)
        label = f"{signal.kind.upper()}  {signal.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       0.65, 2)
        label_y = max(th + 10, ay1 - 8)
        cv2.rectangle(frame, (ax1 - 2, label_y - th - 6),
                       (ax1 + tw + 8, label_y + 4),
                       color_foul, -1)
        cv2.putText(frame, label, (ax1 + 3, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)

    return frame

def draw_event_panel(frame: np.ndarray,
                     active_events: List[FoulEvent],
                     all_events: List[FoulEvent]) -> np.ndarray:
    h, w = frame.shape[:2]
    panel_w = 380
    panel_h = 240
    x0 = 10
    y0 = h - panel_h - 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                   (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h),
                   (200, 200, 200), 1)
    cv2.putText(frame, "FOUL EVENTS",
                (x0 + 10, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)

    cv2.putText(frame, f"ACTIVE: {len(active_events)}",
                (x0 + 10, y0 + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
    y = y0 + 68
    for ev in active_events[:4]:
        color = FOUL_COLORS.get(ev.kind, (200, 200, 200))
        text = (f"{ev.kind}: A{ev.attacker_id}->V{ev.victim_id} "
                f"peak={ev.peak_score:.2f}")
        cv2.putText(frame, text, (x0 + 15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        y += 18

    cv2.putText(frame, f"RECENT: {len(all_events)} total",
                (x0 + 10, y0 + 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y = y0 + 170
    for ev in all_events[-4:]:
        color = FOUL_COLORS.get(ev.kind, (150, 150, 150))
        text = (f"f{ev.start_frame}-{ev.end_frame}: "
                f"{ev.kind} A{ev.attacker_id}/V{ev.victim_id}")
        cv2.putText(frame, text, (x0 + 15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y += 16

    return frame
