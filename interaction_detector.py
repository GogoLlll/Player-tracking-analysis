from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

KP_NOSE = 0
KP_L_EYE, KP_R_EYE = 1, 2
KP_L_EAR, KP_R_EAR = 3, 4
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW, KP_R_ELBOW = 7, 8
KP_L_WRIST, KP_R_WRIST = 9, 10
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_KNEE, KP_R_KNEE = 13, 14
KP_L_ANKLE, KP_R_ANKLE = 15, 16

@dataclass
class FoulSignal:
    kind: str
    score: float
    attacker_id: int
    victim_id: int

    attacker_kpts: List[int] = field(default_factory=list)
    victim_kpts: List[int] = field(default_factory=list)

    cross_lines: List[Tuple[int, int]] = field(default_factory=list)

    reason: str = ""

    candidate: Optional[object] = None

@dataclass
class FoulEvent:
    kind: str
    attacker_id: int
    victim_id: int
    start_frame: int
    end_frame: int
    hit_count: int
    avg_score: float
    peak_score: float

    last_signal: Optional[FoulSignal] = None

def _kp(pose, idx: int, conf_thresh: float = 0.3
        ) -> Optional[Tuple[float, float]]:
    if pose is None or pose.keypoints is None:
        return None
    kp = pose.keypoints
    if kp.shape[0] <= idx:
        return None
    if kp[idx, 2] < conf_thresh:
        return None
    return (float(kp[idx, 0]), float(kp[idx, 1]))

def _midpoint(p1: Optional[Tuple[float, float]],
              p2: Optional[Tuple[float, float]]
              ) -> Optional[Tuple[float, float]]:
    if p1 is None or p2 is None:
        return None
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def _dist(p1: Optional[Tuple[float, float]],
          p2: Optional[Tuple[float, float]]) -> float:
    if p1 is None or p2 is None:
        return float("inf")
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

def _body_height(pose) -> Optional[float]:
    if pose is None:
        return None

    nose = _kp(pose, KP_NOSE)
    ank_l = _kp(pose, KP_L_ANKLE)
    ank_r = _kp(pose, KP_R_ANKLE)
    ank = _midpoint(ank_l, ank_r) if (ank_l and ank_r) else (ank_l or ank_r)
    if nose and ank:
        h = abs(ank[1] - nose[1])
        if h > 10:
            return h

    sh = _midpoint(_kp(pose, KP_L_SHOULDER), _kp(pose, KP_R_SHOULDER))
    hp = _midpoint(_kp(pose, KP_L_HIP), _kp(pose, KP_R_HIP))
    if sh and hp:
        h = abs(hp[1] - sh[1]) * 2.5
        if h > 10:
            return h

    if pose.bbox:
        h = pose.bbox[3] - pose.bbox[1]
        if h > 10:
            return float(h)
    return None

def _torso_angle_deg(pose) -> Optional[float]:
    if pose is None:
        return None
    sh = _midpoint(_kp(pose, KP_L_SHOULDER), _kp(pose, KP_R_SHOULDER))
    hp = _midpoint(_kp(pose, KP_L_HIP), _kp(pose, KP_R_HIP))
    if sh is None or hp is None:
        return None
    dx = hp[0] - sh[0]
    dy = hp[1] - sh[1]
    if abs(dy) < 1e-3 and abs(dx) < 1e-3:
        return None

    return float(np.degrees(np.arctan2(abs(dx), abs(dy))))

def _bbox_distance(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(0, max(ax1, bx1) - min(ax2, bx2))
    dy = max(0, max(ay1, by1) - min(ay2, by2))
    return float(np.hypot(dx, dy))

def _point_segment_distance(p, a, b) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return float(np.hypot(px - ax, py - ay))
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    qx, qy = ax + t * dx, ay + t * dy
    return float(np.hypot(px - qx, py - qy))

def _segment_segment_distance(p1, p2, p3, p4) -> float:

    def _ccw(a, b, c):
        return ((b[0] - a[0]) * (c[1] - a[1]) -
                (b[1] - a[1]) * (c[0] - a[0]))

    d1 = _ccw(p3, p4, p1)
    d2 = _ccw(p3, p4, p2)
    d3 = _ccw(p1, p2, p3)
    d4 = _ccw(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return 0.0

    return min(
        _point_segment_distance(p1, p3, p4),
        _point_segment_distance(p2, p3, p4),
        _point_segment_distance(p3, p1, p2),
        _point_segment_distance(p4, p1, p2),
    )

ATTACKER_LEG_BONES = [
    (KP_L_HIP, KP_L_KNEE),
    (KP_L_KNEE, KP_L_ANKLE),
    (KP_R_HIP, KP_R_KNEE),
    (KP_R_KNEE, KP_R_ANKLE),
]

VICTIM_ALL_BONES = [

    (KP_L_HIP, KP_L_KNEE),
    (KP_L_KNEE, KP_L_ANKLE),
    (KP_R_HIP, KP_R_KNEE),
    (KP_R_KNEE, KP_R_ANKLE),

    (KP_L_SHOULDER, KP_L_HIP),
    (KP_R_SHOULDER, KP_R_HIP),
    (KP_L_HIP, KP_R_HIP),
    (KP_L_SHOULDER, KP_R_SHOULDER),

    (KP_L_SHOULDER, KP_L_ELBOW),
    (KP_L_ELBOW, KP_L_WRIST),
    (KP_R_SHOULDER, KP_R_ELBOW),
    (KP_R_ELBOW, KP_R_WRIST),
]

ATTACKER_ARM_BONES = [
    (KP_L_SHOULDER, KP_L_ELBOW),
    (KP_L_ELBOW, KP_L_WRIST),
    (KP_R_SHOULDER, KP_R_ELBOW),
    (KP_R_ELBOW, KP_R_WRIST),
]

VICTIM_UPPER_BONES = [

    (KP_NOSE, KP_L_EAR),
    (KP_NOSE, KP_R_EAR),
    (KP_L_EAR, KP_R_EAR),
    (KP_NOSE, KP_L_SHOULDER),
    (KP_NOSE, KP_R_SHOULDER),

    (KP_L_SHOULDER, KP_R_SHOULDER),

    (KP_L_SHOULDER, KP_L_HIP),
    (KP_R_SHOULDER, KP_R_HIP),

    (KP_L_HIP, KP_R_HIP),
]

def _bones_min_distance(pose_a, bones_a, pose_b, bones_b,
                        kp_conf_thresh=0.3
                        ) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
    if pose_a is None or pose_b is None:
        return float("inf"), None

    best_d = float("inf")
    best_pair = None

    for (i1, i2) in bones_a:
        p1 = _kp(pose_a, i1, kp_conf_thresh)
        p2 = _kp(pose_a, i2, kp_conf_thresh)
        if p1 is None or p2 is None:
            continue
        for (j1, j2) in bones_b:
            q1 = _kp(pose_b, j1, kp_conf_thresh)
            q2 = _kp(pose_b, j2, kp_conf_thresh)
            if q1 is None or q2 is None:
                continue
            d = _segment_segment_distance(p1, p2, q1, q2)
            if d < best_d:
                best_d = d
                best_pair = (i1, i2, j1, j2)

    return best_d, best_pair

def _victim_descent_metric(pose_before, pose_after_or_list,
                            v_height_baseline: float
                            ) -> Tuple[float, dict]:
    feats = {}
    if v_height_baseline <= 1:
        return 0.0, {"reason": "no baseline height"}

    if isinstance(pose_after_or_list, list):
        poses_after = [p for p in pose_after_or_list if p is not None]
    else:
        poses_after = [pose_after_or_list] if pose_after_or_list else []

    if pose_before is None or not poses_after:
        return 0.0, {"reason": "missing baseline or current poses"}

    hp_before = _midpoint(_kp(pose_before, KP_L_HIP),
                          _kp(pose_before, KP_R_HIP))
    hp_ys_after = []
    for p in poses_after:
        hp = _midpoint(_kp(p, KP_L_HIP), _kp(p, KP_R_HIP))
        if hp is not None:
            hp_ys_after.append(hp[1])

    hip_drop_norm = 0.0
    if hp_before is not None and hp_ys_after:

        hp_y_after_median = float(np.median(hp_ys_after))
        hip_drop = hp_y_after_median - hp_before[1]
        hip_drop_norm = max(0.0, hip_drop / v_height_baseline)
        feats["hip_drop_norm"] = hip_drop_norm

    torso_delta = 0.0
    a_before = _torso_angle_deg(pose_before)
    torso_after_vals = [_torso_angle_deg(p) for p in poses_after]
    torso_after_vals = [v for v in torso_after_vals if v is not None]
    if a_before is not None and torso_after_vals:
        a_after_median = float(np.median(torso_after_vals))
        torso_delta = max(0.0, a_after_median - a_before)
        feats["torso_delta_deg"] = torso_delta

    bh_before = pose_before.bbox[3] - pose_before.bbox[1]
    bh_after_vals = [p.bbox[3] - p.bbox[1] for p in poses_after]
    bbox_compression = 0.0
    bbox_height_ratio = 1.0
    if bh_before > 1 and bh_after_vals:
        bh_after_median = float(np.median(bh_after_vals))
        bbox_height_ratio = bh_after_median / bh_before

        compression_ratio = 1.0 - bbox_height_ratio
        bbox_compression = max(0.0, compression_ratio)
        feats["bbox_compression"] = bbox_compression
        feats["bbox_height_ratio"] = bbox_height_ratio

    if bbox_height_ratio > 1.05:

        hip_drop_norm /= bbox_height_ratio
        feats["zoom_compensated"] = bbox_height_ratio

    descent = (
        1.0 * hip_drop_norm +
        0.4 * min(1.0, torso_delta / 60) +
        0.5 * bbox_compression
    )
    feats["descent"] = descent
    return descent, feats

@dataclass
class RuleTrace:
    rule: str
    attacker_id: int
    victim_id: int
    passed: bool
    features: Dict[str, float] = field(default_factory=dict)
    rejected_by: Optional[str] = None
    reject_reason: Optional[str] = None
    score: float = 0.0

    debug_lines: List[Tuple[int, int]] = field(default_factory=list)

_DEBUG_TRACES: List[RuleTrace] = []
_DEBUG_ENABLED = False

def set_debug_mode(enabled: bool):
    global _DEBUG_ENABLED
    _DEBUG_ENABLED = enabled

def get_debug_traces() -> List[RuleTrace]:
    global _DEBUG_TRACES
    out = _DEBUG_TRACES
    _DEBUG_TRACES = []
    return out

def _trace(rule_name: str, aid: int, vid: int, passed: bool,
           features: Dict[str, float],
           rejected_by: Optional[str] = None,
           reject_reason: Optional[str] = None,
           score: float = 0.0,
           debug_lines: Optional[List[Tuple[int, int]]] = None) -> None:
    if not _DEBUG_ENABLED:
        return
    _DEBUG_TRACES.append(RuleTrace(
        rule=rule_name, attacker_id=aid, victim_id=vid,
        passed=passed, features=features,
        rejected_by=rejected_by, reject_reason=reject_reason,
        score=score, debug_lines=debug_lines or [],
    ))

def rule_high_kick(
    attacker_id: int, attacker_pose,
    victim_id: int, victim_pose,
) -> Optional[FoulSignal]:
    v_torso = _torso_angle_deg(victim_pose)
    if v_torso is None or v_torso > 45:
        return None

    a_hip = _midpoint(_kp(attacker_pose, KP_L_HIP),
                      _kp(attacker_pose, KP_R_HIP))
    if a_hip is None:
        return None

    a_ank_l = _kp(attacker_pose, KP_L_ANKLE)
    a_ank_r = _kp(attacker_pose, KP_R_ANKLE)

    raised = []
    for ap, idx in [(a_ank_l, KP_L_ANKLE), (a_ank_r, KP_R_ANKLE)]:
        if ap is None:
            continue

        a_h = _body_height(attacker_pose) or 100
        if a_hip[1] - ap[1] > 0.2 * a_h:
            raised.append((ap, idx))
    if not raised:
        return None

    v_targets = []
    for p, idx in [
        (_kp(victim_pose, KP_NOSE), KP_NOSE),
        (_kp(victim_pose, KP_L_SHOULDER), KP_L_SHOULDER),
        (_kp(victim_pose, KP_R_SHOULDER), KP_R_SHOULDER),
        (_midpoint(_kp(victim_pose, KP_L_SHOULDER),
                   _kp(victim_pose, KP_R_SHOULDER)), -1),
    ]:
        if p is not None:
            v_targets.append((p, idx if idx >= 0 else KP_L_SHOULDER))

    if not v_targets:
        return None

    v_h = _body_height(victim_pose) or 100

    best_dist = float("inf")
    best_pair = None
    for ap, aidx in raised:
        for vp, vidx in v_targets:
            d = _dist(ap, vp)
            if d < best_dist:
                best_dist = d
                best_pair = (aidx, vidx)

    norm_dist = best_dist / v_h
    if norm_dist > 0.4:
        return None

    closeness = max(0.0, 1.0 - norm_dist / 0.4)

    a_h = _body_height(attacker_pose) or 100
    ank = next(ap for ap, _ in raised)
    height_above_hip = (a_hip[1] - ank[1]) / a_h
    raise_score = min(1.0, height_above_hip / 0.5)
    score = 0.5 * closeness + 0.5 * raise_score
    score = float(np.clip(score, 0.0, 1.0))
    if score < 0.5:
        return None

    return FoulSignal(
        kind="high_kick",
        score=score,
        attacker_id=attacker_id,
        victim_id=victim_id,
        attacker_kpts=[best_pair[0]],
        victim_kpts=[best_pair[1]],
        cross_lines=[best_pair],
        reason=(f"raised foot near victim's upper body "
                f"({norm_dist:.2f}h)"),
    )

def rule_elbow_to_head(
    attacker_id: int, attacker_pose,
    victim_id: int, victim_pose,
) -> Optional[FoulSignal]:
    v_torso = _torso_angle_deg(victim_pose)
    if v_torso is None or v_torso > 50:
        return None

    v_head_targets = []
    for p, idx in [
        (_kp(victim_pose, KP_NOSE), KP_NOSE),
        (_kp(victim_pose, KP_L_EAR), KP_L_EAR),
        (_kp(victim_pose, KP_R_EAR), KP_R_EAR),
    ]:
        if p is not None:
            v_head_targets.append((p, idx))
    if not v_head_targets:
        return None

    v_h = _body_height(victim_pose) or 100

    raised_elbows = []
    for elb_p, elb_idx, sh_p in [
        (_kp(attacker_pose, KP_L_ELBOW), KP_L_ELBOW,
         _kp(attacker_pose, KP_L_SHOULDER)),
        (_kp(attacker_pose, KP_R_ELBOW), KP_R_ELBOW,
         _kp(attacker_pose, KP_R_SHOULDER)),
    ]:
        if elb_p is None or sh_p is None:
            continue

        if sh_p[1] - elb_p[1] > 0.05 * (_body_height(attacker_pose) or 100):
            raised_elbows.append((elb_p, elb_idx))
    if not raised_elbows:
        return None

    best_dist = float("inf")
    best_pair = None
    for ep, eidx in raised_elbows:
        for hp, hidx in v_head_targets:
            d = _dist(ep, hp)
            if d < best_dist:
                best_dist = d
                best_pair = (eidx, hidx)

    norm = best_dist / v_h
    if norm > 0.25:
        return None

    closeness = max(0.0, 1.0 - norm / 0.25)
    score = closeness
    score = float(np.clip(score, 0.0, 1.0))
    if score < 0.55:
        return None

    return FoulSignal(
        kind="elbow_to_head",
        score=score,
        attacker_id=attacker_id,
        victim_id=victim_id,
        attacker_kpts=[best_pair[0]],
        victim_kpts=[best_pair[1]],
        cross_lines=[best_pair],
        reason=f"raised elbow near victim's head ({norm:.2f}h)",
    )

def rule_head_butt(
    attacker_id: int, attacker_pose,
    victim_id: int, victim_pose,
) -> Optional[FoulSignal]:
    a_nose = _kp(attacker_pose, KP_NOSE)
    v_nose = _kp(victim_pose, KP_NOSE)
    if a_nose is None or v_nose is None:
        return None

    v_h = _body_height(victim_pose) or _body_height(attacker_pose) or 100
    d = _dist(a_nose, v_nose)
    norm = d / v_h
    if norm > 0.2:
        return None

    a_torso = _torso_angle_deg(attacker_pose) or 0
    v_torso = _torso_angle_deg(victim_pose) or 0
    if a_torso < 15 and v_torso < 15:

        return None

    closeness = max(0.0, 1.0 - norm / 0.2)
    tilt = min(1.0, max(a_torso, v_torso) / 40)
    score = 0.6 * closeness + 0.4 * tilt
    score = float(np.clip(score, 0.0, 1.0))
    if score < 0.55:
        return None

    return FoulSignal(
        kind="head_butt",
        score=score,
        attacker_id=attacker_id,
        victim_id=victim_id,
        attacker_kpts=[KP_NOSE],
        victim_kpts=[KP_NOSE],
        cross_lines=[(KP_NOSE, KP_NOSE)],
        reason=(f"heads very close ({norm:.2f}h), "
                f"attacker tilted {a_torso:.0f}°"),
    )

STATELESS_RULES = [
    rule_high_kick,
    rule_elbow_to_head,
    rule_head_butt,
]

_STATE_IDLE = "IDLE"
_STATE_WAITING = "WAITING_FOR_FALL"

@dataclass
class _PoseSnapshot:
    frame: int
    pose: Optional["PoseDetection"]

class SlideTackleDetector:

    def __init__(self,
                 contact_thresh: float = 0.05,
                 fall_fast_window: int = 15,
                 fall_slow_window: int = 90,
                 descent_threshold: float = 0.25,
                 cooldown_after_emit: int = 30,
                 contact_min_consecutive: int = 2,
                 pre_filter_bbox_dist: float = 0.5,
                 descent_history_size: int = 5):
        self.contact_thresh = contact_thresh
        self.fall_fast_window = fall_fast_window
        self.fall_slow_window = fall_slow_window
        self.descent_threshold = descent_threshold
        self.cooldown_after_emit = cooldown_after_emit
        self.contact_min_consecutive = contact_min_consecutive
        self.pre_filter_bbox_dist = pre_filter_bbox_dist
        self.descent_history_size = descent_history_size

        self.state = _STATE_IDLE

        self.contact_frame: int = -1

        self.baseline_pose: Optional["PoseDetection"] = None
        self.baseline_height: float = 0.0
        self.baseline_torso: float = 0.0

        self.contact_bones: Optional[Tuple[int, int, int, int]] = None

        self.cooldown_until: int = -1

        self._consecutive_contact: int = 0

        self._pending_contact_pair: Optional[Tuple[int, int, int, int]] = None
        self._pending_contact_norm: float = 0.0

        self._victim_history: deque = deque(maxlen=descent_history_size)

        self.last_contact_distance: float = float("inf")

    def update(self,
               frame_num: int,
               attacker_id: int,
               victim_id: int,
               attacker_pose: Optional["PoseDetection"],
               victim_pose: Optional["PoseDetection"],
               ) -> Optional[FoulSignal]:

        if frame_num < self.cooldown_until:
            return None

        if attacker_pose is None or victim_pose is None:

            if self.state == _STATE_WAITING:

                self._victim_history.append(None)
                self._maybe_timeout(frame_num)

            elif self.state == _STATE_IDLE:
                self._consecutive_contact = 0
            return None

        if self.state == _STATE_IDLE:
            v_h_quick = _body_height(victim_pose) or 0
            if v_h_quick > 0:
                bbox_d = _bbox_distance(attacker_pose.bbox, victim_pose.bbox)
                if bbox_d > self.pre_filter_bbox_dist * v_h_quick:

                    self._consecutive_contact = 0
                    return None

        contact_dist, contact_pair = _bones_min_distance(
            attacker_pose, ATTACKER_LEG_BONES,
            victim_pose, VICTIM_ALL_BONES,
        )
        self.last_contact_distance = contact_dist
        v_h = _body_height(victim_pose) or 0
        contact_norm = contact_dist / v_h if v_h > 0 else float("inf")
        is_contact = (contact_norm < self.contact_thresh)

        if self.state == _STATE_IDLE:
            if is_contact and v_h > 20:

                self._consecutive_contact += 1

                self._pending_contact_pair = contact_pair
                self._pending_contact_norm = contact_norm

                if self._consecutive_contact >= self.contact_min_consecutive:

                    self.state = _STATE_WAITING
                    self.contact_frame = frame_num
                    self.baseline_pose = victim_pose
                    self.baseline_height = v_h
                    self.baseline_torso = _torso_angle_deg(victim_pose) or 0
                    self.contact_bones = contact_pair
                    self._victim_history.clear()
                    self._consecutive_contact = 0

                    _trace("slide_tackle", attacker_id, victim_id, False,
                           features={"contact_norm": contact_norm,
                                     "v_h": v_h,
                                     "state": 1.0,
                                     "torso_baseline": self.baseline_torso},
                           rejected_by="awaiting_fall",
                           reject_reason=f"contact at d={contact_norm:.3f}h, "
                                         f"waiting for fall...",
                           debug_lines=[(contact_pair[0], contact_pair[2])]
                                        if contact_pair else [])
                else:

                    _trace("slide_tackle", attacker_id, victim_id, False,
                           features={"contact_norm": contact_norm,
                                     "consecutive": float(self._consecutive_contact)},
                           rejected_by="contact_unconfirmed",
                           reject_reason=(
                               f"contact {self._consecutive_contact}/"
                               f"{self.contact_min_consecutive} frames"),
                           debug_lines=[(contact_pair[0], contact_pair[2])]
                                        if contact_pair else [])
            else:

                self._consecutive_contact = 0
            return None

        self._victim_history.append(victim_pose)
        history_list = list(self._victim_history)
        descent, feats = _victim_descent_metric(
            self.baseline_pose, history_list, self.baseline_height
        )
        frames_since_contact = frame_num - self.contact_frame

        if descent >= self.descent_threshold:

            if frames_since_contact <= self.fall_fast_window:
                kind = "slide_tackle"
                kind_label = "slide_tackle"

                speed = 1.0 - frames_since_contact / self.fall_fast_window
                conf_score = min(1.0, descent / 0.5)
                score = 0.5 * conf_score + 0.5 * speed
            else:
                kind = "slide_tackle_delayed"
                kind_label = "slide_tackle (delayed)"

                lateness = (frames_since_contact - self.fall_fast_window) / \
                           (self.fall_slow_window - self.fall_fast_window)
                lateness = min(1.0, lateness)
                conf_score = min(1.0, descent / 0.5)

                score = (1.0 - 0.5 * lateness) * conf_score

            score = float(np.clip(score, 0.0, 1.0))

            _trace("slide_tackle", attacker_id, victim_id, True,
                   features={"descent": descent,
                             "frames_since_contact": float(frames_since_contact),
                             **feats,
                             "score": score},
                   score=score,
                   debug_lines=[(self.contact_bones[0],
                                 self.contact_bones[2])]
                               if self.contact_bones else [])

            cb = self.contact_bones
            cross_lines = [(cb[0], cb[2])] if cb else []
            attacker_kpts = [cb[0], cb[1]] if cb else []
            victim_kpts = [cb[2], cb[3]] if cb else []

            reason = (f"contact at f{self.contact_frame}, "
                      f"fall +{frames_since_contact}f, "
                      f"descent={descent:.2f}")

            candidate = None
            try:
                from ml_classifier import FoulCandidate
                candidate = FoulCandidate(
                    kind=kind,
                    contact_norm=self._pending_contact_norm,
                    contact_bone_attacker_kp=cb[0] if cb else -1,
                    contact_bone_victim_kp=cb[2] if cb else -1,
                    descent=descent,
                    hip_drop_norm=feats.get("hip_drop_norm", 0.0),
                    torso_delta_deg=feats.get("torso_delta_deg", 0.0),
                    bbox_compression=feats.get("bbox_compression", 0.0),
                    bbox_height_ratio=feats.get("bbox_height_ratio", 1.0),
                    frames_since_contact=int(frames_since_contact),
                    zoom_compensated=feats.get("zoom_compensated", 1.0),

                    attacker_torso_angle=(
                        _torso_angle_deg(attacker_pose) or 0.0
                    ),
                    attacker_height=_body_height(attacker_pose) or 0.0,
                    attacker_wrist_speed=0.0,
                    attacker_leg_outreach=0.0,
                    victim_torso_baseline=self.baseline_torso,
                    victim_height_baseline=self.baseline_height,
                )
            except ImportError:

                pass

            signal = FoulSignal(
                kind=kind,
                score=score,
                attacker_id=attacker_id,
                victim_id=victim_id,
                attacker_kpts=attacker_kpts,
                victim_kpts=victim_kpts,
                cross_lines=cross_lines,
                reason=reason,
                candidate=candidate,
            )

            self.cooldown_until = frame_num + self.cooldown_after_emit
            self._reset()
            return signal

        if frames_since_contact >= self.fall_slow_window:

            _trace("slide_tackle", attacker_id, victim_id, False,
                   features={"descent": descent,
                             "frames_since_contact": float(frames_since_contact),
                             **feats},
                   rejected_by="timeout",
                   reject_reason=(f"no fall within {self.fall_slow_window} "
                                  f"frames (descent {descent:.2f} < "
                                  f"{self.descent_threshold})"))
            self._reset()
            return None

        _trace("slide_tackle", attacker_id, victim_id, False,
               features={"descent": descent,
                         "frames_since_contact": float(frames_since_contact),
                         **feats},
               rejected_by="awaiting_fall",
               reject_reason=(f"in WAITING +{frames_since_contact}f, "
                              f"descent {descent:.2f} (<{self.descent_threshold})"))
        return None

    def _maybe_timeout(self, frame_num: int):
        if frame_num - self.contact_frame >= self.fall_slow_window:
            self._reset()

    def _reset(self):
        self.state = _STATE_IDLE
        self.contact_frame = -1
        self.baseline_pose = None
        self.baseline_height = 0.0
        self.contact_bones = None
        self._consecutive_contact = 0
        self._pending_contact_pair = None
        self._victim_history.clear()

class ArmStrikeDetector:

    def __init__(self,
                 contact_thresh: float = 0.05,
                 fall_fast_window: int = 15,
                 fall_slow_window: int = 90,
                 descent_threshold: float = 0.25,
                 cooldown_after_emit: int = 30,
                 wrist_history_size: int = 5,
                 high_speed_threshold: float = 0.10,
                 contact_min_consecutive: int = 2,
                 pre_filter_bbox_dist: float = 0.5,
                 descent_history_size: int = 5):
        self.contact_thresh = contact_thresh
        self.fall_fast_window = fall_fast_window
        self.fall_slow_window = fall_slow_window
        self.descent_threshold = descent_threshold
        self.cooldown_after_emit = cooldown_after_emit
        self.wrist_history_size = wrist_history_size
        self.high_speed_threshold = high_speed_threshold
        self.contact_min_consecutive = contact_min_consecutive
        self.pre_filter_bbox_dist = pre_filter_bbox_dist
        self.descent_history_size = descent_history_size

        self.state = _STATE_IDLE
        self.contact_frame: int = -1
        self.baseline_pose: Optional["PoseDetection"] = None
        self.baseline_height: float = 0.0
        self.baseline_torso: float = 0.0
        self.contact_bones: Optional[Tuple[int, int, int, int]] = None

        self.contact_wrist_speed: float = 0.0

        self._contact_norm_at_emission: float = 0.0
        self.cooldown_until: int = -1

        self._wrist_history: deque = deque(maxlen=wrist_history_size)

        self._consecutive_contact: int = 0

        self._victim_history: deque = deque(maxlen=descent_history_size)
        self.last_contact_distance: float = float("inf")

    def update(self,
               frame_num: int,
               attacker_id: int,
               victim_id: int,
               attacker_pose: Optional["PoseDetection"],
               victim_pose: Optional["PoseDetection"],
               ) -> Optional[FoulSignal]:

        if frame_num < self.cooldown_until:
            return None

        if attacker_pose is None or victim_pose is None:

            self._wrist_history.append(None)
            if self.state == _STATE_WAITING:
                self._victim_history.append(None)
                self._maybe_timeout(frame_num)
            elif self.state == _STATE_IDLE:
                self._consecutive_contact = 0
            return None

        if self.state == _STATE_IDLE:
            v_h_quick = _body_height(victim_pose) or 0
            if v_h_quick > 0:
                bbox_d = _bbox_distance(attacker_pose.bbox, victim_pose.bbox)
                if bbox_d > self.pre_filter_bbox_dist * v_h_quick:
                    self._consecutive_contact = 0

                    return None

        wl = _kp(attacker_pose, KP_L_WRIST)
        wr = _kp(attacker_pose, KP_R_WRIST)
        self._wrist_history.append((
            frame_num,
            wl[0] if wl else None, wl[1] if wl else None,
            wr[0] if wr else None, wr[1] if wr else None,
        ))

        contact_dist, contact_pair = _bones_min_distance(
            attacker_pose, ATTACKER_ARM_BONES,
            victim_pose, VICTIM_UPPER_BONES,
        )
        self.last_contact_distance = contact_dist
        v_h = _body_height(victim_pose) or 0
        contact_norm = contact_dist / v_h if v_h > 0 else float("inf")
        is_contact = (contact_norm < self.contact_thresh)

        if self.state == _STATE_IDLE:
            if is_contact and v_h > 20:

                self._consecutive_contact += 1

                if self._consecutive_contact >= self.contact_min_consecutive:
                    self.state = _STATE_WAITING
                    self.contact_frame = frame_num
                    self.baseline_pose = victim_pose
                    self.baseline_height = v_h
                    self.baseline_torso = _torso_angle_deg(victim_pose) or 0
                    self.contact_bones = contact_pair
                    self.contact_wrist_speed = self._estimate_wrist_speed(
                        attacker_pose, contact_pair
                    )
                    self._contact_norm_at_emission = contact_norm
                    self._victim_history.clear()
                    self._consecutive_contact = 0

                    _trace("arm_strike", attacker_id, victim_id, False,
                           features={"contact_norm": contact_norm,
                                     "v_h": v_h,
                                     "wrist_speed": self.contact_wrist_speed,
                                     "state": 1.0},
                           rejected_by="awaiting_fall",
                           reject_reason=(
                               f"contact at d={contact_norm:.3f}h, "
                               f"wrist_v={self.contact_wrist_speed:.3f}, "
                               f"waiting for fall..."),
                           debug_lines=[(contact_pair[0], contact_pair[2])]
                                        if contact_pair else [])
                else:
                    _trace("arm_strike", attacker_id, victim_id, False,
                           features={"contact_norm": contact_norm,
                                     "consecutive": float(self._consecutive_contact)},
                           rejected_by="contact_unconfirmed",
                           reject_reason=(
                               f"contact {self._consecutive_contact}/"
                               f"{self.contact_min_consecutive} frames"),
                           debug_lines=[(contact_pair[0], contact_pair[2])]
                                        if contact_pair else [])
            else:
                self._consecutive_contact = 0
            return None

        self._victim_history.append(victim_pose)
        history_list = list(self._victim_history)
        descent, feats = _victim_descent_metric(
            self.baseline_pose, history_list, self.baseline_height
        )
        frames_since_contact = frame_num - self.contact_frame

        if descent >= self.descent_threshold:
            if frames_since_contact <= self.fall_fast_window:
                kind = "arm_strike"
                speed = 1.0 - frames_since_contact / self.fall_fast_window
                conf_score = min(1.0, descent / 0.5)
                base_score = 0.5 * conf_score + 0.5 * speed
            else:
                kind = "arm_strike_delayed"
                lateness = (frames_since_contact - self.fall_fast_window) / \
                           (self.fall_slow_window - self.fall_fast_window)
                lateness = min(1.0, lateness)
                conf_score = min(1.0, descent / 0.5)
                base_score = (1.0 - 0.5 * lateness) * conf_score

            speed_bonus = min(0.15,
                              0.15 * self.contact_wrist_speed /
                              self.high_speed_threshold)
            speed_bonus = max(0.0, speed_bonus)
            score = float(np.clip(base_score + speed_bonus, 0.0, 1.0))

            _trace("arm_strike", attacker_id, victim_id, True,
                   features={"descent": descent,
                             "frames_since_contact": float(frames_since_contact),
                             "wrist_speed": self.contact_wrist_speed,
                             "speed_bonus": speed_bonus,
                             **feats,
                             "score": score},
                   score=score,
                   debug_lines=[(self.contact_bones[0],
                                 self.contact_bones[2])]
                               if self.contact_bones else [])

            cb = self.contact_bones
            cross_lines = [(cb[0], cb[2])] if cb else []
            attacker_kpts = [cb[0], cb[1]] if cb else []
            victim_kpts = [cb[2], cb[3]] if cb else []

            speed_note = (f", wrist_v={self.contact_wrist_speed:.2f}"
                          if self.contact_wrist_speed > 0 else "")
            reason = (f"contact at f{self.contact_frame}, "
                      f"fall +{frames_since_contact}f, "
                      f"descent={descent:.2f}{speed_note}")

            candidate = None
            try:
                from ml_classifier import FoulCandidate
                candidate = FoulCandidate(
                    kind=kind,
                    contact_norm=self._contact_norm_at_emission,
                    contact_bone_attacker_kp=cb[0] if cb else -1,
                    contact_bone_victim_kp=cb[2] if cb else -1,
                    descent=descent,
                    hip_drop_norm=feats.get("hip_drop_norm", 0.0),
                    torso_delta_deg=feats.get("torso_delta_deg", 0.0),
                    bbox_compression=feats.get("bbox_compression", 0.0),
                    bbox_height_ratio=feats.get("bbox_height_ratio", 1.0),
                    frames_since_contact=int(frames_since_contact),
                    zoom_compensated=feats.get("zoom_compensated", 1.0),
                    attacker_torso_angle=(
                        _torso_angle_deg(attacker_pose) or 0.0
                    ),
                    attacker_height=_body_height(attacker_pose) or 0.0,
                    attacker_wrist_speed=self.contact_wrist_speed,
                    attacker_leg_outreach=0.0,
                    victim_torso_baseline=self.baseline_torso,
                    victim_height_baseline=self.baseline_height,
                )
            except ImportError:
                pass

            signal = FoulSignal(
                kind=kind,
                score=score,
                attacker_id=attacker_id,
                victim_id=victim_id,
                attacker_kpts=attacker_kpts,
                victim_kpts=victim_kpts,
                cross_lines=cross_lines,
                reason=reason,
                candidate=candidate,
            )

            self.cooldown_until = frame_num + self.cooldown_after_emit
            self._reset()
            return signal

        if frames_since_contact >= self.fall_slow_window:
            _trace("arm_strike", attacker_id, victim_id, False,
                   features={"descent": descent,
                             "frames_since_contact": float(frames_since_contact),
                             **feats},
                   rejected_by="timeout",
                   reject_reason=(f"no fall within {self.fall_slow_window} "
                                  f"frames"))
            self._reset()
            return None

        _trace("arm_strike", attacker_id, victim_id, False,
               features={"descent": descent,
                         "frames_since_contact": float(frames_since_contact),
                         **feats},
               rejected_by="awaiting_fall",
               reject_reason=(f"in WAITING +{frames_since_contact}f, "
                              f"descent {descent:.2f}"))
        return None

    def _estimate_wrist_speed(self, attacker_pose,
                               contact_pair) -> float:
        if contact_pair is None:
            return 0.0
        a_kp1, a_kp2, _, _ = contact_pair

        target_idx = None
        if KP_L_WRIST in (a_kp1, a_kp2):
            target_idx = "L"
        elif KP_R_WRIST in (a_kp1, a_kp2):
            target_idx = "R"

        elif KP_L_ELBOW in (a_kp1, a_kp2):

            cur = _kp(attacker_pose, KP_L_ELBOW)
            return self._point_speed_history(cur, "L_ELBOW", attacker_pose)
        elif KP_R_ELBOW in (a_kp1, a_kp2):
            cur = _kp(attacker_pose, KP_R_ELBOW)
            return self._point_speed_history(cur, "R_ELBOW", attacker_pose)

        if target_idx is None:
            return 0.0

        cur = _kp(attacker_pose,
                  KP_L_WRIST if target_idx == "L" else KP_R_WRIST)
        if cur is None:
            return 0.0

        a_h = _body_height(attacker_pose)
        if a_h is None or a_h < 1:
            return 0.0

        for offset in [3, 2, 1]:
            if len(self._wrist_history) <= offset:
                continue
            prev = self._wrist_history[-offset - 1]
            if prev is None:
                continue
            if target_idx == "L":
                px, py = prev[1], prev[2]
            else:
                px, py = prev[3], prev[4]
            if px is None or py is None:
                continue
            dx = cur[0] - px
            dy = cur[1] - py
            dist = float(np.hypot(dx, dy))
            return dist / a_h / offset
        return 0.0

    def _point_speed_history(self, cur, label, attacker_pose) -> float:
        if cur is None:
            return 0.0
        a_h = _body_height(attacker_pose)
        if a_h is None:
            return 0.0

        return 0.0

    def _maybe_timeout(self, frame_num: int):
        if frame_num - self.contact_frame >= self.fall_slow_window:
            self._reset()

    def _reset(self):
        self.state = _STATE_IDLE
        self.contact_frame = -1
        self.baseline_pose = None
        self.baseline_height = 0.0
        self.baseline_torso = 0.0
        self.contact_bones = None
        self.contact_wrist_speed = 0.0
        self._contact_norm_at_emission = 0.0
        self._consecutive_contact = 0
        self._victim_history.clear()

class InteractionDetector:

    def __init__(self,
                 window_size: int = 7,
                 min_hits: int = 4,
                 max_pair_distance: float = 2.0,
                 cooldown: int = 10,
                 classifier: Optional[object] = None,
                 training_data_writer: Optional[object] = None):
        self.window_size = window_size
        self.min_hits = min_hits
        self.max_pair_distance = max_pair_distance
        self.cooldown = cooldown
        self.classifier = classifier
        self.training_data_writer = training_data_writer

        self._windows: Dict[Tuple[int, int, str], deque] = {}

        self._active: Dict[Tuple[int, int, str], FoulEvent] = {}

        self.history: List[FoulEvent] = []

        self._cooldown: Dict[Tuple[int, int, str], int] = {}

        self._slide_detectors: Dict[Tuple[int, int], SlideTackleDetector] = {}

        self._arm_detectors: Dict[Tuple[int, int], ArmStrikeDetector] = {}

        self._last_frame = -1

    def update(self,
               frame_idx: int,
               poses: Dict[int, "PoseDetection"],
               frame_size: Optional[Tuple[int, int]] = None,
               ) -> List[FoulSignal]:
        self._last_frame = frame_idx

        valid_ids = [tid for tid, p in poses.items() if p is not None]
        n = len(valid_ids)

        per_frame_signals: List[FoulSignal] = []

        triggers_this_frame: Dict[Tuple[int, int, str], FoulSignal] = {}

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                aid = valid_ids[i]
                vid = valid_ids[j]
                ap = poses[aid]
                vp = poses[vid]

                v_h = _body_height(vp)
                if v_h is None:
                    continue
                bdist = _bbox_distance(ap.bbox, vp.bbox)
                if bdist > self.max_pair_distance * v_h:
                    continue

                for rule_fn in STATELESS_RULES:
                    sig = rule_fn(aid, ap, vid, vp)
                    if sig is None:
                        continue
                    key = (aid, vid, sig.kind)
                    triggers_this_frame[key] = sig
                    per_frame_signals.append(sig)

        self._update_stateful_detectors(
            frame_idx, poses, per_frame_signals, triggers_this_frame
        )

        self._resolve_stateful_conflicts(
            frame_idx, per_frame_signals, triggers_this_frame
        )

        INSTANT_KINDS = {
            "slide_tackle", "slide_tackle_delayed",
            "arm_strike", "arm_strike_delayed",
        }
        DISPLAY_DURATION = 30

        for key, sig in list(triggers_this_frame.items()):
            attacker_id, victim_id, kind = key
            if kind not in INSTANT_KINDS:
                continue

            cd_until = self._cooldown.get(key, -1)
            if frame_idx < cd_until:
                continue

            final_score = sig.score
            ml_prob = None

            if self.training_data_writer is not None and \
               sig.candidate is not None:
                try:
                    self.training_data_writer.add(
                        candidate=sig.candidate,
                        rule_score=sig.score,
                        frame_num=frame_idx,
                        attacker_id=attacker_id,
                        victim_id=victim_id,
                    )
                except Exception as e:
                    print(f"[ML] training_data_writer.add failed: {e}")

            if self.classifier is not None and sig.candidate is not None:
                try:
                    accepted, final_score, ml_prob = self.classifier.apply(
                        sig.candidate, sig.score
                    )
                    if not accepted:

                        _trace(kind, attacker_id, victim_id, False,
                               features={"rule_score": sig.score,
                                         "ml_prob": ml_prob or -1.0},
                               rejected_by="ml_rejected",
                               reject_reason=(
                                   f"ML prob {ml_prob:.2f} < "
                                   f"{self.classifier.reject_threshold}"))
                        self._cooldown[key] = frame_idx + self.cooldown
                        continue

                    sig.score = final_score
                    if ml_prob is not None:
                        sig.reason = (f"{sig.reason} | "
                                      f"ml={ml_prob:.2f}")
                except Exception as e:
                    print(f"[ML] classifier.apply failed: {e}")

            self._active[key] = FoulEvent(
                kind=kind,
                attacker_id=attacker_id,
                victim_id=victim_id,
                start_frame=frame_idx,
                end_frame=frame_idx + DISPLAY_DURATION,
                hit_count=1,
                avg_score=final_score,
                peak_score=final_score,
                last_signal=sig,
            )

            del triggers_this_frame[key]

        for key in list(self._active.keys()):
            ev = self._active[key]
            if ev.kind in INSTANT_KINDS and frame_idx >= ev.end_frame:
                self.history.append(ev)
                self._cooldown[key] = frame_idx + self.cooldown
                del self._active[key]

        all_keys = set(self._windows.keys()) | set(triggers_this_frame.keys())
        for key in all_keys:
            sig = triggers_this_frame.get(key)
            if key not in self._windows:
                self._windows[key] = deque(maxlen=self.window_size)
            self._windows[key].append(sig)

        for key, window in list(self._windows.items()):
            hits = [s for s in window if s is not None]
            n_hits = len(hits)

            attacker_id, victim_id, kind = key

            cd_until = self._cooldown.get(key, -1)
            in_cooldown = frame_idx < cd_until

            already_active = key in self._active

            if n_hits >= self.min_hits and not in_cooldown:

                avg = float(np.mean([s.score for s in hits]))
                peak = float(max(s.score for s in hits))
                last = hits[-1]
                if already_active:
                    ev = self._active[key]
                    ev.end_frame = frame_idx
                    ev.hit_count += 1 if triggers_this_frame.get(key) else 0
                    ev.avg_score = (ev.avg_score * 0.7 + avg * 0.3)
                    ev.peak_score = max(ev.peak_score, peak)
                    ev.last_signal = last
                else:

                    win_list = list(window)
                    first_hit_offset = next(
                        (len(win_list) - 1 - i for i, s in
                         enumerate(win_list) if s is not None),
                        0
                    )
                    start = frame_idx - first_hit_offset
                    self._active[key] = FoulEvent(
                        kind=kind,
                        attacker_id=attacker_id,
                        victim_id=victim_id,
                        start_frame=start,
                        end_frame=frame_idx,
                        hit_count=n_hits,
                        avg_score=avg,
                        peak_score=peak,
                        last_signal=last,
                    )
            elif already_active:

                ev = self._active.pop(key)
                self.history.append(ev)
                self._cooldown[key] = frame_idx + self.cooldown

            if (not already_active and key not in self._active
                    and all(s is None for s in window)):
                del self._windows[key]

        return per_frame_signals

    def _update_stateful_detectors(
        self,
        frame_idx: int,
        poses: Dict[int, "PoseDetection"],
        per_frame_signals: List[FoulSignal],
        triggers_this_frame: Dict[Tuple[int, int, str], FoulSignal],
    ) -> None:
        valid_ids = list(poses.keys())

        new_pairs = set()
        for aid in valid_ids:
            for vid in valid_ids:
                if aid == vid:
                    continue
                if poses[aid] is None or poses[vid] is None:
                    continue
                key2 = (aid, vid)
                new_pairs.add(key2)
                if key2 not in self._slide_detectors:
                    self._slide_detectors[key2] = SlideTackleDetector()
                if key2 not in self._arm_detectors:
                    self._arm_detectors[key2] = ArmStrikeDetector()

        keys_to_run = (set(self._slide_detectors.keys()) |
                       set(self._arm_detectors.keys()) | new_pairs)

        for (aid, vid) in keys_to_run:
            ap = poses.get(aid)
            vp = poses.get(vid)

            if (aid, vid) in self._slide_detectors:
                sig = self._slide_detectors[(aid, vid)].update(
                    frame_idx, aid, vid, ap, vp
                )
                if sig is not None:
                    key = (aid, vid, sig.kind)
                    triggers_this_frame[key] = sig
                    per_frame_signals.append(sig)

            if (aid, vid) in self._arm_detectors:
                sig = self._arm_detectors[(aid, vid)].update(
                    frame_idx, aid, vid, ap, vp
                )
                if sig is not None:
                    key = (aid, vid, sig.kind)
                    triggers_this_frame[key] = sig
                    per_frame_signals.append(sig)

        for det_dict in (self._slide_detectors, self._arm_detectors):
            to_drop = []
            for (aid, vid), det in det_dict.items():
                if det.state == _STATE_IDLE and \
                   det.cooldown_until < frame_idx - 100 and \
                   (aid, vid) not in new_pairs:
                    to_drop.append((aid, vid))
            for k in to_drop:
                del det_dict[k]

    def _resolve_stateful_conflicts(
        self,
        frame_idx: int,
        per_frame_signals: List[FoulSignal],
        triggers_this_frame: Dict[Tuple[int, int, str], FoulSignal],
    ) -> None:
        SLIDE_KINDS = {"slide_tackle", "slide_tackle_delayed"}
        ARM_KINDS = {"arm_strike", "arm_strike_delayed"}

        by_pair: Dict[Tuple[int, int], List[Tuple[str, FoulSignal]]] = \
            defaultdict(list)
        for key, sig in triggers_this_frame.items():
            aid, vid, kind = key
            if kind in SLIDE_KINDS or kind in ARM_KINDS:
                by_pair[(aid, vid)].append((kind, sig))

        for (aid, vid), items in by_pair.items():
            has_slide = any(k in SLIDE_KINDS for k, _ in items)
            has_arm = any(k in ARM_KINDS for k, _ in items)
            if not (has_slide and has_arm):
                continue

            best_kind, best_sig = max(items, key=lambda x: x[1].score)
            losers = [(k, s) for (k, s) in items if k != best_kind]

            for k, s in losers:
                del triggers_this_frame[(aid, vid, k)]
                try:
                    per_frame_signals.remove(s)
                except ValueError:
                    pass

                if k in SLIDE_KINDS and (aid, vid) in self._slide_detectors:
                    self._slide_detectors[(aid, vid)].cooldown_until = \
                        frame_idx
                elif k in ARM_KINDS and (aid, vid) in self._arm_detectors:
                    self._arm_detectors[(aid, vid)].cooldown_until = \
                        frame_idx

                _trace(k, aid, vid, False,
                       features={"loser_score": s.score,
                                 "winner_score": best_sig.score},
                       rejected_by="conflict_resolved",
                       reject_reason=(f"lost to {best_kind} "
                                      f"({best_sig.score:.2f} > "
                                      f"{s.score:.2f})"))

    def get_active_events(self) -> List[FoulEvent]:
        return list(self._active.values())

    def get_all_events(self) -> List[FoulEvent]:
        return self.history + self.get_active_events()
