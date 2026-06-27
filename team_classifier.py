import cv2
import numpy as np
from collections import defaultdict, deque

TEAM_COLORS = {
    "team_1": {

        "hsv_min": np.array([100, 100, 80]),
        "hsv_max": np.array([130, 255, 255]),
        "display_color": (255, 100, 0),
        "label": "Team 1",
    },
    "team_2": {

        "hsv_min": np.array([0, 0, 180]),
        "hsv_max": np.array([180, 60, 255]),
        "display_color": (255, 255, 255),
        "label": "Team 2",
    },
    "goalkeeper_1": {

        "hsv_min": np.array([35, 50, 30]),
        "hsv_max": np.array([85, 255, 150]),
        "display_color": (0, 130, 0),
        "label": "GK 1",
    },
    "goalkeeper_2": {

        "hsv_min": np.array([5, 120, 150]),
        "hsv_max": np.array([25, 255, 255]),
        "display_color": (0, 140, 255),
        "label": "GK 2",
    },
    "referee": {

        "hsv_min": np.array([0, 0, 0]),
        "hsv_max": np.array([180, 80, 70]),
        "display_color": (50, 50, 50),
        "label": "Referee",
    },
}

TEAM_NAMES = list(TEAM_COLORS.keys())
TEAM_IDS = {name: idx for idx, name in enumerate(TEAM_NAMES)}

class TeamClassifier:

    def __init__(self, history_length=30, crop_top_pct=0.15,
                 crop_bottom_pct=0.45):
        self.history_length = history_length
        self.crop_top_pct = crop_top_pct
        self.crop_bottom_pct = crop_bottom_pct

        self.track_history = defaultdict(
            lambda: deque(maxlen=history_length)
        )

        self.track_team = {}

        self._ranges = [
            (TEAM_COLORS[name]["hsv_min"], TEAM_COLORS[name]["hsv_max"])
            for name in TEAM_NAMES
        ]

    def classify(self, frame, bbox, track_id=None):

        hsv_color = self._extract_torso_color(frame, bbox)
        if hsv_color is None:
            return self._get_cached(track_id)

        scores = self._compute_scores(hsv_color)
        team_idx = np.argmax(scores)
        team_name = TEAM_NAMES[team_idx]
        confidence = scores[team_idx]

        if track_id is not None:
            self.track_history[track_id].append(team_idx)

            history = list(self.track_history[track_id])
            if len(history) >= 5:
                votes = np.bincount(history, minlength=len(TEAM_NAMES))
                stable_idx = np.argmax(votes)
                vote_ratio = votes[stable_idx] / len(history)

                if vote_ratio > 0.6:
                    self.track_team[track_id] = stable_idx
                    team_idx = stable_idx
                    team_name = TEAM_NAMES[team_idx]
                    confidence = vote_ratio

            if track_id in self.track_team:
                team_idx = self.track_team[track_id]
                team_name = TEAM_NAMES[team_idx]

        return team_idx, team_name, confidence

    def classify_batch(self, frame, tracked_objects):
        results = {}
        for obj in tracked_objects:
            bbox = obj[:4]
            tid = int(obj[4])
            team_id, team_name, conf = self.classify(frame, bbox, tid)
            results[tid] = (team_id, team_name, conf)
        return results

    def get_team(self, track_id):
        if track_id in self.track_team:
            idx = self.track_team[track_id]
            return idx, TEAM_NAMES[idx]
        return None, None

    def get_display_color(self, team_name):
        if team_name in TEAM_COLORS:
            return TEAM_COLORS[team_name]["display_color"]
        return (200, 200, 200)

    def get_label(self, team_name):
        if team_name in TEAM_COLORS:
            return TEAM_COLORS[team_name]["label"]
        return "?"

    def same_team(self, tid1, tid2):
        t1 = self.track_team.get(tid1)
        t2 = self.track_team.get(tid2)
        if t1 is None or t2 is None:
            return True
        return t1 == t2

    def _extract_torso_color(self, frame, bbox):
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_frame, x2)
        y2 = min(h_frame, y2)

        bw = x2 - x1
        bh = y2 - y1
        if bw < 5 or bh < 10:
            return None

        torso_y1 = y1 + int(bh * self.crop_top_pct)
        torso_y2 = y1 + int(bh * (1.0 - self.crop_bottom_pct))

        margin_x = int(bw * 0.15)
        torso_x1 = x1 + margin_x
        torso_x2 = x2 - margin_x

        if torso_x2 <= torso_x1 or torso_y2 <= torso_y1:
            return None

        crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        median_h = np.median(hsv[:, :, 0])
        median_s = np.median(hsv[:, :, 1])
        median_v = np.median(hsv[:, :, 2])

        return np.array([median_h, median_s, median_v], dtype=np.float32)

    def _compute_scores(self, hsv_color):
        scores = np.zeros(len(TEAM_NAMES), dtype=np.float32)

        for i, (hsv_min, hsv_max) in enumerate(self._ranges):

            h, s, v = hsv_color

            h_min, h_max = hsv_min[0], hsv_max[0]
            if h_min <= h_max:
                h_in = h_min <= h <= h_max
            else:

                h_in = h >= h_min or h <= h_max

            s_in = hsv_min[1] <= s <= hsv_max[1]
            v_in = hsv_min[2] <= v <= hsv_max[2]

            if h_in and s_in and v_in:

                center = (hsv_min.astype(float) + hsv_max.astype(float)) / 2
                span = (hsv_max.astype(float) - hsv_min.astype(float))
                span = np.where(span < 1, 1, span)

                dh = min(abs(h - center[0]), 180 - abs(h - center[0]))
                ds = abs(s - center[1])
                dv = abs(v - center[2])

                norm_dist = (dh / (span[0]/2 + 1e-6) +
                             ds / (span[1]/2 + 1e-6) +
                             dv / (span[2]/2 + 1e-6)) / 3

                scores[i] = max(0, 1.0 - norm_dist * 0.5)
            else:

                penalties = 0
                count = 0

                if not h_in:
                    if h_min <= h_max:
                        dh = min(abs(h - h_min), abs(h - h_max))
                    else:
                        dh = min(abs(h - h_min), abs(h - h_max),
                                 min(h, 180 - h))
                    penalties += dh / 30.0
                    count += 1

                if not s_in:
                    ds = min(abs(s - hsv_min[1]), abs(s - hsv_max[1]))
                    penalties += ds / 50.0
                    count += 1

                if not v_in:
                    dv = min(abs(v - hsv_min[2]), abs(v - hsv_max[2]))
                    penalties += dv / 50.0
                    count += 1

                if count > 0:
                    scores[i] = max(0, 0.3 - penalties / count)

        return scores

    def _get_cached(self, track_id):
        if track_id is not None and track_id in self.track_team:
            idx = self.track_team[track_id]
            return idx, TEAM_NAMES[idx], 0.5
        return -1, "unknown", 0.0

def draw_team_labels(frame, tracked_objects, team_results):
    for obj in tracked_objects:
        x1, y1, x2, y2 = map(int, obj[:4])
        tid = int(obj[4])

        if tid not in team_results:
            continue

        team_id, team_name, conf = team_results[tid]

        if team_name == "unknown":
            continue

        info = TEAM_COLORS.get(team_name, {})
        color = info.get("display_color", (200, 200, 200))
        label = info.get("label", "?")

        text = f"{label} ({conf:.0%})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                       0.45, 1)
        cv2.rectangle(frame, (x1, y2), (x1 + tw + 4, y2 + th + 6),
                      color, -1)

        brightness = sum(color) / 3
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        cv2.putText(frame, text, (x1 + 2, y2 + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)

        cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1),
                      color, 1)

    return frame

def draw_team_stats(frame, team_results):
    counts = defaultdict(int)
    for tid, (team_id, team_name, conf) in team_results.items():
        if team_name != "unknown":
            counts[team_name] += 1

    y = 110
    for name in TEAM_NAMES:
        info = TEAM_COLORS[name]
        color = info["display_color"]
        label = info["label"]
        count = counts.get(name, 0)

        cv2.putText(frame, f"{label}: {count}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        y += 22

    return frame
