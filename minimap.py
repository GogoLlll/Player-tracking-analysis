from __future__ import annotations
from typing import Tuple

import cv2
import numpy as np

from field_homography import FieldModel, HomographyResult

COLOR_FIELD = (45, 100, 50)
COLOR_LINES = (245, 245, 245)
COLOR_BORDER = (220, 220, 220)
COLOR_BG = (28, 28, 28)

LINE_THICKNESS = 2
PLAYER_RADIUS = 7
PLAYER_BORDER = 2
ID_FONT = cv2.FONT_HERSHEY_SIMPLEX
ID_FONT_SCALE = 0.42
ID_FONT_THICKNESS = 1

class Minimap:

    def __init__(self,
                 map_width: int = 300,
                 margin: int = 12,
                 show_title: bool = True,
                 show_grass_pattern: bool = True):
        self.field_w = map_width
        self.field_h = int(round(map_width * FieldModel.WIDTH /
                                 FieldModel.LENGTH))
        self.margin = margin
        self.show_title = show_title
        self.show_grass_pattern = show_grass_pattern

        self.canvas_w = self.field_w + 2 * margin
        self.canvas_h = self.field_h + 2 * margin
        if show_title:
            self.canvas_h += 22

        self.scale_x = self.field_w / FieldModel.LENGTH
        self.scale_y = self.field_h / FieldModel.WIDTH

        self._bg = self._draw_field_template()

    def world_to_minimap(self, points_XY: np.ndarray) -> np.ndarray:
        points_XY = np.asarray(points_XY, dtype=np.float32).reshape(-1, 2)

        x_field = (points_XY[:, 0] + FieldModel.HALF_L) * self.scale_x
        y_field = (points_XY[:, 1] + FieldModel.HALF_W) * self.scale_y

        title_offset = 22 if self.show_title else 0
        x_canvas = x_field + self.margin
        y_canvas = y_field + self.margin + title_offset

        return np.stack([x_canvas, y_canvas], axis=1)

    def _draw_field_template(self) -> np.ndarray:
        canvas = np.full(
            (self.canvas_h, self.canvas_w, 3), COLOR_BG, dtype=np.uint8
        )
        title_offset = 22 if self.show_title else 0

        if self.show_title:
            cv2.rectangle(canvas, (0, 0), (self.canvas_w, 22),
                          (50, 50, 50), -1)
            cv2.putText(canvas, "MINIMAP",
                        (self.margin, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (220, 220, 220), 1, cv2.LINE_AA)

        x0 = self.margin
        y0 = self.margin + title_offset
        x1 = x0 + self.field_w
        y1 = y0 + self.field_h
        cv2.rectangle(canvas, (x0, y0), (x1, y1), COLOR_FIELD, -1)

        if self.show_grass_pattern:
            stripe_count = 14
            stripe_w = self.field_w / stripe_count
            for i in range(stripe_count):
                if i % 2 == 0:
                    sx0 = int(round(x0 + i * stripe_w))
                    sx1 = int(round(x0 + (i + 1) * stripe_w))

                    overlay = canvas.copy()
                    cv2.rectangle(overlay, (sx0, y0), (sx1, y1),
                                  (55, 115, 60), -1)
                    canvas = cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0)

        cv2.rectangle(canvas, (x0, y0), (x1, y1),
                      COLOR_LINES, LINE_THICKNESS)

        cv2.rectangle(canvas, (0, title_offset),
                      (self.canvas_w - 1, self.canvas_h - 1),
                      COLOR_BORDER, 1)

        cx = (x0 + x1) // 2
        cv2.line(canvas, (cx, y0), (cx, y1),
                 COLOR_LINES, LINE_THICKNESS)

        center = self.world_to_minimap(np.array([[0, 0]]))[0]
        radius_px = int(round(FieldModel.CENTER_CIRCLE_RADIUS * self.scale_x))
        cv2.circle(canvas, (int(center[0]), int(center[1])),
                   radius_px, COLOR_LINES, LINE_THICKNESS)

        cv2.circle(canvas, (int(center[0]), int(center[1])),
                   2, COLOR_LINES, -1)

        self._draw_penalty_box(canvas, side="left")
        self._draw_penalty_box(canvas, side="right")
        self._draw_goal_area(canvas, side="left")
        self._draw_goal_area(canvas, side="right")

        for spot in [(-FieldModel.HALF_L + FieldModel.PENALTY_SPOT_DIST, 0),
                     (+FieldModel.HALF_L - FieldModel.PENALTY_SPOT_DIST, 0)]:
            sp = self.world_to_minimap(np.array([spot]))[0]
            cv2.circle(canvas, (int(sp[0]), int(sp[1])),
                       2, COLOR_LINES, -1)

        return canvas

    def _draw_penalty_box(self, canvas, side: str):
        sign = -1 if side == "left" else +1

        x_outer = sign * FieldModel.HALF_L
        x_inner = sign * (FieldModel.HALF_L - FieldModel.PENALTY_DEPTH)
        y_top = -FieldModel.PENALTY_HALF_W
        y_bot = +FieldModel.PENALTY_HALF_W

        corners = self.world_to_minimap(np.array([
            [x_outer, y_top], [x_inner, y_top],
            [x_inner, y_bot], [x_outer, y_bot],
        ]))
        for i in range(len(corners)):
            p1 = tuple(int(c) for c in corners[i])
            p2 = tuple(int(c) for c in corners[(i + 1) % len(corners)])
            cv2.line(canvas, p1, p2, COLOR_LINES, LINE_THICKNESS)

    def _draw_goal_area(self, canvas, side: str):
        sign = -1 if side == "left" else +1
        x_outer = sign * FieldModel.HALF_L
        x_inner = sign * (FieldModel.HALF_L - FieldModel.GOAL_AREA_DEPTH)
        y_top = -FieldModel.GOAL_AREA_HALF_W
        y_bot = +FieldModel.GOAL_AREA_HALF_W

        corners = self.world_to_minimap(np.array([
            [x_outer, y_top], [x_inner, y_top],
            [x_inner, y_bot], [x_outer, y_bot],
        ]))
        for i in range(len(corners)):
            p1 = tuple(int(c) for c in corners[i])
            p2 = tuple(int(c) for c in corners[(i + 1) % len(corners)])
            cv2.line(canvas, p1, p2, COLOR_LINES, LINE_THICKNESS)

    def render(self,
               tracked_objects,
               homography: HomographyResult,
               team_classifier=None) -> np.ndarray:
        canvas = self._bg.copy()

        if homography is None or len(tracked_objects) == 0:
            return canvas

        boxes_xyxy = []
        track_ids = []
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj[:4]
            tid = int(obj[4])
            boxes_xyxy.append((x1, y1, x2, y2))
            track_ids.append(tid)

        boxes_xyxy = np.array(boxes_xyxy, dtype=np.float32)
        ground_points = np.stack([
            (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0,
            boxes_xyxy[:, 3],
        ], axis=1)

        world_points = homography.project_to_world(ground_points)

        for tid, (Xw, Yw) in zip(track_ids, world_points):

            if not FieldModel.is_inside(Xw, Yw, margin=8.0):
                continue

            color = self._get_player_color(tid, team_classifier)

            mp = self.world_to_minimap(np.array([[Xw, Yw]]))[0]
            cx, cy = int(mp[0]), int(mp[1])

            cv2.circle(canvas, (cx, cy),
                       PLAYER_RADIUS + PLAYER_BORDER,
                       (0, 0, 0), -1, lineType=cv2.LINE_AA)

            cv2.circle(canvas, (cx, cy), PLAYER_RADIUS, color, -1,
                       lineType=cv2.LINE_AA)

            label = str(tid)
            (tw, th), _ = cv2.getTextSize(
                label, ID_FONT, ID_FONT_SCALE, ID_FONT_THICKNESS
            )
            tx = cx - tw // 2
            ty = cy - PLAYER_RADIUS - 4

            cv2.rectangle(canvas,
                          (tx - 1, ty - th - 1),
                          (tx + tw + 1, ty + 2),
                          (0, 0, 0), -1)
            cv2.putText(canvas, label, (tx, ty),
                        ID_FONT, ID_FONT_SCALE,
                        (255, 255, 255), ID_FONT_THICKNESS,
                        cv2.LINE_AA)

        return canvas

    def _get_player_color(self, track_id: int,
                           team_classifier) -> Tuple[int, int, int]:
        if team_classifier is not None:
            try:
                team_id, team_name = team_classifier.get_team(track_id)
                if team_name is not None:
                    return team_classifier.get_display_color(team_name)
            except Exception:
                pass

        np.random.seed(int(track_id) * 7)
        return tuple(int(c) for c in np.random.randint(80, 255, size=3))

def overlay_minimap(frame: np.ndarray,
                    minimap_img: np.ndarray,
                    position: str = "top-right",
                    margin: int = 16,
                    alpha: float = 0.92) -> np.ndarray:
    fh, fw = frame.shape[:2]
    mh, mw = minimap_img.shape[:2]

    if position == "top-right":
        x0, y0 = fw - mw - margin, margin
    elif position == "top-left":
        x0, y0 = margin, margin
    elif position == "bottom-right":
        x0, y0 = fw - mw - margin, fh - mh - margin
    elif position == "bottom-left":
        x0, y0 = margin, fh - mh - margin
    else:
        raise ValueError(f"Unknown position: {position}")

    if x0 < 0 or y0 < 0 or x0 + mw > fw or y0 + mh > fh:
        return frame

    if alpha >= 1.0:
        frame[y0:y0 + mh, x0:x0 + mw] = minimap_img
    else:
        roi = frame[y0:y0 + mh, x0:x0 + mw]
        blended = cv2.addWeighted(minimap_img, alpha, roi, 1.0 - alpha, 0)
        frame[y0:y0 + mh, x0:x0 + mw] = blended

    return frame
