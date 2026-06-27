from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

@dataclass
class RunConfig:

    input_video: str = ""
    output_video: str = "results/result_tracked.mp4"
    merged_video: str = "results/result_merged.mp4"

    detection_model: str = "models/yolo12m.pt"
    pose_model: str = "models/yolo11l-pose.pt"
    reid_weights: str = "osnet_x1_0_msmt17.pt"
    device: str = "cuda:0"

    confidence: float = 0.25
    iou_thresh: float = 0.5
    min_bbox_height: int = 30

    enable_pose: bool = True
    enable_foul_detection: bool = True
    enable_teams: bool = True
    enable_minimap: bool = True
    enable_merge: bool = True
    enable_reid_bank: bool = True

    pose_conf: float = 0.25
    pose_iou: float = 0.5
    pose_imgsz: int = 640
    pose_match_iou: float = 0.4
    draw_skeletons: bool = True
    skeleton_kp_conf: float = 0.5
    enable_pose_smoothing: bool = True
    pose_smooth_alpha: float = 0.5
    pose_smooth_min_conf: float = 0.3

    foul_window_size: int = 7
    foul_min_hits: int = 4
    foul_cooldown: int = 30

    enable_ml_classifier: bool = False
    ml_model_path: str = "models/foul_classifier.txt"
    ml_reject_threshold: float = 0.3
    ml_ensemble_alpha: float = 0.5

    enable_training_export: bool = False
    training_data_csv: str = "data/training_fouls.csv"
    enable_foul_debug: bool = False

    homography_cache_file: str = "homography.npz"
    homography_interpolate: bool = True
    homography_smooth_window: int = 3
    minimap_width: int = 420
    burn_minimap_into_video: bool = False
    minimap_position: str = "top-right"

    team_display_colors: Optional[Dict[str, Tuple[int, int, int]]] = None

    track_buffer: int = 1000
    max_frame_gap: int = 180
    max_spatial_dist: int = 200
    merge_cost_thresh: float = 0.7
    max_player_speed_ms: float = 15.0
    reid_collect_every: int = 5
    reid_match_thresh: float = 0.75

    trail_length: int = 50
    bbox_thickness: int = 2
    font_scale: float = 0.6

@dataclass
class FrameResult:
    phase: str
    frame_num: int
    total_frames: int
    display_frame: Optional[np.ndarray] = None
    minimap_img: Optional[np.ndarray] = None
    tracks: List[dict] = field(default_factory=list)
    events: List[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    message: str = ""

def _color_for_id(track_id: int, team_classifier=None) -> tuple:
    if team_classifier is not None:
        team_id, team_name = team_classifier.get_team(track_id)
        if team_name is not None:
            return team_classifier.get_display_color(team_name)
    np.random.seed(int(track_id) * 7)
    return tuple(int(c) for c in np.random.randint(80, 255, size=3))

def _extract_reid_model_from_tracker(tracker):
    for attr in ("model", "reid_model", "encoder", "appearance"):
        obj = getattr(tracker, attr, None)
        if obj is None:
            continue
        if hasattr(obj, "get_features") or callable(obj):
            return obj
    return None

TEAM_LABELS = {
    "team_1": "Команда 1",
    "team_2": "Команда 2",
    "goalkeeper_1": "Вратарь 1",
    "goalkeeper_2": "Вратарь 2",
    "referee": "Арбитр",
}

class PipelineEngine:

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self._stop = threading.Event()
        self._pause = threading.Event()

        self.model = None
        self.pf_service = None
        self.tracker = None
        self.team_classifier = None
        self.reid_bank = None
        self.homography_tracker = None
        self.minimap = None

        self.tracklets: Dict[int, object] = {}
        self.frame_track_data: Dict[int, list] = {}
        self.trails: Dict[int, list] = {}

        self.fps: float = 30.0
        self._id_map: Dict[int, int] = {}
        self._raw_tracks: Dict[int, dict] = {}
        self._frames_total = 0
        self._frames_with_homo = 0
        self._world_attempts = 0
        self._world_valid = 0
        self._minimap_warning = ""

        self._mods = {}

    def request_stop(self):
        self._stop.set()
        self._pause.clear()

    def request_pause(self):
        self._pause.set()

    def request_resume(self):
        self._pause.clear()

    @property
    def is_paused(self) -> bool:
        return self._pause.is_set()

    def _setup(self):
        cfg = self.cfg

        from ultralytics import YOLO
        from boxmot import BotSort
        from tracklet_merger import (
            Tracklet, TrackletMerger, ReIDBank, apply_merge_to_video,
        )
        from pose_and_fouls import PoseAndFoulsService, PoseFoulsConfig
        from pose_estimator import draw_skeleton
        from field_homography import (
            HomographyTracker, CachedHomography, bbox_to_world_position,
        )
        from minimap import Minimap

        self._mods = dict(
            Tracklet=Tracklet,
            TrackletMerger=TrackletMerger,
            ReIDBank=ReIDBank,
            apply_merge_to_video=apply_merge_to_video,
            bbox_to_world_position=bbox_to_world_position,
            draw_skeleton=draw_skeleton,
        )

        input_path = Path(cfg.input_video)
        if not input_path.exists():
            raise FileNotFoundError(f"Видео не найдено: {input_path}")

        self.model = YOLO(cfg.detection_model)

        self.pf_service = PoseAndFoulsService(PoseFoulsConfig(
            enable_pose=cfg.enable_pose,
            pose_model_name=cfg.pose_model,
            pose_conf=cfg.pose_conf,
            pose_iou=cfg.pose_iou,
            pose_match_iou=cfg.pose_match_iou,
            enable_pose_smoothing=cfg.enable_pose_smoothing,
            pose_smooth_alpha=cfg.pose_smooth_alpha,
            pose_smooth_min_conf=cfg.pose_smooth_min_conf,
            draw_skeletons=cfg.draw_skeletons,
            enable_foul_detection=cfg.enable_foul_detection,
            foul_window_size=cfg.foul_window_size,
            foul_min_hits=cfg.foul_min_hits,
            foul_cooldown=cfg.foul_cooldown,
            enable_ml_classifier=cfg.enable_ml_classifier,
            ml_model_path=cfg.ml_model_path,
            ml_reject_threshold=cfg.ml_reject_threshold,
            ml_ensemble_alpha=cfg.ml_ensemble_alpha,
            enable_training_export=cfg.enable_training_export,
            training_data_csv=cfg.training_data_csv,
            enable_foul_debug=cfg.enable_foul_debug,
        ))

        self.tracker = BotSort(
            Path(cfg.reid_weights), cfg.device, False,
            track_high_thresh=0.3, track_low_thresh=0.1,
            new_track_thresh=0.55, match_thresh=0.82,
            track_buffer=cfg.track_buffer, cmc_method="sof",
            proximity_thresh=0.5, appearance_thresh=0.25, with_reid=True,
        )

        if cfg.enable_teams:
            from team_classifier import TeamClassifier
            self.team_classifier = TeamClassifier()

            if cfg.team_display_colors:
                import team_classifier as tc_mod
                for name, color in cfg.team_display_colors.items():
                    if name in tc_mod.TEAM_COLORS:
                        tc_mod.TEAM_COLORS[name]["display_color"] = \
                            tuple(int(c) for c in color)

        if cfg.enable_reid_bank:
            reid_model = _extract_reid_model_from_tracker(self.tracker)
            self.reid_bank = ReIDBank(
                max_embeddings_per_track=50,
                match_thresh=cfg.reid_match_thresh,
                reid_model=reid_model,
            )

        if cfg.enable_minimap:
            cache = Path(cfg.homography_cache_file)
            if cache.exists():
                try:
                    cached = CachedHomography(
                        cache_path=str(cache),
                        interpolate=cfg.homography_interpolate)
                    self.homography_tracker = HomographyTracker(
                        estimator=cached, keyframe_interval=1,
                        smooth_window=cfg.homography_smooth_window)
                    self.minimap = Minimap(
                        map_width=cfg.minimap_width, show_title=False,
                        show_grass_pattern=True)
                except Exception as e:
                    self._minimap_warning = f"Гомография не загрузилась: {e}"
            else:
                self._minimap_warning = (
                    f"Не найден файл гомографии: {cache}. "
                    f"Мини-карта отключена (запусти precompute_homography.py).")

    def run(self) -> Iterator[FrameResult]:
        cfg = self.cfg
        self._minimap_warning = ""
        self._setup()

        Tracklet = self._mods["Tracklet"]
        bbox_to_world_position = self._mods["bbox_to_world_position"]
        draw_skeleton = self._mods["draw_skeleton"]

        cap = cv2.VideoCapture(str(cfg.input_video))
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {cfg.input_video}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = fps

        out_path = Path(cfg.output_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        frame_num = 0
        max_simultaneous = 0
        t_start = time.time()

        if self._minimap_warning:
            yield FrameResult(phase="tracking", frame_num=0,
                              total_frames=total, message=self._minimap_warning)

        try:
            while True:
                if self._stop.is_set():
                    break
                while self._pause.is_set() and not self._stop.is_set():
                    time.sleep(0.05)
                if self._stop.is_set():
                    break

                ret, frame = cap.read()
                if not ret:
                    break
                frame_num += 1

                results = self.model.predict(
                    frame, conf=cfg.confidence, iou=cfg.iou_thresh,
                    classes=[0], verbose=False)
                dets = []
                if results[0].boxes is not None and len(results[0].boxes):
                    for box in results[0].boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        if (xyxy[3] - xyxy[1]) >= cfg.min_bbox_height:
                            dets.append([*xyxy, conf, cls])
                dets = (np.array(dets, dtype=np.float32) if dets
                        else np.empty((0, 6), dtype=np.float32))

                tracked = self.tracker.update(dets, frame)

                h_frame, w_frame = frame.shape[:2]
                track_poses = self.pf_service.process_frame(
                    frame, frame_num, tracked, frame_size=(w_frame, h_frame))

                homo_result = None
                if self.homography_tracker is not None:
                    homo_result = self.homography_tracker.get(frame, frame_num)
                self._frames_total += 1
                if homo_result is not None:
                    self._frames_with_homo += 1

                tracks_info: List[dict] = []

                if len(tracked) > 0:
                    self.frame_track_data[frame_num] = []
                    for obj in tracked:
                        tid = int(obj[4])
                        box = obj[:4].copy()
                        conf_val = float(obj[5])
                        self.frame_track_data[frame_num].append(
                            [*box, tid, conf_val])

                        if tid not in self.tracklets:
                            self.tracklets[tid] = Tracklet(track_id=tid)
                        t = self.tracklets[tid]
                        t.frames.append(frame_num)
                        t.boxes.append(box.tolist())
                        t.poses.append(track_poses.get(tid))
                        world_pos = (bbox_to_world_position(box, homo_result)
                                     if homo_result is not None else None)
                        t.world_positions.append(world_pos)
                        self._world_attempts += 1
                        if world_pos is not None:
                            self._world_valid += 1

                        team_name = None
                        if self.team_classifier is not None:
                            _, team_name = self.team_classifier.get_team(tid)
                        tracks_info.append({
                            "id": tid,
                            "team": TEAM_LABELS.get(team_name, "-")
                                    if team_name else "-",
                            "conf": round(conf_val, 2),
                            "bbox": [int(v) for v in box[:4]],
                            "world": (round(world_pos[0], 1),
                                      round(world_pos[1], 1))
                                     if world_pos is not None else None,
                            "has_pose": track_poses.get(tid) is not None,
                            "speed": self._current_speed(t, fps),
                        })

                    if (self.reid_bank is not None
                            and frame_num % cfg.reid_collect_every == 0):
                        self.reid_bank.collect_from_frame(
                            frame, tracked, frame_num,
                            homography_result=homo_result)

                    team_results = None
                    if self.team_classifier is not None:
                        team_results = self.team_classifier.classify_batch(
                            frame, tracked)

                    self._draw_tracks(frame, tracked)

                    if team_results is not None:
                        from team_classifier import (
                            draw_team_labels, draw_team_stats)
                        frame = draw_team_labels(frame, tracked, team_results)
                        frame = draw_team_stats(frame, team_results)

                    if cfg.draw_skeletons and track_poses:
                        for tid, p in track_poses.items():
                            if p is None:
                                continue
                            skeleton_color = None
                            if self.team_classifier is not None:
                                try:
                                    _, tn = self.team_classifier.get_team(tid)
                                    if tn is not None:
                                        skeleton_color = (
                                            self.team_classifier
                                            .get_display_color(tn))
                                except Exception:
                                    skeleton_color = None
                            frame = draw_skeleton(
                                frame, p, color=skeleton_color,
                                conf_thresh=cfg.skeleton_kp_conf,
                                use_part_colors=(skeleton_color is None))

                    frame = self.pf_service.draw_foul_overlays(
                        frame, track_poses)
                    max_simultaneous = max(max_simultaneous, len(tracked))

                cv2.putText(frame,
                            f"On field: {len(tracked)} | "
                            f"Unique IDs: {len(self.trails)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame {frame_num}/{total}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                frame = self.pf_service.draw_event_panel_overlay(frame)

                minimap_img = None
                if (homo_result is not None and self.minimap is not None
                        and len(tracked) > 0):
                    minimap_img = self.minimap.render(
                        tracked, homo_result, self.team_classifier)

                save_frame = frame
                if cfg.burn_minimap_into_video and minimap_img is not None:
                    from minimap import overlay_minimap
                    save_frame = overlay_minimap(
                        frame.copy(), minimap_img,
                        position=cfg.minimap_position, margin=16)
                writer.write(save_frame)

                elapsed = time.time() - t_start
                proc_fps = frame_num / elapsed if elapsed > 0 else 0.0
                stats = {
                    "frame_num": frame_num, "total": total,
                    "tracked": int(len(tracked)),
                    "unique_ids": len(self.trails),
                    "max_simultaneous": max_simultaneous,
                    "proc_fps": round(proc_fps, 1),
                }

                yield FrameResult(
                    phase="tracking", frame_num=frame_num, total_frames=total,
                    display_frame=frame, minimap_img=minimap_img,
                    tracks=tracks_info, events=self._collect_events(),
                    stats=stats)
        finally:
            cap.release()
            writer.release()

        self._snapshot_raw_tracks()
        self._id_map = {tid: tid for tid in self.tracklets}

        try:
            self.pf_service.finalize()
        except Exception:
            pass

        stopped = self._stop.is_set()
        if cfg.enable_merge and len(self.tracklets) > 0:
            msg_suffix = " (обработанная часть)" if stopped else ""
            yield FrameResult(phase="postprocess", frame_num=frame_num,
                              total_frames=total,
                              message=f"Пост-обработка треков{msg_suffix}...")
            self._id_map = self._run_merge(fps)
            merged_path = Path(cfg.merged_video)
            merged_path.parent.mkdir(parents=True, exist_ok=True)

            for mf, mt in self._mods["apply_merge_to_video"](
                    cfg.input_video, str(merged_path),
                    self.frame_track_data, self._id_map, fps,
                    team_classifier=self.team_classifier,
                    end_frame=frame_num):
                pct = int(100 * mf / max(1, mt))
                yield FrameResult(
                    phase="postprocess", frame_num=mf, total_frames=mt,
                    stats={"merge_frame": mf, "merge_total": mt},
                    message=f"Сохранение merged-видео{msg_suffix}… {pct}%")

        if stopped:
            done_msg = (f"Остановлено на кадре {frame_num}. "
                        f"Видео сохранено до этой точки.")
        else:
            done_msg = "Готово."
        yield FrameResult(
            phase="done", frame_num=frame_num, total_frames=total,
            stats={"frame_num": frame_num, "total": total},
            message=done_msg)

    def _draw_tracks(self, frame, tracked_objects):
        cfg = self.cfg
        for obj in tracked_objects:
            x1, y1, x2, y2 = map(int, obj[:4])
            tid = int(obj[4])
            conf = obj[5]
            color = _color_for_id(tid, self.team_classifier)

            cx, cy = (x1 + x2) // 2, y2
            self.trails.setdefault(tid, []).append((cx, cy))
            if len(self.trails[tid]) > cfg.trail_length:
                self.trails[tid] = self.trails[tid][-cfg.trail_length:]
            pts = self.trails[tid]
            for i in range(1, len(pts)):
                thickness = max(1, int(3 * (i / len(pts))))
                cv2.line(frame, pts[i - 1], pts[i], color, thickness)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cfg.bbox_thickness)
            label = f"ID {tid} ({conf:.2f})"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1),
                          color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale,
                        (255, 255, 255), 2)
        return frame

    @staticmethod
    def _current_speed(tracklet, fps, max_speed=12.0) -> Optional[float]:
        valid = [(f, p) for f, p in zip(tracklet.frames,
                                        tracklet.world_positions)
                 if p is not None]
        if len(valid) < 2:
            return None
        (f0, p0), (f1, p1) = valid[-2], valid[-1]
        df = f1 - f0
        if df <= 0:
            return None
        v = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) / (df / fps)
        if v > max_speed:
            return None
        return round(v, 1)

    def _collect_events(self) -> List[dict]:
        if (self.pf_service is None
                or not self.pf_service.has_foul_detector):
            return []
        out = []
        try:
            for ev in self.pf_service.foul_detector.get_active_events():
                out.append({
                    "kind": ev.kind, "attacker_id": ev.attacker_id,
                    "victim_id": ev.victim_id,
                    "peak_score": round(float(ev.peak_score), 2),
                    "start_frame": ev.start_frame, "end_frame": ev.end_frame})
        except Exception:
            pass
        return out

    def all_events(self) -> List[dict]:
        if (self.pf_service is None
                or not self.pf_service.has_foul_detector):
            return []
        out = []
        try:
            for ev in self.pf_service.foul_detector.get_all_events():
                out.append({
                    "kind": ev.kind, "attacker_id": ev.attacker_id,
                    "victim_id": ev.victim_id,
                    "peak_score": round(float(ev.peak_score), 2),
                    "avg_score": round(float(ev.avg_score), 2),
                    "start_frame": ev.start_frame, "end_frame": ev.end_frame})
        except Exception:
            pass
        return out

    def _run_merge(self, fps) -> dict:
        cfg = self.cfg
        TrackletMerger = self._mods["TrackletMerger"]
        merger = TrackletMerger(
            max_frame_gap=cfg.max_frame_gap,
            max_spatial_dist=cfg.max_spatial_dist,
            merge_cost_thresh=cfg.merge_cost_thresh,
            team_classifier=self.team_classifier,
            reid_bank=self.reid_bank, fps=fps,
            max_player_speed_ms=cfg.max_player_speed_ms)
        return merger.merge(self.tracklets)

    def _snapshot_raw_tracks(self):
        self._raw_tracks = {}
        for tid, t in self.tracklets.items():
            self._raw_tracks[tid] = {
                "frames": list(t.frames),
                "world": list(t.world_positions),
                "pose": [p is not None for p in t.poses],
            }

    def compute_analytics(self) -> dict:
        fps = self.fps or 30.0
        raw = self._raw_tracks or {}
        id_map = self._id_map or {tid: tid for tid in raw}

        def resolve(tid):
            seen = set()
            cur = tid
            while id_map.get(cur, cur) != cur and cur not in seen:
                seen.add(cur)
                cur = id_map[cur]
            return cur

        groups: Dict[int, List[int]] = defaultdict(list)
        for tid in raw:
            groups[resolve(tid)].append(tid)

        players = []
        for final_id, members in sorted(groups.items()):
            frames_all, world_all, pose_all = [], [], []
            for tid in members:
                rt = raw[tid]
                frames_all.extend(rt["frames"])
                world_all.extend(rt["world"])
                pose_all.extend(rt["pose"])

            if not frames_all:
                continue
            dist, max_v = self._distance_and_max_speed(
                frames_all, world_all, fps)
            span = max(frames_all) - min(frames_all) + 1
            time_s = span / fps
            avg_v = dist / time_s if time_s > 0 else 0.0
            pose_cov = (100.0 * sum(pose_all) / len(pose_all)
                        if pose_all else 0.0)

            team_label = "-"
            if self.team_classifier is not None:
                for tid in members:
                    _, tn = self.team_classifier.get_team(tid)
                    if tn:
                        team_label = TEAM_LABELS.get(tn, tn)
                        break

            players.append({
                "id": int(final_id), "team": team_label,
                "frames": len(frames_all),
                "time_s": round(time_s, 1),
                "distance_m": round(dist, 1),
                "avg_speed_ms": round(avg_v, 2),
                "max_speed_ms": round(max_v, 2),
                "pose_coverage_pct": round(pose_cov, 1),
            })

        team_agg: Dict[str, dict] = defaultdict(
            lambda: {"players": 0, "distance_m": 0.0})
        for p in players:
            t = team_agg[p["team"]]
            t["players"] += 1
            t["distance_m"] += p["distance_m"]
        teams = []
        for name, agg in team_agg.items():
            teams.append({
                "team": name, "players": agg["players"],
                "total_distance_m": round(agg["distance_m"], 1),
                "avg_distance_m": round(
                    agg["distance_m"] / max(1, agg["players"]), 1),
            })

        quality = {
            "frames_total": self._frames_total,
            "homography_coverage_pct": round(
                100.0 * self._frames_with_homo / max(1, self._frames_total), 1),
            "world_valid_pct": round(
                100.0 * self._world_valid / max(1, self._world_attempts), 1),
            "ids_before_merge": len(raw),
            "ids_after_merge": len(groups),
        }

        return {
            "fps": round(fps, 2),
            "players": players,
            "teams": teams,
            "events": self.all_events(),
            "quality": quality,
        }

    @staticmethod
    def _distance_and_max_speed(frames, world, fps,
                                 max_speed=12.0) -> Tuple[float, float]:
        pts = sorted(((f, p) for f, p in zip(frames, world) if p is not None),
                     key=lambda x: x[0])
        dist = 0.0
        max_v = 0.0
        for (f0, p0), (f1, p1) in zip(pts, pts[1:]):
            df = f1 - f0
            if df <= 0:
                continue
            d = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            v = d / (df / fps)
            if v > max_speed:
                continue
            dist += d
            max_v = max(max_v, v)
        return dist, max_v

def precompute_homography_iter(video_path, output_path, pnlcalib_repo,
                                kp_weights, line_weights, device=None,
                                keyframe_interval=30, start=0, end=-1,
                                stop_event=None):
    from pnlcalib_estimator import PnLCalibHomography

    estimator = PnLCalibHomography(
        repo_path=pnlcalib_repo,
        kp_weights=kp_weights,
        line_weights=line_weights,
        device=device,
        verbose=False,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        yield {"phase": "error",
               "message": f"Не удалось открыть видео: {video_path}"}
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = total if end < 0 else min(end, total)

    frames_list, matrices_list, conf_list, failed = [], [], [], []
    frame_num = start - 1
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            if frame_num >= end_frame:
                break
            if (frame_num - start) % keyframe_interval != 0:
                continue

            result = estimator.estimate(frame, frame_num)
            if result is not None:
                frames_list.append(frame_num)
                matrices_list.append(result.matrix)
                conf_list.append(result.confidence)
            else:
                failed.append(frame_num)

            cov = 100.0 * len(frames_list) / max(1, len(frames_list) + len(failed))
            yield {"phase": "compute", "frame": frame_num, "total": end_frame,
                   "ok": len(frames_list), "failed": len(failed),
                   "coverage": round(cov, 1)}
    finally:
        cap.release()

    if not frames_list:
        yield {"phase": "error",
               "message": ("Ни одного успешного кадра. Проверь, что PnLCalib "
                           "и веса (SV_kp / SV_lines) работают на этом видео.")}
        return

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out),
        frames=np.asarray(frames_list, dtype=np.int64),
        matrices=np.asarray(matrices_list, dtype=np.float64),
        confidences=np.asarray(conf_list, dtype=np.float32),
        method="pnlcalib",
        video_path=str(video_path),
        keyframe_interval=keyframe_interval,
        failed_frames=np.asarray(failed, dtype=np.int64),
    )
    cov = 100.0 * len(frames_list) / max(1, len(frames_list) + len(failed))
    yield {"phase": "done", "output": str(out), "ok": len(frames_list),
           "failed": len(failed), "coverage": round(cov, 1)}
