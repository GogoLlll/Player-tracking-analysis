from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np

from pose_estimator import PoseEstimator, PoseDetection, PoseSmoother, match_poses_to_tracks
from interaction_detector import InteractionDetector, set_debug_mode
from foul_visualization import (
    draw_foul_visualization, draw_event_panel,
)
from ml_classifier import (
    FoulClassifier, TrainingDataWriter,
)

@dataclass
class PoseFoulsConfig:

    enable_pose: bool = True
    pose_model_name: str = "models/yolo11l-pose.pt"
    pose_conf: float = 0.25
    pose_iou: float = 0.5
    pose_match_iou: float = 0.4

    enable_pose_smoothing: bool = True
    pose_smooth_alpha: float = 0.5
    pose_smooth_min_conf: float = 0.3

    draw_skeletons: bool = True

    enable_foul_detection: bool = True
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

class PoseAndFoulsService:

    def __init__(self, cfg: PoseFoulsConfig):
        self.cfg = cfg

        self.pose_estimator: Optional[PoseEstimator] = None
        if cfg.enable_pose:
            try:
                self.pose_estimator = PoseEstimator(
                    model_name=cfg.pose_model_name,
                    conf=cfg.pose_conf,
                    iou=cfg.pose_iou,
                )
                print(f"[PoseFoul] Pose estimator: {cfg.pose_model_name}")
            except Exception as e:
                print(f"[PoseFoul] Не удалось загрузить pose model: {e}")
                self.pose_estimator = None

        self.pose_smoother: Optional[PoseSmoother] = None
        if self.pose_estimator is not None and cfg.enable_pose_smoothing:
            self.pose_smoother = PoseSmoother(
                alpha=cfg.pose_smooth_alpha,
                min_conf=cfg.pose_smooth_min_conf,
            )
            print(f"[PoseFoul] Pose smoothing: "
                  f"alpha={cfg.pose_smooth_alpha}, "
                  f"min_conf={cfg.pose_smooth_min_conf}")

        self.foul_classifier: Optional[FoulClassifier] = None
        self.foul_training_writer: Optional[TrainingDataWriter] = None
        self.foul_detector: Optional[InteractionDetector] = None

        if self.pose_estimator is not None and cfg.enable_foul_detection:

            if cfg.enable_ml_classifier:
                self.foul_classifier = FoulClassifier(
                    reject_threshold=cfg.ml_reject_threshold,
                    ensemble_alpha=cfg.ml_ensemble_alpha,
                )
                if self.foul_classifier.load(cfg.ml_model_path):
                    print(f"[PoseFoul] ML classifier: {cfg.ml_model_path}")
                    print(f"  reject_threshold="
                          f"{cfg.ml_reject_threshold}, "
                          f"ensemble_alpha={cfg.ml_ensemble_alpha}")
                else:
                    print(f"[PoseFoul] ML модель не загружена - "
                          f"продолжаем без ML")
                    self.foul_classifier = None

            if cfg.enable_training_export:
                self.foul_training_writer = TrainingDataWriter(
                    cfg.training_data_csv
                )
                print(f"[PoseFoul] Training export → "
                      f"{cfg.training_data_csv}")

            if cfg.enable_foul_debug:
                set_debug_mode(True)
                print(f"[PoseFoul] Foul debug включён")

            self.foul_detector = InteractionDetector(
                window_size=cfg.foul_window_size,
                min_hits=cfg.foul_min_hits,
                cooldown=cfg.foul_cooldown,
                classifier=self.foul_classifier,
                training_data_writer=self.foul_training_writer,
            )
            print(f"[PoseFoul] Foul detector: "
                  f"window={cfg.foul_window_size}, "
                  f"min_hits={cfg.foul_min_hits}, "
                  f"cooldown={cfg.foul_cooldown}")

        self._pose_stats = {
            "frames_processed": 0,
            "total_poses_found": 0,
            "total_matched": 0,
            "total_drawn": 0,
        }

    @property
    def enabled(self) -> bool:
        return (self.pose_estimator is not None
                or self.foul_detector is not None)

    @property
    def has_pose(self) -> bool:
        return self.pose_estimator is not None

    @property
    def has_foul_detector(self) -> bool:
        return self.foul_detector is not None

    def process_frame(
        self,
        frame: np.ndarray,
        frame_num: int,
        tracked: np.ndarray,
        frame_size: Tuple[int, int],
    ) -> Dict[int, Optional[PoseDetection]]:
        track_poses: Dict[int, Optional[PoseDetection]] = {}

        if self.pose_estimator is None or len(tracked) == 0:
            return track_poses

        poses = self.pose_estimator.predict_frame(frame)
        n_poses_raw = len(poses)
        track_poses = match_poses_to_tracks(
            tracked, poses, iou_thresh=self.cfg.pose_match_iou
        )
        n_matched = sum(1 for p in track_poses.values() if p is not None)
        self._pose_stats["frames_processed"] += 1
        self._pose_stats["total_poses_found"] += n_poses_raw
        self._pose_stats["total_matched"] += n_matched

        if self.pose_smoother is not None and track_poses:
            track_poses = self.pose_smoother.smooth(frame_num, track_poses)

        if self.foul_detector is not None and track_poses:
            try:
                self.foul_detector.update(
                    frame_num, track_poses, frame_size=frame_size
                )
            except TypeError:

                self.foul_detector.update(frame_num, track_poses)

        return track_poses

    def draw_foul_overlays(
        self,
        frame: np.ndarray,
        track_poses: Dict[int, Optional[PoseDetection]],
    ) -> np.ndarray:
        if self.foul_detector is None:
            return frame

        active_events = self.foul_detector.get_active_events()
        for ev in active_events:
            if ev.last_signal is None:
                continue
            pose_a = track_poses.get(ev.attacker_id)
            pose_v = track_poses.get(ev.victim_id)
            if pose_a is None and pose_v is None:
                continue
            frame = draw_foul_visualization(
                frame, ev.last_signal, pose_a, pose_v
            )
        return frame

    def draw_event_panel_overlay(self, frame: np.ndarray) -> np.ndarray:
        if self.foul_detector is None:
            return frame
        active = self.foul_detector.get_active_events()
        all_evs = self.foul_detector.get_all_events()
        if not active and not all_evs:
            return frame
        return draw_event_panel(frame, active, all_evs)

    def draw_overlays(self, frame, track_poses):
        frame = self.draw_foul_overlays(frame, track_poses)
        frame = self.draw_event_panel_overlay(frame)
        return frame

    def finalize(self) -> None:

        if self.pose_estimator is not None \
                and self._pose_stats["frames_processed"] > 0:
            ps = self._pose_stats
            n_proc = ps["frames_processed"]
            avg_poses = ps["total_poses_found"] / n_proc
            avg_matched = ps["total_matched"] / n_proc
            print(f"\n{'='*50}")
            print(f"POSE STATISTICS")
            print(f"{'='*50}")
            print(f"  Кадров обработано: {n_proc}")
            print(f"  Поз найдено всего: {ps['total_poses_found']} "
                  f"(в среднем {avg_poses:.1f}/кадр)")
            print(f"  Сопоставлено с треками: {ps['total_matched']} "
                  f"(в среднем {avg_matched:.1f}/кадр)")
            print(f"  Скелетов отрисовано: {ps['total_drawn']}")

        if self.foul_detector is not None:
            all_events = self.foul_detector.get_all_events()
            print(f"\n{'='*50}")
            print(f"FOUL EVENTS")
            print(f"{'='*50}")
            if all_events:
                by_kind = defaultdict(list)
                for ev in all_events:
                    by_kind[ev.kind].append(ev)
                total = len(all_events)
                print(f"  Всего нарушений: {total}")
                for kind in sorted(by_kind):
                    evs = by_kind[kind]
                    print(f"\n  [{kind}] - {len(evs)} событий:")
                    for ev in evs:
                        print(f"    кадры {ev.start_frame}-{ev.end_frame} "
                              f"({ev.end_frame - ev.start_frame + 1}f) "
                              f"A{ev.attacker_id}->V{ev.victim_id} "
                              f"peak={ev.peak_score:.2f} "
                              f"avg={ev.avg_score:.2f}")
            else:
                print(f"  Нарушений не зафиксировано")

            if self.foul_training_writer is not None:
                print(f"\n[ML] Training data: "
                      f"{len(self.foul_training_writer)} событий")
                self.foul_training_writer.save()

            if self.foul_classifier is not None \
                    and self.foul_classifier.is_loaded():
                print(f"\n[ML] Использован классификатор: "
                      f"{self.cfg.ml_model_path}")
                fi = self.foul_classifier.get_feature_importance()
                if fi:
                    print(f"[ML] Top-5 features by importance:")
                    for name, val in fi[:5]:
                        print(f"  {name:30s} {val:>10.1f}")
