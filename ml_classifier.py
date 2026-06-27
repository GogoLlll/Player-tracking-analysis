from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

FEATURE_NAMES: List[str] = [

    "is_slide_tackle",
    "is_arm_strike",

    "contact_norm",
    "contact_zone_legs",
    "contact_zone_torso",
    "contact_zone_head",
    "contact_zone_arms",

    "descent",
    "hip_drop_norm",
    "torso_delta_deg",
    "bbox_compression",
    "bbox_height_ratio",
    "frames_since_contact",
    "zoom_compensated",

    "attacker_torso_angle",
    "attacker_height",
    "attacker_wrist_speed",
    "attacker_leg_outreach",

    "victim_torso_baseline",
    "victim_height_baseline",
]
NUM_FEATURES = len(FEATURE_NAMES)

_KP_HEAD = {0, 1, 2, 3, 4}
_KP_ARMS = {7, 8, 9, 10}
_KP_TORSO = {5, 6, 11, 12}
_KP_LEGS = {13, 14, 15, 16}

def _zone_from_kp(kp_idx: int) -> str:
    if kp_idx in _KP_HEAD:
        return "head"
    if kp_idx in _KP_ARMS:
        return "arms"
    if kp_idx in _KP_TORSO:
        return "torso"
    if kp_idx in _KP_LEGS:
        return "legs"
    return "torso"

@dataclass
class FoulCandidate:

    kind: str

    contact_norm: float
    contact_bone_attacker_kp: int = -1
    contact_bone_victim_kp: int = -1

    descent: float = 0.0
    hip_drop_norm: float = 0.0
    torso_delta_deg: float = 0.0
    bbox_compression: float = 0.0
    bbox_height_ratio: float = 1.0
    frames_since_contact: int = 0
    zoom_compensated: float = 1.0

    attacker_torso_angle: float = 0.0
    attacker_height: float = 0.0
    attacker_wrist_speed: float = 0.0
    attacker_leg_outreach: float = 0.0

    victim_torso_baseline: float = 0.0
    victim_height_baseline: float = 0.0

class FeatureExtractor:

    @staticmethod
    def extract(candidate: FoulCandidate) -> np.ndarray:
        kind = candidate.kind

        is_slide = 1.0 if kind.startswith("slide_tackle") else 0.0
        is_arm = 1.0 if kind.startswith("arm_strike") else 0.0

        zone = _zone_from_kp(candidate.contact_bone_victim_kp)
        z_legs = 1.0 if zone == "legs" else 0.0
        z_torso = 1.0 if zone == "torso" else 0.0
        z_head = 1.0 if zone == "head" else 0.0
        z_arms = 1.0 if zone == "arms" else 0.0

        v_h_norm = candidate.victim_height_baseline / 300.0
        a_h_norm = candidate.attacker_height / 300.0

        feats = np.array([
            is_slide,
            is_arm,
            candidate.contact_norm,
            z_legs,
            z_torso,
            z_head,
            z_arms,
            candidate.descent,
            candidate.hip_drop_norm,
            candidate.torso_delta_deg / 90.0,
            candidate.bbox_compression,
            candidate.bbox_height_ratio,
            candidate.frames_since_contact / 90.0,
            candidate.zoom_compensated,
            candidate.attacker_torso_angle / 90.0,
            a_h_norm,
            candidate.attacker_wrist_speed,
            candidate.attacker_leg_outreach,
            candidate.victim_torso_baseline / 90.0,
            v_h_norm,
        ], dtype=np.float32)

        feats = np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0)
        return feats

    @staticmethod
    def to_dict(candidate: FoulCandidate) -> Dict[str, float]:
        vec = FeatureExtractor.extract(candidate)
        return {name: float(v) for name, v in zip(FEATURE_NAMES, vec)}

class FoulClassifier:

    def __init__(self,
                 reject_threshold: float = 0.3,
                 ensemble_alpha: float = 0.5):
        self.model = None
        self.model_path: Optional[str] = None
        self.reject_threshold = float(reject_threshold)
        self.ensemble_alpha = float(ensemble_alpha)
        self._feature_names_used: Optional[List[str]] = None

    def load(self, path: str) -> bool:
        path = str(path)
        if not Path(path).exists():
            print(f"[FoulClassifier] Файл модели не найден: {path}")
            return False

        try:
            import lightgbm as lgb
        except ImportError:
            print("[FoulClassifier] lightgbm не установлен; "
                  "pip install lightgbm")
            return False

        try:
            if path.endswith(".pkl"):
                import pickle
                with open(path, "rb") as f:
                    self.model = pickle.load(f)
            else:
                self.model = lgb.Booster(model_file=path)

            try:
                model_features = self.model.feature_name()
                if model_features and model_features != FEATURE_NAMES:
                    diff_new = set(FEATURE_NAMES) - set(model_features)
                    diff_old = set(model_features) - set(FEATURE_NAMES)
                    print(f"[FoulClassifier] ВНИМАНИЕ: имена фичей модели "
                          f"не совпадают с ожидаемыми.")
                    if diff_new:
                        print(f"  Новые (в коде, нет в модели): {diff_new}")
                    if diff_old:
                        print(f"  Старые (в модели, нет в коде): {diff_old}")
                    print(f"  Модель будет использоваться "
                          f"по позиционному соответствию.")
                self._feature_names_used = model_features
            except Exception:
                pass

            self.model_path = path
            print(f"[FoulClassifier] Загружена модель: {path}")
            return True
        except Exception as e:
            print(f"[FoulClassifier] Ошибка загрузки {path}: "
                  f"{type(e).__name__}: {e}")
            self.model = None
            return False

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, features: np.ndarray
                ) -> Tuple[Optional[float], Optional[int]]:
        if self.model is None:
            return None, None

        try:

            if features.ndim == 1:
                features = features.reshape(1, -1)

            prob = float(self.model.predict(features)[0])
            label = int(prob >= 0.5)
            return prob, label
        except Exception as e:
            print(f"[FoulClassifier] predict failed: "
                  f"{type(e).__name__}: {e}")
            return None, None

    def apply(self,
              candidate: FoulCandidate,
              rule_score: float
              ) -> Tuple[bool, float, Optional[float]]:
        if self.model is None:
            return True, rule_score, None

        feats = FeatureExtractor.extract(candidate)
        prob, _ = self.predict(feats)
        if prob is None:

            return True, rule_score, None

        if prob < self.reject_threshold:
            return False, 0.0, prob

        final = (self.ensemble_alpha * rule_score +
                 (1 - self.ensemble_alpha) * prob)
        final = float(np.clip(final, 0.0, 1.0))
        return True, final, prob

    def get_feature_importance(self,
                                importance_type: str = "gain"
                                ) -> Optional[List[Tuple[str, float]]]:
        if self.model is None:
            return None
        try:
            values = self.model.feature_importance(
                importance_type=importance_type
            )
            names = self._feature_names_used or FEATURE_NAMES
            pairs = list(zip(names, values.tolist() if hasattr(values, 'tolist')
                              else list(values)))
            return sorted(pairs, key=lambda x: x[1], reverse=True)
        except Exception:
            return None

class TrainingDataWriter:

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.rows: List[Dict] = []

    def add(self,
            candidate: FoulCandidate,
            rule_score: float,
            video_path: str = "",
            frame_num: int = -1,
            attacker_id: int = -1,
            victim_id: int = -1,
            label: int = -1) -> None:
        row = FeatureExtractor.to_dict(candidate)
        row.update({
            "_video": video_path,
            "_frame": frame_num,
            "_attacker_id": attacker_id,
            "_victim_id": victim_id,
            "_kind": candidate.kind,
            "_rule_score": float(rule_score),
            "_label": int(label),
        })
        self.rows.append(row)

    def save(self) -> None:
        if not self.rows:
            print(f"[TrainingDataWriter] Нет данных для сохранения")
            return

        import csv

        meta_cols = ["_video", "_frame", "_attacker_id", "_victim_id",
                     "_kind", "_rule_score", "_label"]
        cols = FEATURE_NAMES + meta_cols

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({k: row.get(k, "") for k in cols})

        print(f"[TrainingDataWriter] Сохранено {len(self.rows)} строк "
              f"в {self.output_path}")

    def __len__(self) -> int:
        return len(self.rows)
