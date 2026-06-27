import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

@dataclass
class Tracklet:
    track_id: int
    frames: list = field(default_factory=list)
    boxes: list = field(default_factory=list)
    embeddings: list = field(default_factory=list)

    poses: list = field(default_factory=list)

    world_positions: list = field(default_factory=list)

    @property
    def start_frame(self):
        return self.frames[0] if self.frames else 0

    @property
    def end_frame(self):
        return self.frames[-1] if self.frames else 0

    def last_world_position(self):
        for pos in reversed(self.world_positions):
            if pos is not None:
                return pos
        return None

    def first_world_position(self):
        for pos in self.world_positions:
            if pos is not None:
                return pos
        return None

    def end_velocity_mps(self, fps=30.0, k=10):
        return self._velocity_from(self.world_positions, self.frames,
                                    fps=fps, k=k, from_end=True)

    def start_velocity_mps(self, fps=30.0, k=10):
        return self._velocity_from(self.world_positions, self.frames,
                                    fps=fps, k=k, from_end=False)

    @staticmethod
    def _velocity_from(world_positions, frames, fps, k, from_end):
        if not world_positions or not frames:
            return None

        valid = [(fr, p) for fr, p in zip(frames, world_positions)
                 if p is not None]
        if len(valid) < 2:
            return None
        seq = valid[-k:] if from_end else valid[:k]
        if len(seq) < 2:
            return None

        first_fr, first_pos = seq[0]
        last_fr, last_pos = seq[-1]
        dframes = last_fr - first_fr
        if dframes <= 0:
            return None
        vx = (last_pos[0] - first_pos[0]) / dframes * fps
        vy = (last_pos[1] - first_pos[1]) / dframes * fps
        return np.array([vx, vy], dtype=np.float64)

    @property
    def duration(self):
        return self.end_frame - self.start_frame + 1

    @property
    def start_pos(self):
        if not self.boxes:
            return np.zeros(2)
        b = self.boxes[0]
        return np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])

    @property
    def end_pos(self):
        if not self.boxes:
            return np.zeros(2)
        b = self.boxes[-1]
        return np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])

    @property
    def end_velocity(self):
        if len(self.boxes) < 3:
            return np.zeros(2)
        positions = []
        for b in self.boxes[-5:]:
            positions.append(np.array([(b[0]+b[2])/2, (b[1]+b[3])/2]))
        velocities = [positions[i+1] - positions[i]
                      for i in range(len(positions)-1)]
        return np.mean(velocities, axis=0)

    @property
    def start_velocity(self):
        if len(self.boxes) < 3:
            return np.zeros(2)
        positions = []
        for b in self.boxes[:5]:
            positions.append(np.array([(b[0]+b[2])/2, (b[1]+b[3])/2]))
        velocities = [positions[i+1] - positions[i]
                      for i in range(len(positions)-1)]
        return np.mean(velocities, axis=0)

    @property
    def avg_size(self):
        if not self.boxes:
            return np.zeros(2)
        sizes = [(b[2]-b[0], b[3]-b[1]) for b in self.boxes]
        return np.mean(sizes, axis=0)

    @property
    def mean_embedding(self):
        if not self.embeddings:
            return None
        valid = [e for e in self.embeddings if e is not None]
        if not valid:
            return None
        return np.mean(valid, axis=0)

class ReIDBank:

    def __init__(self, max_embeddings_per_track=50, match_thresh=0.75,
                 reid_model=None, fps=30.0):
        self.max_per_track = max_embeddings_per_track
        self.match_thresh = match_thresh
        self.fps = float(fps)
        self.reid_model = reid_model

        self.mode = "osnet" if reid_model is not None else "histogram"

        self.bank = defaultdict(list)

        self.track_lifespan = {}

        self.last_world_pos: Dict[int, Optional[Tuple[float, float]]] = {}

        self.last_world_vel: Dict[int, Optional[Tuple[float, float]]] = {}

        self._prev_world_pos: Dict[int, Tuple[float, float]] = {}
        self._prev_world_frame: Dict[int, int] = {}
        self._last_world_frame: Dict[int, int] = {}

        self._extract_failures = 0

    def collect(self, track_id, frame_num, crop, reid_model=None,
                embedding=None):
        if embedding is not None:
            emb = embedding
        elif reid_model is not None and crop is not None:
            emb = self._extract_embedding(crop, reid_model)
            if emb is None:
                return
        else:
            return

        entries = self.bank[track_id]
        if len(entries) >= self.max_per_track:

            mid = len(entries) // 2
            entries.pop(mid)

        entries.append((frame_num, emb))

        if track_id not in self.track_lifespan:
            self.track_lifespan[track_id] = (frame_num, frame_num)
        else:
            first, last = self.track_lifespan[track_id]
            self.track_lifespan[track_id] = (min(first, frame_num),
                                              max(last, frame_num))

    def collect_from_frame(self, frame, tracked_objects, frame_num,
                            reid_model=None,
                            homography_result=None):

        model = reid_model if reid_model is not None else self.reid_model

        if len(tracked_objects) == 0:
            return

        if homography_result is not None:

            try:
                from field_homography import bbox_to_world_position
                for obj in tracked_objects:
                    tid = int(obj[4])
                    pos = bbox_to_world_position(
                        obj[:4], homography_result
                    )
                    if pos is None:
                        continue

                    prev_pos = self._prev_world_pos.get(tid)
                    prev_fr = self._prev_world_frame.get(tid)
                    if prev_pos is not None and prev_fr is not None:
                        dframes = frame_num - prev_fr
                        if dframes > 0:

                            fps = getattr(self, "fps", 30.0)
                            vx = (pos[0] - prev_pos[0]) / dframes * fps
                            vy = (pos[1] - prev_pos[1]) / dframes * fps

                            speed = float(np.hypot(vx, vy))
                            if speed < 30.0:
                                self.last_world_vel[tid] = (vx, vy)

                    self._prev_world_pos[tid] = pos
                    self._prev_world_frame[tid] = frame_num
                    self.last_world_pos[tid] = pos
                    self._last_world_frame[tid] = frame_num
            except ImportError:
                pass

        h, w = frame.shape[:2]

        if model is not None:

            xyxys = []
            tids = []
            for obj in tracked_objects:
                x1, y1, x2, y2 = map(int, obj[:4])
                tid = int(obj[4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 - x1 < 10 or y2 - y1 < 20:
                    continue
                xyxys.append([x1, y1, x2, y2])
                tids.append(tid)

            if not xyxys:
                return

            xyxys_np = np.array(xyxys, dtype=np.float32)
            embs = self._extract_embeddings_batch(xyxys_np, frame, model)

            if embs is None:

                self._extract_failures += 1
                return

            for tid, emb in zip(tids, embs):
                if emb is not None:
                    self.collect(tid, frame_num, None, embedding=emb)
            return

        for obj in tracked_objects:
            x1, y1, x2, y2 = map(int, obj[:4])
            tid = int(obj[4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 10 or y2 - y1 < 20:
                continue

            crop = frame[y1:y2, x1:x2]
            emb = self._color_histogram(crop)
            if emb is not None:
                self.collect(tid, frame_num, None, embedding=emb)

    def get_mean_embedding(self, track_id):
        entries = self.bank.get(track_id, [])
        if not entries:
            return None
        embeddings = [e for _, e in entries]
        return np.mean(embeddings, axis=0)

    def get_recent_embedding(self, track_id, n=10):
        entries = self.bank.get(track_id, [])
        if not entries:
            return None
        recent = [e for _, e in entries[-n:]]
        return np.mean(recent, axis=0)

    def find_match(self, track_id, max_frame_gap=300,
                    team_classifier=None,
                    fps: float = 30.0,
                    max_player_speed_ms: float = 15.0):
        emb = self.get_mean_embedding(track_id)
        if emb is None:
            return None, 0

        lifespan = self.track_lifespan.get(track_id)
        if lifespan is None:
            return None, 0

        my_start = lifespan[0]
        my_world_pos = self.last_world_pos.get(track_id)
        best_id = None
        best_sim = self.match_thresh

        for other_id, other_entries in self.bank.items():
            if other_id == track_id:
                continue

            other_lifespan = self.track_lifespan.get(other_id)
            if other_lifespan is None:
                continue

            other_end = other_lifespan[1]
            gap = my_start - other_end
            if gap < 0 or gap > max_frame_gap:
                continue

            if team_classifier is not None:
                if not team_classifier.same_team(track_id, other_id):
                    continue

            other_world_pos = self.last_world_pos.get(other_id)
            world_prediction_bonus = 0.0
            if my_world_pos is not None and other_world_pos is not None:
                gap_seconds = max(1, gap) / fps
                max_dist = max_player_speed_ms * gap_seconds
                dx = my_world_pos[0] - other_world_pos[0]
                dy = my_world_pos[1] - other_world_pos[1]
                dist_m = float(np.hypot(dx, dy))
                if dist_m > max_dist:
                    continue

                other_vel = self.last_world_vel.get(other_id)
                if other_vel is not None:
                    pred_x = other_world_pos[0] + other_vel[0] * gap_seconds
                    pred_y = other_world_pos[1] + other_vel[1] * gap_seconds
                    pred_dist = float(np.hypot(
                        pred_x - my_world_pos[0],
                        pred_y - my_world_pos[1]
                    ))

                    if pred_dist < 5.0:
                        world_prediction_bonus = 0.10
                    elif pred_dist < 15.0:

                        world_prediction_bonus = 0.10 * (1.0 - (pred_dist - 5.0) / 10.0)

            other_emb = self.get_recent_embedding(other_id)
            if other_emb is None:
                continue

            sim = np.dot(emb, other_emb) / (
                np.linalg.norm(emb) * np.linalg.norm(other_emb) + 1e-6
            )

            adjusted_sim = sim + world_prediction_bonus

            if adjusted_sim > best_sim:
                best_sim = adjusted_sim
                best_id = other_id

        return best_id, best_sim

    def get_merge_map(self, tracklets, max_frame_gap=300,
                       team_classifier=None,
                       fps: float = 30.0,
                       max_player_speed_ms: float = 15.0):

        sorted_ids = sorted(
            self.track_lifespan.keys(),
            key=lambda tid: self.track_lifespan[tid][0]
        )

        merge_map = {}
        already_matched = set()
        resurrections = 0

        for tid in sorted_ids:
            if tid in already_matched:
                continue

            match_id, sim = self.find_match(
                tid, max_frame_gap, team_classifier,
                fps=fps, max_player_speed_ms=max_player_speed_ms,
            )

            if match_id is not None and match_id not in already_matched:
                merge_map[tid] = match_id
                already_matched.add(tid)
                already_matched.add(match_id)
                resurrections += 1

        if resurrections > 0:
            print(f"[ReID BANK] Воскрешено треков: {resurrections}")
            for new_id, old_id in merge_map.items():
                old_span = self.track_lifespan[old_id]
                new_span = self.track_lifespan[new_id]
                gap = new_span[0] - old_span[1]
                sim = self._similarity(old_id, new_id)
                print(f"  ID {new_id} (frame {new_span[0]}) → "
                      f"ID {old_id} (ended frame {old_span[1]}) "
                      f"gap={gap} frames, similarity={sim:.2f}")

        return merge_map

    def _similarity(self, tid_a, tid_b):
        emb_a = self.get_mean_embedding(tid_a)
        emb_b = self.get_mean_embedding(tid_b)
        if emb_a is None or emb_b is None:
            return 0
        return np.dot(emb_a, emb_b) / (
            np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-6
        )

    def _extract_embeddings_batch(self, xyxys, frame, reid_model):

        if hasattr(reid_model, "get_features"):
            try:
                embs = reid_model.get_features(xyxys, frame)
                embs = self._normalize_embeddings(embs)
                if embs is not None and len(embs) == len(xyxys):
                    return embs
            except Exception as e:
                print(f"[ReID BANK] get_features failed: "
                      f"{type(e).__name__}: {e}")

        try:
            embs = reid_model(xyxys, frame)
            embs = self._normalize_embeddings(embs)
            if embs is not None and len(embs) == len(xyxys):
                return embs
        except Exception:
            pass

        embs = []
        for box in xyxys:
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            emb = self._extract_embedding_single(crop, reid_model)
            if emb is None:
                return None
            embs.append(emb)
        return np.array(embs, dtype=np.float32)

    def _normalize_embeddings(self, embs):
        if embs is None:
            return None

        try:
            import torch
            if isinstance(embs, torch.Tensor):
                embs = embs.detach().cpu().numpy()
        except ImportError:
            pass

        embs = np.asarray(embs, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        if embs.size == 0:
            return None

        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms > 1e-8, norms, 1.0)
        return embs / norms

    def _extract_embedding_single(self, crop, reid_model):
        if crop is None or crop.size == 0:
            return None
        try:
            import torch
            import cv2

            img = cv2.resize(crop, (128, 256))
            img = img[:, :, ::-1].copy()
            img = img.astype(np.float32) / 255.0
            img = (img - np.array([0.485, 0.456, 0.406])) / \
                  np.array([0.229, 0.224, 0.225])
            img = np.transpose(img, (2, 0, 1))
            tensor = torch.FloatTensor(img).unsqueeze(0)

            if hasattr(reid_model, "device"):
                tensor = tensor.to(reid_model.device)

            with torch.no_grad():
                emb = reid_model(tensor)
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().numpy().flatten()

            norm = np.linalg.norm(emb)
            if norm < 1e-8:
                return None
            return (emb / norm).astype(np.float32)
        except Exception as e:
            print(f"[ReID BANK] single extract failed: "
                  f"{type(e).__name__}: {e}")
            return None

    def _color_histogram(self, crop):
        import cv2

        if crop.size == 0:
            return None

        h, w = crop.shape[:2]
        torso = crop[int(h*0.15):int(h*0.55),
                     int(w*0.15):int(w*0.85)]

        if torso.size == 0:
            torso = crop

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

        hist_h = cv2.calcHist([hsv], [0], None, [30], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

        hist = np.concatenate([hist_h, hist_s, hist_v])

        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm

        return hist

    def get_stats(self):
        total_emb = sum(len(v) for v in self.bank.values())

        emb_dim = None
        for entries in self.bank.values():
            if entries:
                emb_dim = len(entries[0][1])
                break
        return {
            "mode": self.mode,
            "embedding_dim": emb_dim,
            "tracks": len(self.bank),
            "total_embeddings": total_emb,
            "avg_per_track": total_emb / max(1, len(self.bank)),
            "extract_failures": self._extract_failures,
        }

class TrackletMerger:
    def __init__(self,
                 max_frame_gap=180,
                 max_spatial_dist=200,
                 min_tracklet_len=5,
                 embedding_thresh=0.4,
                 velocity_weight=0.5,
                 size_thresh=0.5,
                 merge_cost_thresh=0.7,
                 team_classifier=None,
                 reid_bank=None,
                 fps=30.0,
                 max_player_speed_ms=15.0,
                 world_dist_norm=20.0,
                 verbose=True):
        self.max_frame_gap = max_frame_gap
        self.max_spatial_dist = max_spatial_dist
        self.min_tracklet_len = min_tracklet_len
        self.embedding_thresh = embedding_thresh
        self.velocity_weight = velocity_weight
        self.size_thresh = size_thresh
        self.merge_cost_thresh = merge_cost_thresh
        self.team_classifier = team_classifier
        self.reid_bank = reid_bank
        self.fps = float(fps)
        self.max_player_speed_ms = float(max_player_speed_ms)
        self.world_dist_norm = float(world_dist_norm)
        self.verbose = bool(verbose)

        self.world_gate_rejections = 0

        self._failed_pairs_debug = []

    def merge(self, tracklets: dict) -> dict:
        valid = {tid: t for tid, t in tracklets.items()
                 if len(t.frames) >= self.min_tracklet_len}

        short = {tid: t for tid, t in tracklets.items()
                 if len(t.frames) < self.min_tracklet_len}

        print(f"[MERGE] Всего треков: {len(tracklets)}")
        print(f"[MERGE] Валидных (>={self.min_tracklet_len} кадров): {len(valid)}")
        print(f"[MERGE] Коротких (отброшены): {len(short)}")

        if self.team_classifier is not None:
            classified = sum(1 for tid in valid
                             if self.team_classifier.get_team(tid)[0] is not None)
            print(f"[MERGE] С командой: {classified}/{len(valid)}")

        sorted_ids = sorted(valid.keys(),
                            key=lambda tid: valid[tid].start_frame)

        id_map = {tid: tid for tid in tracklets}
        merge_count = 0
        blocked_by_team = 0

        merged_into = set()

        frames_of = {tid: set(t.frames) for tid, t in tracklets.items()}
        group_frames = {tid: set(fs) for tid, fs in frames_of.items()}
        overlap_rejections = 0

        N = len(sorted_ids)
        BIG = 1e6
        cost_mat = np.full((N, N), BIG, dtype=np.float64)
        for ii in range(N):
            ta = valid[sorted_ids[ii]]
            for jj in range(N):
                if ii == jj:
                    continue
                tb = valid[sorted_ids[jj]]
                gap = tb.start_frame - ta.end_frame
                if gap < 0 or gap > self.max_frame_gap:
                    continue

                if self.team_classifier is not None and \
                        not self.team_classifier.same_team(
                            sorted_ids[ii], sorted_ids[jj]):
                    if not (len(ta.frames) < 30 or len(tb.frames) < 30):
                        blocked_by_team += 1
                        continue
                if not self._world_gate(ta, tb, gap):
                    self.world_gate_rejections += 1
                    continue
                if frames_of[sorted_ids[ii]] & frames_of[sorted_ids[jj]]:
                    overlap_rejections += 1
                    continue
                c = self._compute_merge_cost(ta, tb, gap)
                if c is not None and c < self.merge_cost_thresh:
                    cost_mat[ii, jj] = c

        try:
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(cost_mat)
            pairs = list(zip(rows, cols))
        except ImportError:
            pairs = []
            used_r, used_c = set(), set()
            order = sorted(((cost_mat[r, c], r, c)
                            for r in range(N) for c in range(N)
                            if cost_mat[r, c] < BIG), key=lambda x: x[0])
            for _, r, c in order:
                if r in used_r or c in used_c:
                    continue
                used_r.add(r); used_c.add(c)
                pairs.append((r, c))
        links = [(sorted_ids[r], sorted_ids[c])
                 for r, c in pairs if cost_mat[r, c] < BIG]

        for a, b in links:
            ra = self._resolve_id(id_map, a)
            rb = self._resolve_id(id_map, b)
            if ra == rb:
                continue
            if group_frames[ra] & group_frames[rb]:
                overlap_rejections += 1
                continue
            id_map[rb] = ra
            group_frames[ra] |= group_frames[rb]
            merge_count += 1

        for tid_s, t_s in short.items():
            best_match = None
            best_cost = self.merge_cost_thresh

            for tid_v in sorted_ids:
                if tid_v in merged_into:
                    continue
                t_v = valid[tid_v]

                if group_frames[self._resolve_id(id_map, tid_v)] & frames_of[tid_s]:
                    overlap_rejections += 1
                    continue

                gap_after = t_s.start_frame - t_v.end_frame
                gap_before = t_v.start_frame - t_s.end_frame

                if 0 < gap_after <= self.max_frame_gap:
                    if not self._world_gate(t_v, t_s, gap_after):
                        self.world_gate_rejections += 1
                        continue
                    cost = self._compute_merge_cost(t_v, t_s, gap_after)
                    if cost is not None and cost < best_cost:
                        best_cost = cost
                        best_match = tid_v
                elif 0 < gap_before <= self.max_frame_gap:
                    if not self._world_gate(t_s, t_v, gap_before):
                        self.world_gate_rejections += 1
                        continue
                    cost = self._compute_merge_cost(t_s, t_v, gap_before)
                    if cost is not None and cost < best_cost:
                        best_cost = cost
                        best_match = tid_v

            if best_match is not None:
                root = self._resolve_id(id_map, best_match)
                id_map[tid_s] = root
                group_frames[root] |= frames_of[tid_s]
                merge_count += 1

        for tid in id_map:
            id_map[tid] = self._resolve_id(id_map, tid)

        reid_merges = 0
        if self.reid_bank is not None:
            reid_map = self.reid_bank.get_merge_map(
                tracklets,
                max_frame_gap=self.max_frame_gap * 2,
                team_classifier=self.team_classifier,
            )

            for new_id, old_id in reid_map.items():

                resolved_new = self._resolve_id(id_map, new_id)
                resolved_old = self._resolve_id(id_map, old_id)

                if resolved_new != resolved_old:

                    if group_frames[resolved_new] & group_frames[resolved_old]:
                        overlap_rejections += 1
                        continue

                    id_map[resolved_new] = resolved_old
                    group_frames[resolved_old] |= group_frames[resolved_new]
                    reid_merges += 1

            for tid in id_map:
                id_map[tid] = self._resolve_id(id_map, tid)

        if self.team_classifier is not None:
            self._inherit_teams(id_map, tracklets)

        unique_after = len(set(id_map.values()))
        print(f"[MERGE] Объединений: {merge_count}")
        if overlap_rejections > 0:
            print(f"[MERGE] Отклонено по пересечению кадров: "
                  f"{overlap_rejections}")
        if reid_merges > 0:
            print(f"[MERGE] Воскрешено через Re-ID: {reid_merges}")
        if blocked_by_team > 0:
            print(f"[MERGE] Заблокировано (разные команды): {blocked_by_team}")
        print(f"[MERGE] Уникальных ID после: {unique_after}")

        if self.team_classifier is not None:
            self._validate_teams(id_map)

        if self.verbose and self._failed_pairs_debug:
            self._print_failure_summary()

        return id_map

    def _print_failure_summary(self):
        from collections import Counter
        reasons = Counter(f["reason"] for f in self._failed_pairs_debug)
        print(f"\n[MERGE DEBUG] Причины отказа склейки "
              f"(сохранено {len(self._failed_pairs_debug)} пар):")
        for reason, count in reasons.most_common():
            print(f"  {reason}: {count}")

        emb_failures = [
            f for f in self._failed_pairs_debug
            if f["reason"] == "embedding_too_different"
            and "emb_cost" in f["details"]
        ]
        if emb_failures:

            emb_failures.sort(key=lambda f: f["details"]["emb_cost"])
            print(f"\n[MERGE DEBUG] Топ-5 пар close-to-pass по embedding "
                  f"(на пороге {self.embedding_thresh}):")
            for f in emb_failures[:5]:
                d = f["details"]
                print(f"  A{f['tid_a']:>3}({f['len_a']:>3}f) ↔ "
                      f"B{f['tid_b']:>3}({f['len_b']:>3}f) "
                      f"gap={f['frame_gap']:>3}f "
                      f"cos_sim={d.get('cos_sim', 0):.3f} "
                      f"emb_cost={d.get('emb_cost', 0):.3f}")

    def _inherit_teams(self, id_map, tracklets):

        groups = defaultdict(list)
        for old_id, new_id in id_map.items():
            groups[new_id].append(old_id)

        inherited = 0
        for new_id, member_ids in groups.items():
            if len(member_ids) <= 1:
                continue

            best_team = None
            best_len = 0
            for tid in member_ids:
                team_idx = self.team_classifier.track_team.get(tid)
                t = tracklets.get(tid)
                if team_idx is not None and t is not None:
                    if len(t.frames) > best_len:
                        best_len = len(t.frames)
                        best_team = team_idx

            if best_team is not None:
                for tid in member_ids:
                    if tid not in self.team_classifier.track_team:
                        self.team_classifier.track_team[tid] = best_team
                        inherited += 1

        if inherited > 0:
            print(f"[MERGE] Унаследовано команд: {inherited}")

    def _validate_teams(self, id_map):
        from team_classifier import TEAM_NAMES, TEAM_COLORS

        final_ids = set(id_map.values())
        team_counts = defaultdict(set)

        for tid in final_ids:
            team_idx = self.team_classifier.track_team.get(tid)
            if team_idx is not None:
                team_name = TEAM_NAMES[team_idx]
                team_counts[team_name].add(tid)

        print(f"\n[MERGE] Валидация по командам:")
        total_classified = 0
        for name in TEAM_NAMES:
            count = len(team_counts.get(name, set()))
            total_classified += count
            label = TEAM_COLORS[name]["label"]

            expected = {
                "team_1": (10, 11), "team_2": (10, 11),
                "goalkeeper_1": (1, 1), "goalkeeper_2": (1, 1),
                "referee": (1, 4),
            }
            exp_min, exp_max = expected.get(name, (0, 99))

            status = "OK" if exp_min <= count <= exp_max else "WARNING"
            if status == "WARNING":
                print(f"  [{status}] {label}: {count} "
                      f"(ожидалось {exp_min}-{exp_max})")
            else:
                print(f"  [{status}] {label}: {count}")

        unclassified = len(final_ids) - total_classified
        if unclassified > 0:
            print(f"  [INFO] Без команды: {unclassified}")

    def _world_gate(self, t_a, t_b, frame_gap) -> bool:
        last_a = t_a.last_world_position()
        first_b = t_b.first_world_position()
        if last_a is None or first_b is None:
            return True

        gap_seconds = max(1, frame_gap) / self.fps
        max_distance = self.max_player_speed_ms * gap_seconds

        dist_m = float(np.hypot(
            last_a[0] - first_b[0], last_a[1] - first_b[1]
        ))
        return dist_m <= max_distance

    def _record_failure(self, t_a, t_b, frame_gap, reason, details=None):
        if not self.verbose:
            return

        if (len(t_a.frames) < self.min_tracklet_len or
                len(t_b.frames) < self.min_tracklet_len):
            return
        self._failed_pairs_debug.append({
            "tid_a": t_a.track_id,
            "tid_b": t_b.track_id,
            "frame_gap": frame_gap,
            "len_a": len(t_a.frames),
            "len_b": len(t_b.frames),
            "reason": reason,
            "details": details or {},
        })

        if len(self._failed_pairs_debug) > 200:
            self._failed_pairs_debug = self._failed_pairs_debug[-100:]

    def _compute_merge_cost(self, t_a, t_b, frame_gap):
        costs = []
        weights = []

        a_end_w = t_a.last_world_position()
        b_start_w = t_b.first_world_position()
        a_end_vel_w = t_a.end_velocity_mps(fps=self.fps)

        use_world_spatial = (a_end_w is not None and b_start_w is not None)

        if use_world_spatial:

            gap_seconds = max(1, frame_gap) / self.fps

            if a_end_vel_w is not None:
                pred_x = a_end_w[0] + a_end_vel_w[0] * gap_seconds
                pred_y = a_end_w[1] + a_end_vel_w[1] * gap_seconds
            else:

                pred_x, pred_y = a_end_w[0], a_end_w[1]

            predict_dist_m = float(np.hypot(
                pred_x - b_start_w[0], pred_y - b_start_w[1]
            ))
            spatial_cost = min(1.0, predict_dist_m / self.world_dist_norm)
            costs.append(spatial_cost)
            weights.append(0.30)

            direct_dist_m = float(np.hypot(
                a_end_w[0] - b_start_w[0], a_end_w[1] - b_start_w[1]
            ))

            if direct_dist_m > self.world_dist_norm * 3.0:
                self._record_failure(t_a, t_b, frame_gap,
                                     reason="direct_world_too_far",
                                     details={"direct_m": direct_dist_m})
                return None
            direct_cost = min(1.0, direct_dist_m / (self.world_dist_norm * 1.5))
            costs.append(direct_cost)
            weights.append(0.10)
        else:

            predicted_pos = t_a.end_pos + t_a.end_velocity * frame_gap
            actual_pos = t_b.start_pos

            spatial_dist = np.linalg.norm(predicted_pos - actual_pos)
            if spatial_dist > self.max_spatial_dist:
                self._record_failure(t_a, t_b, frame_gap,
                                     reason="pred_pixel_too_far",
                                     details={"pred_px": float(spatial_dist)})
                return None

            spatial_cost = spatial_dist / self.max_spatial_dist
            costs.append(spatial_cost)
            weights.append(0.30)

            direct_dist = np.linalg.norm(t_a.end_pos - t_b.start_pos)
            if direct_dist > self.max_spatial_dist * 1.5:
                self._record_failure(t_a, t_b, frame_gap,
                                     reason="direct_pixel_too_far",
                                     details={"direct_px": float(direct_dist)})
                return None

            direct_cost = direct_dist / (self.max_spatial_dist * 1.5)
            costs.append(direct_cost)
            weights.append(0.10)

        size_a = t_a.avg_size
        size_b = t_b.avg_size
        if size_a[0] > 0 and size_b[0] > 0:
            size_diff = np.abs(size_a - size_b) / (size_a + 1e-6)
            if np.any(size_diff > self.size_thresh):
                self._record_failure(t_a, t_b, frame_gap,
                                     reason="size_too_different",
                                     details={"size_diff_max":
                                              float(np.max(size_diff))})
                return None
            size_cost = np.mean(size_diff) / self.size_thresh
            costs.append(size_cost)
            weights.append(0.10)

        if use_world_spatial:

            vel_a = a_end_vel_w
            vel_b = t_b.start_velocity_mps(fps=self.fps)
            if vel_a is not None and vel_b is not None and \
               np.linalg.norm(vel_a) > 0.5 and np.linalg.norm(vel_b) > 0.5:
                vel_cos = np.dot(vel_a, vel_b) / (
                    np.linalg.norm(vel_a) * np.linalg.norm(vel_b) + 1e-6)
                vel_cost = (1 - vel_cos) / 2
                costs.append(vel_cost)
                weights.append(0.10)
        else:
            vel_a = t_a.end_velocity
            vel_b = t_b.start_velocity
            if np.linalg.norm(vel_a) > 0.5 and np.linalg.norm(vel_b) > 0.5:
                vel_cos = np.dot(vel_a, vel_b) / (
                    np.linalg.norm(vel_a) * np.linalg.norm(vel_b) + 1e-6)
                vel_cost = (1 - vel_cos) / 2
                costs.append(vel_cost)
                weights.append(0.10)

        emb_a = t_a.mean_embedding
        emb_b = t_b.mean_embedding
        emb_cost_value = None
        if emb_a is not None and emb_b is not None:
            cos_sim = np.dot(emb_a, emb_b) / (
                np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-6)
            emb_cost = (1 - cos_sim) / 2
            emb_cost_value = float(emb_cost)
            if emb_cost > self.embedding_thresh:
                self._record_failure(t_a, t_b, frame_gap,
                                     reason="embedding_too_different",
                                     details={"emb_cost": emb_cost_value,
                                              "cos_sim": float(cos_sim)})
                return None
            costs.append(emb_cost)
            weights.append(0.20)

        gap_cost = frame_gap / self.max_frame_gap
        costs.append(gap_cost)
        weights.append(0.10)

        if self.team_classifier is not None:
            tid_a = t_a.track_id
            tid_b = t_b.track_id
            team_a = self.team_classifier.track_team.get(tid_a)
            team_b = self.team_classifier.track_team.get(tid_b)

            if team_a is not None and team_b is not None:
                if team_a == team_b:

                    costs.append(0.0)
                    weights.append(0.15)
                else:

                    SHORT_TEAM_THRESH = 30
                    a_short = len(t_a.frames) < SHORT_TEAM_THRESH
                    b_short = len(t_b.frames) < SHORT_TEAM_THRESH
                    if a_short or b_short:

                        costs.append(0.4)
                        weights.append(0.10)
                    else:

                        self._record_failure(t_a, t_b, frame_gap,
                                             reason="team_mismatch",
                                             details={"team_a": team_a,
                                                      "team_b": team_b})
                        return None
            elif team_a is not None or team_b is not None:

                costs.append(0.3)
                weights.append(0.05)

        weights = np.array(weights)
        weights /= weights.sum()
        total_cost = np.dot(costs, weights)

        return total_cost

    def _resolve_id(self, id_map, tid):
        visited = set()
        current = tid
        while id_map.get(current, current) != current:
            if current in visited:
                break
            visited.add(current)
            current = id_map[current]
        return current

def _gsi_boxes(track_data, id_map, eff_total):
    try:
        from tracking_refine import gsi_smooth
    except Exception:
        gsi_smooth = None

    seq = {}
    rep_old = {}
    for fn, dets in track_data.items():
        if eff_total is not None and fn > eff_total:
            continue
        for det in dets:
            old_id = int(det[4])
            nid = id_map.get(old_id, old_id)
            seq.setdefault(nid, {})[int(fn)] = [
                float(det[0]), float(det[1]), float(det[2]), float(det[3])]
            rep_old.setdefault(nid, old_id)

    out = {}
    for nid, fb in seq.items():
        fr = sorted(fb.keys())
        if gsi_smooth is None or len(fr) < 3:
            for f in fr:
                out.setdefault(f, []).append((nid, rep_old[nid], fb[f]))
            continue
        boxes = np.array([fb[f] for f in fr], dtype=np.float64)
        sm = gsi_smooth(fr, boxes, tau=10.0, noise=3.0, max_fill=20)
        obs = set(fr)
        for f, box in sm.items():

            b = fb[f] if f in obs else box
            out.setdefault(f, []).append((nid, rep_old[nid], b))
    return out

def apply_merge_to_video(input_video, output_video, track_data, id_map,
                          fps=30.0, team_classifier=None, end_frame=None):
    import cv2

    cap = cv2.VideoCapture(str(input_video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    eff_total = end_frame if (end_frame is not None and end_frame > 0) else total

    writer = cv2.VideoWriter(str(output_video),
                              cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (w, h))

    frame_num = 0
    trails = {}

    smoothed_boxes = _gsi_boxes(track_data, id_map, eff_total)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        if end_frame is not None and frame_num > end_frame:
            break

        dets_sm = smoothed_boxes.get(frame_num, [])

        for new_id, rep_old, box in dets_sm:
            x1, y1, x2, y2 = (int(box[0]), int(box[1]),
                              int(box[2]), int(box[3]))

            if team_classifier is not None:
                team_id, team_name = team_classifier.get_team(rep_old)
                if team_name is not None:
                    color = team_classifier.get_display_color(team_name)
                    team_label = team_classifier.get_label(team_name)
                else:
                    color = (200, 200, 200)
                    team_label = "?"
            else:
                np.random.seed(int(new_id) * 7)
                color = tuple(int(c) for c in np.random.randint(80, 255, 3))
                team_label = None

            cx, cy = (x1 + x2) // 2, y2
            if new_id not in trails:
                trails[new_id] = []
            trails[new_id].append((cx, cy))
            if len(trails[new_id]) > 50:
                trails[new_id] = trails[new_id][-50:]

            for i in range(1, len(trails[new_id])):
                alpha = i / len(trails[new_id])
                thickness = max(1, int(3 * alpha))
                cv2.line(frame, trails[new_id][i-1], trails[new_id][i],
                         color, thickness)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"ID {new_id}"
            if team_label:
                label += f" [{team_label}]"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                           0.6, 2)
            cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+6, y1), color, -1)

            brightness = sum(color) / 3
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
            cv2.putText(frame, label, (x1+3, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        active = len(dets_sm)
        cv2.putText(frame, f"On field: {active} | Unique: {len(trails)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_num}/{total} [MERGED]",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1)

        writer.write(frame)

        if frame_num % 3 == 0:
            yield (frame_num, eff_total)

        if frame_num % 500 == 0:
            pct = frame_num / total * 100 if total > 0 else 0
            print(f"  [MERGE VIDEO] Frame {frame_num}/{total} ({pct:.1f}%)")

    cap.release()
    writer.release()
    print(f"[MERGE] Видео сохранено: {output_video}")
    yield (eff_total, eff_total)
