from collections import defaultdict

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot import BotSort
from tracklet_merger import Tracklet, TrackletMerger, ReIDBank, apply_merge_to_video
from team_classifier import TeamClassifier, draw_team_labels, draw_team_stats
from field_homography import (
    HomographyTracker, CachedHomography, bbox_to_world_position,
)
from minimap import Minimap, overlay_minimap
from pose_and_fouls import PoseAndFoulsService, PoseFoulsConfig

INPUT_VIDEO = "test_video/test_5-1.mp4"
OUTPUT_VIDEO = "results/result_tracked.mp4"
MODEL_NAME = "models/yolo12m.pt"

CONFIDENCE = 0.25
IOU_THRESH = 0.5
MIN_BBOX_HEIGHT = 30

SHOW_PREVIEW = True
TRAIL_LENGTH = 50
BBOX_THICKNESS = 2
FONT_SCALE = 0.6

ENABLE_MERGE = True
MERGED_VIDEO = "results/result_tracked_MERGED.mp4"
MAX_FRAME_GAP = 180
MAX_SPATIAL_DIST = 200
MERGE_COST_THRESH = 0.7

MAX_PLAYER_SPEED_MS = 15.0

ENABLE_REID_BANK = True
REID_COLLECT_EVERY = 5
REID_MATCH_THRESH = 0.75

ENABLE_TEAMS = True

ENABLE_MINIMAP = True
HOMOGRAPHY_CACHE_FILE = "homography.npz"
HOMOGRAPHY_CACHE_INTERPOLATE = True
HOMOGRAPHY_SMOOTH_WINDOW = 3
MINIMAP_WIDTH = 300
MINIMAP_POSITION = "top-right"

ENABLE_POSE = True
POSE_MODEL = "models/yolo11l-pose.pt"
POSE_CONF = 0.25
POSE_IOU = 0.5
POSE_IMGSZ = 640

POSE_MATCH_IOU = 0.4
DRAW_SKELETONS = True
SKELETON_KP_CONF = 0.5

ENABLE_POSE_SMOOTHING = True
POSE_SMOOTH_ALPHA = 0.5
POSE_SMOOTH_MIN_CONF = 0.3

ENABLE_FOUL_DETECTION = True

FOUL_WINDOW_SIZE = 7
FOUL_MIN_HITS = 4
FOUL_COOLDOWN = 30

ENABLE_ML_CLASSIFIER = False
ML_MODEL_PATH = "models/foul_classifier.txt"
ML_REJECT_THRESHOLD = 0.3
ML_ENSEMBLE_ALPHA = 0.5

ENABLE_TRAINING_EXPORT = False
TRAINING_DATA_CSV = "data/training_fouls.csv"

ENABLE_FOUL_DEBUG = False

def create_tracker():
    return BotSort(
        Path("osnet_x1_0_msmt17.pt"),
        "cuda:0",
        False,

        track_high_thresh=0.3,
        track_low_thresh=0.1,
        new_track_thresh=0.55,
        match_thresh=0.82,

        track_buffer=1000,

        cmc_method="sof",

        proximity_thresh=0.5,
        appearance_thresh=0.25,
        with_reid=True,
    )

def extract_reid_model_from_tracker(tracker):

    candidates = [
        "model",
        "reid_model",
        "encoder",
        "appearance",
    ]

    for attr in candidates:
        obj = getattr(tracker, attr, None)
        if obj is None:
            continue

        if hasattr(obj, "get_features") or callable(obj):
            print(f"[INFO] Re-ID модель найдена в трекере: "
                  f"tracker.{attr} ({type(obj).__name__})")
            return obj

    print("[WARNING] Не удалось найти Re-ID модель в трекере. "
          "ReIDBank переключится на цветовую гистограмму "
          "(работать будет, но качество ниже).")
    return None

def get_color_for_id(track_id: int, team_classifier=None) -> tuple:
    if team_classifier is not None:
        team_id, team_name = team_classifier.get_team(track_id)
        if team_name is not None:
            return team_classifier.get_display_color(team_name)

    np.random.seed(int(track_id) * 7)
    return tuple(int(c) for c in np.random.randint(80, 255, size=3))

def draw_tracks(frame, tracked_objects, trails, team_classifier=None):
    for obj in tracked_objects:
        x1, y1, x2, y2 = map(int, obj[:4])
        tid = int(obj[4])
        conf = obj[5]
        color = get_color_for_id(tid, team_classifier)

        cx, cy = (x1 + x2) // 2, y2
        if tid not in trails:
            trails[tid] = []
        trails[tid].append((cx, cy))
        if len(trails[tid]) > TRAIL_LENGTH:
            trails[tid] = trails[tid][-TRAIL_LENGTH:]

        pts = trails[tid]
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            thickness = max(1, int(3 * alpha))
            cv2.line(frame, pts[i - 1], pts[i], color, thickness)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BBOX_THICKNESS)

        label = f"ID {tid} ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       FONT_SCALE, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1),
                      color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                    (255, 255, 255), 2)

    return frame

def init_homography_tracker(cache_file: str):
    if not cache_file or not Path(cache_file).exists():
        print(f"[HOMOGRAPHY] Не найден файл кеша: {cache_file}")
        print(f"  Сначала запусти precompute_homography.py "
              f"чтобы создать его.")
        return None

    try:
        cached = CachedHomography(
            cache_path=cache_file,
            interpolate=HOMOGRAPHY_CACHE_INTERPOLATE,
        )
    except Exception as e:
        print(f"[HOMOGRAPHY] Ошибка загрузки кеша: {e}")
        return None

    return HomographyTracker(
        estimator=cached,
        keyframe_interval=1,
        smooth_window=HOMOGRAPHY_SMOOTH_WINDOW,
    )

def main():
    input_path = Path(INPUT_VIDEO)
    if not input_path.exists():
        print(f"[ERROR] Видео не найдено: {input_path.resolve()}")
        return

    print(f"[INFO] Загрузка модели {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    pf_service = PoseAndFoulsService(PoseFoulsConfig(

        enable_pose=ENABLE_POSE,
        pose_model_name=POSE_MODEL,
        pose_conf=POSE_CONF,
        pose_iou=POSE_IOU,
        pose_match_iou=POSE_MATCH_IOU,

        enable_pose_smoothing=ENABLE_POSE_SMOOTHING,
        pose_smooth_alpha=POSE_SMOOTH_ALPHA,
        pose_smooth_min_conf=POSE_SMOOTH_MIN_CONF,

        draw_skeletons=DRAW_SKELETONS,

        enable_foul_detection=ENABLE_FOUL_DETECTION,
        foul_window_size=FOUL_WINDOW_SIZE,
        foul_min_hits=FOUL_MIN_HITS,
        foul_cooldown=FOUL_COOLDOWN,

        enable_ml_classifier=ENABLE_ML_CLASSIFIER,
        ml_model_path=ML_MODEL_PATH,
        ml_reject_threshold=ML_REJECT_THRESHOLD,
        ml_ensemble_alpha=ML_ENSEMBLE_ALPHA,

        enable_training_export=ENABLE_TRAINING_EXPORT,
        training_data_csv=TRAINING_DATA_CSV,

        enable_foul_debug=ENABLE_FOUL_DEBUG,
    ))

    print(f"[INFO] Инициализация трекера: BoT-SORT")
    tracker = create_tracker()

    team_classifier = None
    if ENABLE_TEAMS:
        team_classifier = TeamClassifier()
        print(f"[INFO] Классификация команд включена")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[ERROR] Не удалось открыть: {input_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = Path(OUTPUT_VIDEO)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    print(f"[INFO] Видео: {w}x{h} @ {fps:.1f} FPS, кадров: {total}")
    print(f"[INFO] Результат: {output_path.resolve()}")
    print(f"[INFO] Обработка...\n")

    trails = {}
    frame_num = 0
    max_simultaneous = 0

    tracklets = {}
    frame_track_data = {}

    world_diag = {
        "frames_total": 0,
        "frames_with_homo": 0,
        "world_pos_attempts": 0,
        "world_pos_valid": 0,
    }

    reid_bank = None
    if ENABLE_REID_BANK:

        reid_model = extract_reid_model_from_tracker(tracker)

        reid_bank = ReIDBank(
            max_embeddings_per_track=50,
            match_thresh=REID_MATCH_THRESH,
            reid_model=reid_model,
        )
        mode = "OSNet" if reid_model is not None else "color-histogram (fallback)"
        print(f"[INFO] Re-ID банк включён, режим: {mode} "
              f"(сбор каждые {REID_COLLECT_EVERY} кадров)")

    homography_tracker = None
    minimap = None
    if ENABLE_MINIMAP:
        homography_tracker = init_homography_tracker(
            cache_file=HOMOGRAPHY_CACHE_FILE,
        )
        if homography_tracker is not None:
            minimap = Minimap(
                map_width=MINIMAP_WIDTH,
                show_title=True,
                show_grass_pattern=True,
            )
            print(f"[INFO] Мини-карта: метод=cached (PnLCalib), "
                  f"размер={MINIMAP_WIDTH}px")
        else:
            print("[WARNING] Гомография не инициализирована, "
                  "мини-карта отключена")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        results = model.predict(
            frame,
            conf=CONFIDENCE,
            iou=IOU_THRESH,
            classes=[0],
            verbose=False,
        )

        dets = []
        if results[0].boxes is not None and len(results[0].boxes):
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                bh = xyxy[3] - xyxy[1]
                if bh >= MIN_BBOX_HEIGHT:
                    dets.append([*xyxy, conf, cls])

        if dets:
            dets = np.array(dets, dtype=np.float32)
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        tracked = tracker.update(dets, frame)

        h_frame, w_frame = frame.shape[:2]
        track_poses = pf_service.process_frame(
            frame, frame_num, tracked, frame_size=(w_frame, h_frame),
        )

        homo_result = None
        if homography_tracker is not None:
            homo_result = homography_tracker.get(frame, frame_num)

        world_diag["frames_total"] += 1
        if homo_result is not None:
            world_diag["frames_with_homo"] += 1

        if len(tracked) > 0:

            frame_track_data[frame_num] = []
            for obj in tracked:
                tid = int(obj[4])
                box = obj[:4].copy()
                conf_val = obj[5]

                frame_track_data[frame_num].append(
                    [*box, tid, conf_val]
                )

                if tid not in tracklets:
                    tracklets[tid] = Tracklet(track_id=tid)
                tracklets[tid].frames.append(frame_num)
                tracklets[tid].boxes.append(box.tolist())

                pose_for_track = track_poses.get(tid)
                tracklets[tid].poses.append(pose_for_track)

                world_pos = bbox_to_world_position(box, homo_result) \
                            if homo_result is not None else None
                tracklets[tid].world_positions.append(world_pos)
                world_diag["world_pos_attempts"] += 1
                if world_pos is not None:
                    world_diag["world_pos_valid"] += 1

            if reid_bank is not None and frame_num % REID_COLLECT_EVERY == 0:

                reid_bank.collect_from_frame(
                    frame, tracked, frame_num,
                    homography_result=homo_result,
                )

            team_results = None
            if team_classifier is not None:
                team_results = team_classifier.classify_batch(frame, tracked)

            frame = draw_tracks(frame, tracked, trails, team_classifier)

            if team_results is not None:
                frame = draw_team_labels(frame, tracked, team_results)
                frame = draw_team_stats(frame, team_results)

            if DRAW_SKELETONS and track_poses:
                for tid, p in track_poses.items():
                    if p is None:
                        continue

                    skeleton_color = None
                    if team_classifier is not None:
                        try:
                            _, team_name = team_classifier.get_team(tid)
                            if team_name is not None:
                                skeleton_color = team_classifier.get_display_color(
                                    team_name
                                )
                        except Exception:
                            skeleton_color = None
                    frame = draw_skeleton(
                        frame, p,
                        color=skeleton_color,
                        conf_thresh=SKELETON_KP_CONF,
                        use_part_colors=(skeleton_color is None),
                    )

            frame = pf_service.draw_foul_overlays(frame, track_poses)

            current_count = len(tracked)
            max_simultaneous = max(max_simultaneous, current_count)
            count_text = f"On field: {current_count} | Unique IDs: {len(trails)}"
        else:
            count_text = "On field: 0"

        cv2.putText(frame, count_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame {frame_num}/{total}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, f"Tracker: BoT-SORT", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        if homo_result is not None and minimap is not None and len(tracked) > 0:
            mini_img = minimap.render(tracked, homo_result, team_classifier)
            frame = overlay_minimap(
                frame, mini_img,
                position=MINIMAP_POSITION,
                margin=16,
            )

        frame = pf_service.draw_event_panel_overlay(frame)

        writer.write(frame)

        if SHOW_PREVIEW:
            preview = cv2.resize(frame, (1280, 720)) if w > 1280 else frame
            cv2.imshow("Football Tracker v2", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] Остановлено (q)")
                break

        if frame_num % 100 == 0:
            pct = frame_num / total * 100 if total > 0 else 0
            n = len(tracked) if len(tracked) > 0 else 0
            line = (f"  Frame {frame_num}/{total} ({pct:.1f}%) | "
                    f"Tracked: {n} | Unique IDs: {len(trails)}")

            if pf_service.has_pose \
                    and pf_service._pose_stats["frames_processed"] > 0:
                ps = pf_service._pose_stats
                avg_poses = ps["total_poses_found"] / ps["frames_processed"]
                avg_matched = ps["total_matched"] / ps["frames_processed"]
                line += (f" | Pose: avg {avg_poses:.1f} found, "
                         f"{avg_matched:.1f} matched/frame")
            print(line)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print(f"[DONE] Обработано кадров: {frame_num}")
    print(f"[DONE] Всего уникальных ID: {len(trails)}")
    print(f"[DONE] Макс. одновременно на поле: {max_simultaneous}")
    print(f"[DONE] Результат: {output_path.resolve()}")

    pf_service.finalize()

    if team_classifier is not None:
        print(f"\n{'='*50}")
        print(f"СТАТИСТИКА ПО КОМАНДАМ")
        print(f"{'='*50}")
        team_counts = defaultdict(list)
        for tid, team_idx in team_classifier.track_team.items():
            from team_classifier import TEAM_NAMES
            team_counts[TEAM_NAMES[team_idx]].append(tid)

        for team_name, tids in team_counts.items():
            print(f"  {team_name}: {len(tids)} игроков - "
                  f"IDs {sorted(tids)}")

    if reid_bank is not None:
        stats = reid_bank.get_stats()
        dim_str = f"{stats['embedding_dim']}-d" if stats['embedding_dim'] else "?"
        print(f"\n[ReID BANK] Режим: {stats['mode']} ({dim_str})")
        print(f"[ReID BANK] Треков: {stats['tracks']}, "
              f"эмбеддингов: {stats['total_embeddings']}, "
              f"среднее на трек: {stats['avg_per_track']:.1f}")
        if stats['extract_failures'] > 0:
            print(f"[ReID BANK] Ошибок извлечения: "
                  f"{stats['extract_failures']}")

    if homography_tracker is not None:
        h_stats = homography_tracker.get_stats()
        success_rate = (
            h_stats["keyframes_computed"] /
            max(1, h_stats["keyframes_computed"] + h_stats["keyframes_failed"])
        ) * 100
        print(f"\n[HOMOGRAPHY] Метод: cached (PnLCalib)")
        print(f"[HOMOGRAPHY] Ключевых кадров обсчитано: "
              f"{h_stats['keyframes_computed']} "
              f"(из них провалов: {h_stats['keyframes_failed']}, "
              f"успех: {success_rate:.1f}%)")
        print(f"[HOMOGRAPHY] Использовано экстраполяций: "
              f"{h_stats['extrapolated']}")
        if h_stats['no_homography'] > 0:
            print(f"[HOMOGRAPHY] Кадров без гомографии: "
                  f"{h_stats['no_homography']}")

    ft = world_diag["frames_total"]
    fh = world_diag["frames_with_homo"]
    wa = world_diag["world_pos_attempts"]
    wv = world_diag["world_pos_valid"]
    print(f"\n[WORLD] Кадров с гомографией: {fh}/{ft} "
          f"({100*fh/max(1,ft):.1f}%)")
    print(f"[WORLD] Мировых позиций: {wv}/{wa} "
          f"({100*wv/max(1,wa):.1f}%)")

    if pf_service.has_pose and len(tracklets) > 0:
        total_frames_count = 0
        total_with_pose = 0
        per_track_coverage = []
        for tid, t in tracklets.items():
            if not t.poses:
                continue
            n = len(t.poses)
            n_with = sum(1 for p in t.poses if p is not None)
            total_frames_count += n
            total_with_pose += n_with
            per_track_coverage.append((tid, n, n_with))

        if total_frames_count > 0:
            cov = 100.0 * total_with_pose / total_frames_count
            print(f"\n[POSE] Покрытие поз: "
                  f"{total_with_pose}/{total_frames_count} "
                  f"кадров-треков ({cov:.1f}%)")
            per_track_coverage.sort(key=lambda x: x[2] / max(1, x[1]))
            print(f"[POSE] Худшее покрытие у треков:")
            for tid, n, n_with in per_track_coverage[:5]:
                pct = 100.0 * n_with / max(1, n)
                print(f"        ID {tid}: {n_with}/{n} ({pct:.1f}%)")

    if ENABLE_MERGE and len(tracklets) > 0:
        print(f"\n{'='*50}")
        print(f"ПОСТ-ОБРАБОТКА ТРЕКОВ")
        print(f"{'='*50}")

        merger = TrackletMerger(
            max_frame_gap=MAX_FRAME_GAP,
            max_spatial_dist=MAX_SPATIAL_DIST,
            merge_cost_thresh=MERGE_COST_THRESH,
            team_classifier=team_classifier,
            reid_bank=reid_bank,
            fps=fps,
            max_player_speed_ms=MAX_PLAYER_SPEED_MS,
        )

        id_map = merger.merge(tracklets)

        if merger.world_gate_rejections > 0:
            print(f"[MERGE] Мировой гейт отверг "
                  f"{merger.world_gate_rejections} пар "
                  f"(физически невозможные)")

        unique_before = len(set(tracklets.keys()))
        unique_after = len(set(id_map.values()))
        print(f"\n[MERGE] ID до объединения: {unique_before}")
        print(f"[MERGE] ID после объединения: {unique_after}")
        print(f"[MERGE] Сокращение: {unique_before - unique_after} треков")

        merges = [(old, new) for old, new in id_map.items() if old != new]
        if merges:
            print(f"\n[MERGE] Объединения:")
            grouped = defaultdict(list)
            for old, new in merges:
                grouped[new].append(old)
            for new_id, old_ids in grouped.items():
                all_ids = sorted([new_id] + old_ids)
                durations = []
                for tid in all_ids:
                    if tid in tracklets:
                        t = tracklets[tid]
                        durations.append(
                            f"ID {tid} (frames {t.start_frame}-{t.end_frame})"
                        )
                print(f"  → ID {new_id}: {' + '.join(durations)}")

        merged_path = Path(MERGED_VIDEO)
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n[MERGE] Создание видео с объединёнными ID...")

        for _ in apply_merge_to_video(
            INPUT_VIDEO, str(merged_path),
            frame_track_data, id_map, fps,
            team_classifier=team_classifier,
        ):
            pass
        print(f"[MERGE] Готово: {merged_path.resolve()}")

if __name__ == "__main__":
    main()
