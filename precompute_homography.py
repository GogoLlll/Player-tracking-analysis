from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

def main():
    p = argparse.ArgumentParser(
        description="Precompute homography via PnLCalib"
    )
    p.add_argument("video", help="Путь к видео")
    p.add_argument("--output", "-o", default="homography.npz",
                   help="Куда сохранить (default: homography.npz)")
    p.add_argument("--pnlcalib-repo", required=True,
                   help="Путь к репо PnLCalib")
    p.add_argument("--kp-weights", required=True,
                   help="SV_kp.pth")
    p.add_argument("--line-weights", required=True,
                   help="SV_lines.pth")
    p.add_argument("--keyframe-interval", type=int, default=30,
                   help="Считать гомографию каждые N кадров")
    p.add_argument("--device", default=None,
                   help="cuda|cpu (default: auto)")
    p.add_argument("--start", type=int, default=0,
                   help="С какого кадра начать")
    p.add_argument("--end", type=int, default=-1,
                   help="До какого кадра (-1 = до конца)")
    p.add_argument("--show-preview", action="store_true",
                   help="Показывать кадры в окне (медленно)")
    p.add_argument("--kp-threshold", type=float, default=0.3434,
                   help="Порог keypoint heatmap (PnLCalib default)")
    p.add_argument("--line-threshold", type=float, default=0.7867,
                   help="Порог line heatmap (PnLCalib default)")
    p.add_argument("--no-pnl-refine", action="store_true",
                   help="Отключить PnL refinement (быстрее, чуть менее точно)")

    args = p.parse_args()

    video_path = Path(args.video).resolve()
    output_path = Path(args.output).resolve()

    if not video_path.exists():
        print(f"[ERROR] Видео не найдено: {video_path}")
        sys.exit(1)

    print(f"[1/3] Инициализация PnLCalib...")
    print(f"  repo: {args.pnlcalib_repo}")
    print(f"  kp_weights: {args.kp_weights}")
    print(f"  line_weights: {args.line_weights}")

    try:
        from pnlcalib_estimator import PnLCalibHomography
    except ImportError as e:
        print(f"[ERROR] Не удалось импортировать pnlcalib_estimator: {e}")
        sys.exit(1)

    try:
        estimator = PnLCalibHomography(
            repo_path=args.pnlcalib_repo,
            kp_weights=args.kp_weights,
            line_weights=args.line_weights,
            device=args.device,
            kp_threshold=args.kp_threshold,
            line_threshold=args.line_threshold,
            pnl_refine=not args.no_pnl_refine,
        )
    except Exception as e:
        print(f"[ERROR] Не удалось создать PnLCalibHomography:")
        print(f"  {type(e).__name__}: {e}")
        sys.exit(1)

    print(f"\n[2/3] Обработка видео: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Не могу открыть {video_path}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    end_frame = total if args.end < 0 else min(args.end, total)

    print(f"  Размер: {w}x{h}, FPS: {fps:.1f}, всего кадров: {total}")
    print(f"  Диапазон: {args.start}..{end_frame}")
    print(f"  Keyframe interval: {args.keyframe_interval}")

    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    frames_list = []
    matrices_list = []
    confidences_list = []
    failed_frames = []

    frame_num = args.start - 1
    processed = 0
    last_print = time.time()
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        if frame_num >= end_frame:
            break

        if (frame_num - args.start) % args.keyframe_interval != 0:
            continue

        processed += 1
        t0 = time.time()
        result = estimator.estimate(frame, frame_num)
        t_inf = time.time() - t0

        if result is not None:
            frames_list.append(frame_num)
            matrices_list.append(result.matrix)
            confidences_list.append(result.confidence)
        else:
            failed_frames.append(frame_num)

        if args.show_preview:
            preview = frame.copy()
            status = "OK" if result is not None else "FAILED"
            color = (0, 255, 0) if result is not None else (0, 0, 255)
            cv2.putText(preview, f"f{frame_num} [{status}] {t_inf*1000:.0f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            preview = cv2.resize(preview, (1280, 720)) if w > 1280 else preview
            cv2.imshow("PnLCalib Preview", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Прерывание пользователем (Q)")
                break

        if time.time() - last_print > 1.0:
            elapsed = time.time() - t_start
            kf_processed = processed
            kf_total = (end_frame - args.start) // args.keyframe_interval
            eta = elapsed / max(1, kf_processed) * \
                  max(0, kf_total - kf_processed)
            n_ok = len(frames_list)
            print(f"  f{frame_num}/{end_frame} | "
                  f"keyframes: {kf_processed}/{kf_total} | "
                  f"OK: {n_ok}, failed: {len(failed_frames)} | "
                  f"last inf: {t_inf*1000:.0f}ms | "
                  f"ETA: {eta:.0f}s")
            last_print = time.time()

    cap.release()
    if args.show_preview:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\n  Обработано за {elapsed:.1f}s")
    print(f"  Успешных кадров: {len(frames_list)}")
    print(f"  Failed кадров: {len(failed_frames)}")

    coverage = (
        100.0 * len(frames_list) / max(1, len(frames_list) + len(failed_frames))
    )
    print(f"  Coverage: {coverage:.1f}%")

    if not frames_list:
        print(f"\n[ERROR] Ни одного успешного кадра. "
              f"Проверь что PnLCalib работает на твоём видео.")
        sys.exit(1)

    if coverage < 50:
        print(f"\n[WARN] Низкий coverage ({coverage:.1f}%). Возможные причины:")
        print(f"  - Видео не футбол / нестандартный ракурс")
        print(f"  - Веса PnLCalib плохо подходят к твоей камере")
        print(f"  - Попробуй понизить --kp-threshold (default 0.3434)")

    print(f"\n[3/3] Сохранение в {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(output_path),
        frames=np.asarray(frames_list, dtype=np.int64),
        matrices=np.asarray(matrices_list, dtype=np.float64),
        confidences=np.asarray(confidences_list, dtype=np.float32),
        method="pnlcalib",
        video_path=str(video_path),
        keyframe_interval=args.keyframe_interval,
        failed_frames=np.asarray(failed_frames, dtype=np.int64),
    )

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Сохранено: {output_path} ({size_mb:.2f} MB)")
    print(f"\n[OK] Для использования в основном пайплайне:")
    print(f"  HOMOGRAPHY_CACHE_FILE = \"{output_path}\"")

if __name__ == "__main__":
    main()
