from __future__ import annotations

import csv
import json
import sys
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

import threading

from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTranslator, QLibraryInfo, QTimer,
)
from PyQt6.QtGui import QImage, QPixmap, QColor, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox, QFileDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QListWidget, QProgressBar, QMessageBox, QGridLayout, QSizePolicy, QColorDialog, QTreeWidget, QTreeWidgetItem, QTableWidget, QTableWidgetItem, QHeaderView, QSpinBox, QProgressDialog, QDoubleSpinBox, QScrollArea

from pipeline_engine import (
    RunConfig, PipelineEngine, FrameResult, precompute_homography_iter,
)

def bgr_to_pixmap(frame: np.ndarray, max_w: int, max_h: int) -> QPixmap:
    if frame is None:
        return QPixmap()
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = frame.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    if max_w > 0 and max_h > 0:
        pix = pix.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
    return pix

class ColorButton(QPushButton):

    colorChanged = pyqtSignal(str)

    def __init__(self, rgb=(255, 255, 255)):
        super().__init__()
        self._color = QColor(*rgb)
        self.setFixedSize(60, 24)
        self.clicked.connect(self._pick)
        self._refresh()

    def _pick(self):

        dlg = QColorDialog(self._color, self.window())
        dlg.setOption(
            QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
        dlg.setWindowTitle("Выбери цвет")

        en = {
            "тон": "Hue:", "оттенок": "Hue:",
            "нас": "Sat:", "насыщенность": "Sat:",
            "ярк": "Val:", "знач": "Val:", "яркость": "Val:",
            "красный": "Red:", "зелёный": "Green:", "зеленый": "Green:",
            "синий": "Blue:", "альфа-канал": "Alpha:",
        }
        for lbl in dlg.findChildren(QLabel):
            key = lbl.text().replace("&", "").replace(":", "").strip().lower()
            if key in en:
                lbl.setText(en[key])
        if dlg.exec() and dlg.currentColor().isValid():
            self._color = dlg.currentColor()
            self._refresh()
            self.colorChanged.emit(self._color.name())

    def set_rgb(self, rgb):
        self._color = QColor(*rgb)
        self._refresh()
        self.colorChanged.emit(self._color.name())

    def _refresh(self):
        self.setStyleSheet(
            f"background-color: {self._color.name()};"
            f" border:1px solid #555; border-radius:4px;")

    def hex(self):
        return self._color.name().upper()

    def bgr(self):
        c = self._color
        return (c.blue(), c.green(), c.red())

class Spinner(QWidget):

    def __init__(self, parent=None, diameter=72):
        super().__init__(parent)
        self._angle = 0
        self.setFixedSize(diameter, diameter)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def _tick(self):
        self._angle = (self._angle + 30) % 360
        self.update()

    def start(self):
        self._timer.start(90)
        self.show()
        self.raise_()

    def stop(self):
        self._timer.stop()
        self.hide()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.translate(self.width() / 2, self.height() / 2)
        p.rotate(self._angle)
        n = 12
        r = self.width() / 2
        for i in range(n):
            alpha = int(40 + 215 * (i + 1) / n)
            pen = QPen(QColor(108, 152, 255, alpha))
            pen.setWidth(4)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawLine(0, int(r * 0.45), 0, int(r * 0.78))
            p.rotate(360 / n)

class PipelineWorker(QThread):
    frameReady = pyqtSignal(object)
    errorOccurred = pyqtSignal(str)
    finishedRun = pyqtSignal()

    def __init__(self, cfg: RunConfig):
        super().__init__()
        self.cfg = cfg
        self.engine = PipelineEngine(cfg)

    def run(self):
        try:
            for result in self.engine.run():
                self.frameReady.emit(result)
        except Exception as e:
            self.errorOccurred.emit(
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
        finally:
            self.finishedRun.emit()

    def pause(self):
        self.engine.request_pause()

    def resume(self):
        self.engine.request_resume()

    def stop(self):
        self.engine.request_stop()

class HomographyWorker(QThread):
    progress = pyqtSignal(object)
    done = pyqtSignal(object)
    errorOccurred = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._stop = threading.Event()

    def run(self):
        try:
            for upd in precompute_homography_iter(
                    stop_event=self._stop, **self.params):
                phase = upd.get("phase")
                if phase == "error":
                    self.errorOccurred.emit(upd.get("message", "Ошибка"))
                    return
                if phase == "done":
                    self.done.emit(upd)
                    return
                self.progress.emit(upd)

            self.errorOccurred.emit("Прервано пользователем.")
        except Exception as e:
            self.errorOccurred.emit(
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")

    def stop(self):
        self._stop.set()

class DeviceProbe(QThread):
    ready = pyqtSignal(list)

    def run(self):
        items = []
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        name = torch.cuda.get_device_name(i)
                    except Exception:
                        name = f"GPU {i}"
                    items.append((f"cuda:{i}", name))
        except Exception:
            pass
        items.append(("cpu", "CPU"))
        self.ready.emit(items)

class SetupTab(QWidget):
    runRequested = pyqtSignal(object)

    COLOR_CLASSES = [
        ("team_1", "Команда 1", (0, 100, 255)),
        ("team_2", "Команда 2", (255, 255, 255)),
        ("goalkeeper_1", "Вратарь 1", (0, 160, 0)),
        ("goalkeeper_2", "Вратарь 2", (255, 140, 0)),
        ("referee", "Арбитр", (60, 60, 60)),
    ]

    def __init__(self):
        super().__init__()
        self._color_buttons = {}
        self._adv = {}
        self._build_ui()
        self._wire_dependencies()
        self._populate_models()
        self._probe_devices()

    def _build_ui(self):
        root = QVBoxLayout(self)

        _content = QWidget()
        content_layout = QVBoxLayout(_content)
        content_layout.setContentsMargins(0, 0, 0, 0)

        COMBO_W = 200
        BROWSE_W = 86

        gb_video = QGroupBox("Видео")
        v = QHBoxLayout(gb_video)
        self.le_video = QLineEdit()
        self.le_video.setPlaceholderText("Путь к видео матча (.mp4)")
        btn_video = QPushButton("Обзор…")
        btn_video.setFixedWidth(BROWSE_W)
        btn_video.clicked.connect(self._pick_video)
        v.addWidget(self.le_video)
        v.addWidget(btn_video)
        content_layout.addWidget(gb_video)

        main_row = QHBoxLayout()
        main_row.setContentsMargins(0, 0, 0, 0)
        left_col = QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)

        gb_models = QGroupBox("Модели")
        mg = QGridLayout(gb_models)
        mg.setColumnStretch(1, 1)
        self.cmb_det = QComboBox(); self.cmb_det.setEditable(True)
        btn_det = QPushButton("Обзор…")
        btn_det.clicked.connect(lambda: self._pick_model(self.cmb_det))
        mg.addWidget(QLabel("YOLO детектор:"), 0, 0)
        mg.addWidget(self.cmb_det, 0, 1)
        mg.addWidget(btn_det, 0, 2)

        self.cmb_pose = QComboBox(); self.cmb_pose.setEditable(True)
        self.btn_pose = QPushButton("Обзор…")
        self.btn_pose.clicked.connect(lambda: self._pick_model(self.cmb_pose))
        mg.addWidget(QLabel("YOLO-pose модель:"), 1, 0)
        mg.addWidget(self.cmb_pose, 1, 1)
        mg.addWidget(self.btn_pose, 1, 2)

        self.cmb_device = QComboBox()

        self.cmb_device.addItem("Видеокарта (GPU)", "cuda:0")
        self.cmb_device.addItem("CPU", "cpu")
        mg.addWidget(QLabel("Устройство:"), 2, 0)
        mg.addWidget(self.cmb_device, 2, 1, 1, 2)
        right_col.addWidget(gb_models)

        gb_colors = QGroupBox("Цвета")
        cwrap = QVBoxLayout(gb_colors)
        cform_w = QWidget()
        cv = QVBoxLayout(cform_w)
        cv.setSpacing(10)
        ac = Qt.AlignmentFlag.AlignVCenter
        for key, label, rgb in self.COLOR_CLASSES:
            btn = ColorButton(rgb)
            self._color_buttons[key] = btn
            hexlbl = QLabel(btn.hex())
            hexlbl.setStyleSheet("color:#aaa; font-family:monospace;")
            hexlbl.setFixedWidth(64)
            btn.colorChanged.connect(lambda h, lb=hexlbl: lb.setText(h.upper()))
            role = QLabel(label + ":")

            role.setFixedWidth(95)
            row = QHBoxLayout()
            row.addStretch(1)
            row.addWidget(role, 0, ac)
            row.addWidget(btn, 0, ac)
            row.addWidget(hexlbl, 0, ac)
            row.addStretch(1)
            cv.addLayout(row)
        cwrap.addStretch(1)
        cwrap.addWidget(cform_w)
        cwrap.addStretch(1)
        left_col.addWidget(gb_colors)

        gb_features = QGroupBox("Что анализировать")
        fcol2 = QVBoxLayout(gb_features)
        self.chk_pose = QCheckBox("Анализ поз"); self.chk_pose.setChecked(True)
        self.chk_fouls = QCheckBox("Детектор нарушений"); self.chk_fouls.setChecked(True)
        self.chk_teams = QCheckBox("Классификация команд"); self.chk_teams.setChecked(True)
        self.chk_minimap = QCheckBox("Мини-карта"); self.chk_minimap.setChecked(True)
        self.chk_merge = QCheckBox("Слияние треков (пост-обработка)"); self.chk_merge.setChecked(True)
        self.chk_reid = QCheckBox("Re-ID банк"); self.chk_reid.setChecked(True)
        for c in (self.chk_pose, self.chk_fouls, self.chk_teams,
                  self.chk_minimap, self.chk_merge, self.chk_reid):
            fcol2.addWidget(c)
        fcol2.addStretch(1)
        left_col.insertWidget(0, gb_features)

        gb_homo = QGroupBox("Мини-карта / гомография")
        hv = QVBoxLayout(gb_homo)
        hv.setSpacing(10)

        def _path_row(label_text, line_edit, browse_cb, label_w=110):
            line_edit.setMinimumWidth(140)
            lbl = QLabel(label_text)
            if label_w:
                lbl.setMinimumWidth(label_w)
            b = QPushButton("Обзор…")
            b.clicked.connect(browse_cb)
            h = QHBoxLayout()
            h.addWidget(lbl)
            h.addWidget(line_edit, 1)
            h.addWidget(b)
            return h

        self.le_homo = QLineEdit("homography.npz")
        hv.addLayout(_path_row("Файл:", self.le_homo, self._pick_homography))

        self.chk_burn = QCheckBox("Вжигать мини-карту в видео")
        self.chk_burn.setChecked(False)
        hv.addWidget(self.chk_burn)

        self.btn_toggle_gen = QPushButton("▸  Создать гомографию (если файла нет)")
        self.btn_toggle_gen.setCheckable(True)
        self.btn_toggle_gen.setToolTip("Нажми, чтобы раскрыть поля генерации")
        self.btn_toggle_gen.setStyleSheet(
            "QPushButton { text-align:left; padding:8px 12px;"
            " border:1px solid #3a3d45; border-radius:7px;"
            " background:#2a2c33; color:#cfd2d8; font-weight:600; }"
            "QPushButton:hover { background:#33363f; border-color:#4c8bf5; }"
            "QPushButton:checked { background:#2f3a52;"
            " border-color:#4c8bf5; color:#ffffff; }")
        self.btn_toggle_gen.clicked.connect(self._toggle_gen)
        hv.addWidget(self.btn_toggle_gen)

        self._gen_widget = QWidget()
        gv = QVBoxLayout(self._gen_widget)
        gv.setContentsMargins(0, 0, 0, 0)
        gv.setSpacing(10)
        self.le_pnl_repo = QLineEdit("PnLCalib")
        gv.addLayout(_path_row(
            "Репозиторий:", self.le_pnl_repo,
            lambda: self._browse_into(self.le_pnl_repo, directory=True)))

        self.le_pnl_kp = QLineEdit("PnLCalib/weights/SV_kp")
        self.le_pnl_kp.setMinimumWidth(120)
        b_kp = QPushButton("Обзор…")
        b_kp.clicked.connect(lambda: self._browse_into(self.le_pnl_kp))
        self.le_pnl_line = QLineEdit("PnLCalib/weights/SV_lines")
        self.le_pnl_line.setMinimumWidth(120)
        b_line = QPushButton("Обзор…")
        b_line.clicked.connect(lambda: self._browse_into(self.le_pnl_line))
        wrow = QHBoxLayout()
        wrow.addWidget(QLabel("Веса kp:"))
        wrow.addWidget(self.le_pnl_kp, 1)
        wrow.addWidget(b_kp)
        wrow.addSpacing(14)
        wrow.addWidget(QLabel("Веса линий:"))
        wrow.addWidget(self.le_pnl_line, 1)
        wrow.addWidget(b_line)
        gv.addLayout(wrow)

        self.btn_make_homo = QPushButton("⚙ Посчитать гомографию")
        self.btn_make_homo.setToolTip(
            "Посчитать гомографию для выбранного видео через PnLCalib")
        self.btn_make_homo.clicked.connect(self._make_homography)
        gv.addWidget(self.btn_make_homo)

        self._gen_widget.setVisible(False)
        hv.addWidget(self._gen_widget)

        self.le_homo.textChanged.connect(self._check_homography)
        right_col.addWidget(gb_homo)

        gb_adv = self._build_advanced_group()
        right_col.addWidget(gb_adv)

        gb_out = QGroupBox("Сохранение")
        ov = QVBoxLayout(gb_out)

        def _save_row(label, line_edit, browse_cb):
            line_edit.setFixedWidth(240)
            lbl = QLabel(label); lbl.setFixedWidth(120)
            b = QPushButton("Обзор…")
            b.clicked.connect(browse_cb)
            h = QHBoxLayout()
            h.addStretch(1)
            h.addWidget(lbl)
            h.addWidget(line_edit)
            h.addWidget(b)
            h.addStretch(1)
            ov.addLayout(h)

        self.le_out = QLineEdit("results/result_tracked.mp4")
        _save_row("Обычное видео:", self.le_out,
                  lambda: self._browse_save(self.le_out))
        self.le_merged = QLineEdit("results/result_merged.mp4")
        _save_row("Merged видео:", self.le_merged,
                  lambda: self._browse_save(self.le_merged))
        left_col.addWidget(gb_out)

        left_col.addStretch(1)
        right_col.addStretch(1)
        lw = QWidget(); lw.setLayout(left_col)
        rw = QWidget(); rw.setLayout(right_col)
        main_row.addWidget(lw, 2)
        main_row.addWidget(rw, 3)
        content_layout.addLayout(main_row, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        scroll.setWidget(_content)
        root.addWidget(scroll, 1)

        self.btn_run = QPushButton("▶  Запустить")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.clicked.connect(self._emit_run)
        root.addWidget(self.btn_run)

    @staticmethod
    def _wrap(layout) -> QWidget:
        w = QWidget(); w.setLayout(layout); return w

    def _wire_dependencies(self):
        self.chk_pose.toggled.connect(self._update_pose_dependents)
        self.chk_minimap.toggled.connect(self._check_homography)
        self._update_pose_dependents(self.chk_pose.isChecked())
        self._check_homography()

    def _update_pose_dependents(self, pose_on: bool):
        self.cmb_pose.setEnabled(pose_on)
        self.btn_pose.setEnabled(pose_on)
        self.chk_fouls.setEnabled(pose_on)
        if not pose_on:
            self.chk_fouls.setChecked(False)

    def _check_homography(self, *args):
        if not self.chk_minimap.isChecked():
            self.le_homo.setStyleSheet("")
            self.le_homo.setToolTip("Мини-карта выключена")
            return
        if Path(self.le_homo.text().strip()).exists():

            self.le_homo.setStyleSheet(
                "background:#143a14; color:#dfffdf;")
            self.le_homo.setToolTip("✓ Файл гомографии найден")
        else:

            self.le_homo.setStyleSheet(
                "background:#3a2a14; color:#ffe9cf;")
            self.le_homo.setToolTip(
                "⚠ Файл не найден - мини-карта не будет построена. "
                "Посчитай гомографию кнопкой ниже.")

    def _pick_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбери видео", "",
            "Видео (*.mp4 *.avi *.mov *.mkv);;Все файлы (*)")
        if path:
            self.le_video.setText(path)

    def _pick_model(self, combo: QComboBox):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбери модель", "models", "Веса YOLO (*.pt);;Все файлы (*)")
        if path:
            combo.setCurrentText(path)

    def _pick_homography(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбери файл гомографии", "", "NumPy (*.npz);;Все файлы (*)")
        if path:
            self.le_homo.setText(path)

    def _make_homography(self):
        video = self.le_video.text().strip()
        if not video or not Path(video).exists():
            QMessageBox.warning(self, "Нет видео",
                                "Сначала выбери видео матча.")
            return
        out = self.le_homo.text().strip() or "homography.npz"
        if Path(out).exists():
            ans = QMessageBox.question(
                self, "Файл существует",
                f"{out} уже есть. Пересчитать и перезаписать?")
            if ans != QMessageBox.StandardButton.Yes:
                return
        params = dict(
            video_path=video, output_path=out,
            pnlcalib_repo=self.le_pnl_repo.text().strip(),
            kp_weights=self.le_pnl_kp.text().strip(),
            line_weights=self.le_pnl_line.text().strip(),
            device=(self.cmb_device.currentData() or "cuda:0"),
            keyframe_interval=30,
        )
        self._homo_worker = HomographyWorker(params)
        self._homo_dlg = QProgressDialog(
            "Инициализация PnLCalib (загрузка моделей)…",
            "Отмена", 0, 100, self)
        self._homo_dlg.setWindowTitle("Создание гомографии")
        self._homo_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._homo_dlg.setAutoClose(False)
        self._homo_dlg.setAutoReset(False)
        self._homo_dlg.setMinimumDuration(0)
        self._homo_dlg.canceled.connect(self._homo_worker.stop)
        self._homo_worker.progress.connect(self._on_homo_progress)
        self._homo_worker.done.connect(self._on_homo_done)
        self._homo_worker.errorOccurred.connect(self._on_homo_error)
        self.btn_make_homo.setEnabled(False)
        self._homo_worker.start()
        self._homo_dlg.show()

    def _on_homo_progress(self, upd):
        total = max(1, upd.get("total", 1))
        self._homo_dlg.setMaximum(total)
        self._homo_dlg.setValue(upd.get("frame", 0))
        self._homo_dlg.setLabelText(
            f"Кадр {upd.get('frame', 0)}/{total}\n"
            f"Успешно: {upd.get('ok', 0)}, провалов: {upd.get('failed', 0)}\n"
            f"Покрытие: {upd.get('coverage', 0)}%")

    def _on_homo_done(self, upd):
        self._homo_dlg.close()
        self.btn_make_homo.setEnabled(True)
        QMessageBox.information(
            self, "Готово",
            f"Гомография сохранена:\n{upd.get('output')}\n\n"
            f"Ключевых кадров: {upd.get('ok')}, "
            f"покрытие {upd.get('coverage')}%")
        self._check_homography()

    def _on_homo_error(self, msg):
        if hasattr(self, "_homo_dlg") and self._homo_dlg is not None:
            self._homo_dlg.close()
        self.btn_make_homo.setEnabled(True)
        QMessageBox.critical(self, "Ошибка гомографии", msg)

    def _populate_models(self):
        models_dir = Path("models")
        pts = sorted(str(p) for p in models_dir.glob("*.pt")) \
            if models_dir.exists() else []
        det_default, pose_default = "models/yolo12m.pt", "models/yolo11l-pose.pt"
        self.cmb_det.addItems(pts or [det_default])
        self.cmb_pose.addItems(pts or [pose_default])
        self._set_if_present(self.cmb_det, det_default)
        self._set_if_present(self.cmb_pose, pose_default)

    @staticmethod
    def _set_if_present(combo: QComboBox, value: str):
        idx = combo.findText(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setCurrentText(value)

    def _probe_devices(self):
        self._dev_probe = DeviceProbe()
        self._dev_probe.ready.connect(self._fill_devices)
        self._dev_probe.start()

    def _fill_devices(self, items):
        cur = self.cmb_device.currentData()
        self.cmb_device.clear()
        for dev, disp in items:
            self.cmb_device.addItem(disp, dev)
        idx = self.cmb_device.findData(cur)
        if idx >= 0:
            self.cmb_device.setCurrentIndex(idx)

    def _browse_into(self, line_edit, directory=False):
        if directory:
            path = QFileDialog.getExistingDirectory(self, "Выбери папку")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Выбери файл")
        if path:
            line_edit.setText(path)

    def _toggle_gen(self):
        vis = self.btn_toggle_gen.isChecked()
        self._gen_widget.setVisible(vis)
        arrow = "▾ " if vis else "▸ "
        self.btn_toggle_gen.setText(
            arrow + "Создать гомографию (если файла нет)")

    def _browse_save(self, line_edit):
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить как", line_edit.text(),
            "Видео (*.mp4);;Все файлы (*)")
        if path:
            line_edit.setText(path)

    def _build_advanced_group(self) -> QGroupBox:
        gb = QGroupBox("Параметры пайплайна")
        grid = QGridLayout(gb)
        grid.setHorizontalSpacing(10)
        self._adv = {}
        groups = [[], [], []]

        def add_int(gi, key, label, lo, hi, default, step=1):
            sp = QSpinBox()
            sp.setRange(lo, hi); sp.setSingleStep(step); sp.setValue(default)
            sp.setMinimumWidth(96)
            self._adv[key] = sp
            groups[gi].append((label, sp))

        def add_float(gi, key, label, lo, hi, default, step=0.05, dec=2):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi); sp.setSingleStep(step); sp.setDecimals(dec)
            sp.setValue(default); sp.setMinimumWidth(96)
            self._adv[key] = sp
            groups[gi].append((label, sp))

        add_float(0, "confidence", "Уверенность", 0.0, 1.0, 0.25)
        add_float(0, "iou_thresh", "IoU (NMS)", 0.0, 1.0, 0.5)
        add_int(0, "min_bbox_height", "Мин. высота bbox", 0, 500, 30)
        add_int(0, "track_buffer", "Буфер трекера", 1, 5000, 1000, step=50)
        add_int(0, "max_frame_gap", "Макс. разрыв", 1, 2000, 180, step=10)

        add_int(1, "max_spatial_dist", "Макс. дист., px", 10, 2000, 200,
                step=10)
        add_float(1, "merge_cost_thresh", "Порог склейки", 0.0, 2.0, 0.7)
        add_float(1, "max_player_speed_ms", "Макс. скорость, м/с",
                  1.0, 30.0, 15.0, step=0.5, dec=1)
        add_float(1, "reid_match_thresh", "Re-ID порог", 0.0, 1.0, 0.75)
        add_int(1, "reid_collect_every", "Re-ID каждые N", 1, 60, 5)

        add_int(2, "homography_smooth_window", "Сглаж. гомографии", 1, 30, 3)
        add_int(2, "trail_length", "Длина трейла", 0, 300, 50, step=5)
        add_int(2, "bbox_thickness", "Толщина рамки", 1, 10, 2)
        add_float(2, "font_scale", "Масштаб шрифта", 0.2, 2.0, 0.6,
                  step=0.1, dec=1)

        for gi, items in enumerate(groups):
            base = gi * 3
            for ri, (label, sp) in enumerate(items):
                grid.addWidget(QLabel(label), ri, base)
                grid.addWidget(sp, ri, base + 1)
            grid.setColumnStretch(base + 2, 1)
        return gb

    def _emit_run(self):
        video = self.le_video.text().strip()
        if not video or not Path(video).exists():
            QMessageBox.warning(self, "Нет видео", "Укажи существующий файл видео.")
            return
        team_colors = {k: b.bgr() for k, b in self._color_buttons.items()}
        cfg = RunConfig(
            input_video=video,
            output_video=self.le_out.text().strip(),
            merged_video=self.le_merged.text().strip(),
            detection_model=self.cmb_det.currentText().strip(),
            pose_model=self.cmb_pose.currentText().strip(),
            device=(self.cmb_device.currentData() or "cuda:0"),
            enable_pose=self.chk_pose.isChecked(),
            enable_foul_detection=(self.chk_fouls.isChecked()
                                   and self.chk_pose.isChecked()),
            enable_teams=self.chk_teams.isChecked(),
            enable_minimap=self.chk_minimap.isChecked(),
            enable_merge=self.chk_merge.isChecked(),
            enable_reid_bank=self.chk_reid.isChecked(),
            homography_cache_file=self.le_homo.text().strip(),
            burn_minimap_into_video=self.chk_burn.isChecked(),
            team_display_colors=team_colors,

            confidence=self._adv["confidence"].value(),
            iou_thresh=self._adv["iou_thresh"].value(),
            min_bbox_height=self._adv["min_bbox_height"].value(),
            track_buffer=self._adv["track_buffer"].value(),
            max_frame_gap=self._adv["max_frame_gap"].value(),
            max_spatial_dist=self._adv["max_spatial_dist"].value(),
            merge_cost_thresh=self._adv["merge_cost_thresh"].value(),
            max_player_speed_ms=self._adv["max_player_speed_ms"].value(),
            reid_match_thresh=self._adv["reid_match_thresh"].value(),
            reid_collect_every=self._adv["reid_collect_every"].value(),
            homography_smooth_window=self._adv["homography_smooth_window"].value(),
            trail_length=self._adv["trail_length"].value(),
            bbox_thickness=self._adv["bbox_thickness"].value(),
            font_scale=self._adv["font_scale"].value(),
        )
        self.runRequested.emit(cfg)

class TracksTree(QTreeWidget):

    _FIELDS = ["Команда", "Скорость", "Позиция", "BBox", "Conf", "Поза"]

    def __init__(self):
        super().__init__()
        self.setHeaderHidden(True)
        self._items = {}

    def update_tracks(self, tracks: List[dict]):
        present = set()
        for tr in tracks:
            tid = tr["id"]
            present.add(tid)
            if tid not in self._items:
                top = QTreeWidgetItem(self)
                children = {}
                for fld in self._FIELDS:
                    children[fld] = QTreeWidgetItem(top)
                self._items[tid] = (top, children)
            top, ch = self._items[tid]
            top.setText(0, f"ID {tid} - {tr.get('team', '-')}")
            ch["Команда"].setText(0, f"Команда: {tr.get('team', '-')}")
            sp = tr.get("speed")
            ch["Скорость"].setText(
                0, f"Скорость: {sp} м/с" if sp is not None else "Скорость: -")
            wp = tr.get("world")
            ch["Позиция"].setText(
                0, f"Позиция: {wp[0]}, {wp[1]} м" if wp else "Позиция: -")
            ch["BBox"].setText(0, f"BBox: {tr.get('bbox')}")
            ch["Conf"].setText(0, f"Conf: {tr.get('conf')}")
            ch["Поза"].setText(
                0, f"Поза: {'да' if tr.get('has_pose') else 'нет'}")

        for tid in list(self._items):
            if tid not in present:
                top, _ = self._items.pop(tid)
                idx = self.indexOfTopLevelItem(top)
                if idx >= 0:
                    self.takeTopLevelItem(idx)

    def clear_tracks(self):
        self.clear()
        self._items.clear()

class LiveTab(QWidget):
    finishedWithEngine = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.worker: Optional[PipelineWorker] = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        top = QHBoxLayout()

        self.lbl_video = QLabel("Видео появится здесь после запуска")
        self.lbl_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_video.setMinimumSize(720, 405)
        self.lbl_video.setStyleSheet(
            "background:#1a1a1a; color:#888; border:1px solid #333;")
        self.lbl_video.setSizePolicy(QSizePolicy.Policy.Expanding,
                                     QSizePolicy.Policy.Expanding)

        _vlay = QVBoxLayout(self.lbl_video)
        self.spinner = Spinner(self.lbl_video)
        self.lbl_loading = QLabel("Загрузка видео…")
        self.lbl_loading.setStyleSheet("color:#aab; background:transparent;")
        self.lbl_loading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        _vlay.addStretch(1)
        _vlay.addWidget(self.spinner, 0, Qt.AlignmentFlag.AlignHCenter)
        _vlay.addWidget(self.lbl_loading, 0, Qt.AlignmentFlag.AlignHCenter)
        _vlay.addStretch(1)
        self.spinner.hide()
        self.lbl_loading.hide()
        top.addWidget(self.lbl_video, stretch=6)

        rightw = QWidget()
        rightw.setMaximumWidth(300)
        right = QVBoxLayout(rightw)
        right.setContentsMargins(0, 0, 0, 0)
        self.lbl_minimap = QLabel("Мини-карта")
        self.lbl_minimap.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_minimap.setFixedSize(280, 182)
        self.lbl_minimap.setStyleSheet(
            "background:#101810; color:#888; border:1px solid #333;")
        right.addWidget(self.lbl_minimap, alignment=Qt.AlignmentFlag.AlignHCenter)

        right.addWidget(QLabel("Активные треки (клик - подробности):"))
        self.tracks_tree = TracksTree()
        right.addWidget(self.tracks_tree, stretch=2)

        self.gb_fouls = QGroupBox("Активные нарушения")
        fl = QVBoxLayout(self.gb_fouls)
        self.lst_events = QListWidget()
        fl.addWidget(self.lst_events)
        right.addWidget(self.gb_fouls, stretch=1)

        top.addWidget(rightw, stretch=0)
        root.addLayout(top, stretch=1)

        self.lbl_stats = QLabel("-")
        self.lbl_stats.setStyleSheet("color:#ccc;")
        root.addWidget(self.lbl_stats)

        self.progress = QProgressBar()
        root.addWidget(self.progress)

        ctrl = QHBoxLayout()
        self.btn_pause = QPushButton("⏸ Пауза")
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_pause.setEnabled(False)
        self.btn_stop = QPushButton("⏹ Стоп")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_stop.setEnabled(False)
        ctrl.addWidget(self.btn_pause); ctrl.addWidget(self.btn_stop)
        ctrl.addStretch(1)
        root.addLayout(ctrl)

    def start(self, cfg: RunConfig):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Занято", "Обработка уже идёт.")
            return

        self.gb_fouls.setVisible(cfg.enable_pose and cfg.enable_foul_detection)
        self.lst_events.clear()
        self.tracks_tree.clear_tracks()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self._first_frame = False
        self.lbl_video.setText("")
        self.lbl_loading.show()
        self.spinner.start()
        self.worker = PipelineWorker(cfg)
        self.worker.frameReady.connect(self._on_frame)
        self.worker.errorOccurred.connect(self._on_error)
        self.worker.finishedRun.connect(self._on_finished)
        self.worker.start()
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_pause.setText("⏸ Пауза")

    def _toggle_pause(self):
        if self.worker is None:
            return
        if self.worker.engine.is_paused:
            self.worker.resume(); self.btn_pause.setText("⏸ Пауза")
        else:
            self.worker.pause(); self.btn_pause.setText("▶ Продолжить")

    def _stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.btn_stop.setEnabled(False)
            self.btn_pause.setEnabled(False)

    def _on_frame(self, result: FrameResult):
        if result.phase == "tracking":
            if result.display_frame is not None:
                if not getattr(self, "_first_frame", True):

                    self._first_frame = True
                    self.spinner.stop()
                    self.lbl_loading.hide()
                self.lbl_video.setPixmap(bgr_to_pixmap(
                    result.display_frame,
                    self.lbl_video.width(), self.lbl_video.height()))
            if result.minimap_img is not None:
                self.lbl_minimap.setPixmap(bgr_to_pixmap(
                    result.minimap_img,
                    self.lbl_minimap.width(), self.lbl_minimap.height()))

            self.tracks_tree.update_tracks(result.tracks)

            st = result.stats or {}
            total = st.get("total", 0) or result.total_frames
            if total > 0:
                self.progress.setMaximum(total)
                self.progress.setValue(st.get("frame_num", 0))
            if result.message:
                self.lbl_stats.setText(result.message)
            else:
                self.lbl_stats.setText(
                    f"Кадр {st.get('frame_num', 0)}/{total}  |  "
                    f"на поле: {st.get('tracked', 0)}  |  "
                    f"уникальных ID: {st.get('unique_ids', 0)}  |  "
                    f"скорость: {st.get('proc_fps', 0)} к/с")

            if self.gb_fouls.isVisible():
                self.lst_events.clear()
                for ev in result.events:
                    self.lst_events.addItem(
                        f"{ev['kind']}: A{ev['attacker_id']} → "
                        f"V{ev['victim_id']}  (score {ev['peak_score']})")

        elif result.phase == "postprocess":
            self.lbl_stats.setText(result.message or "Пост-обработка…")
            st = result.stats or {}
            mt = st.get("merge_total")
            if mt:

                self.progress.setRange(0, mt)
                self.progress.setValue(st.get("merge_frame", 0))
            else:
                self.progress.setRange(0, 0)

        elif result.phase == "done":
            self.progress.setRange(0, 1); self.progress.setValue(1)
            self.lbl_stats.setText(result.message or "Готово.")

    def _on_error(self, msg: str):
        self.spinner.stop()
        self.lbl_loading.hide()
        QMessageBox.critical(self, "Ошибка обработки", msg)

    def _on_finished(self):
        self.spinner.stop()
        self.lbl_loading.hide()
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 1); self.progress.setValue(1)
        if self.worker is not None:
            self.finishedWithEngine.emit(self.worker.engine)

class AnalyticsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.engine = None
        self._data = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        bar = QHBoxLayout()
        self.btn_refresh = QPushButton("↻ Обновить")
        self.btn_refresh.clicked.connect(self._refresh)
        self.btn_csv_players = QPushButton("Скачать игроков (CSV)")
        self.btn_csv_players.clicked.connect(self._save_players_csv)
        self.btn_csv_events = QPushButton("Скачать события (CSV)")
        self.btn_csv_events.clicked.connect(self._save_events_csv)
        self.btn_json = QPushButton("Скачать всё (JSON)")
        self.btn_json.clicked.connect(self._save_json)
        for b in (self.btn_csv_players, self.btn_csv_events, self.btn_json):
            b.setEnabled(False)
        bar.addWidget(self.btn_refresh)
        bar.addStretch(1)
        bar.addWidget(self.btn_csv_players)
        bar.addWidget(self.btn_csv_events)
        bar.addWidget(self.btn_json)
        root.addLayout(bar)

        self.lbl_quality = QLabel("Запусти обработку, затем нажми «Обновить».")
        self.lbl_quality.setWordWrap(True)
        root.addWidget(self.lbl_quality)

        root.addWidget(QLabel("Игроки:"))
        self.tbl_players = self._make_table(
            ["ID", "Класс", "Кадров", "Время, с", "Дистанция, м",
             "Ср. скор., м/с", "Макс. скор., м/с", "Поза, %"])
        root.addWidget(self.tbl_players, stretch=2)

        root.addWidget(QLabel("Команды:"))
        self.tbl_teams = self._make_table(
            ["Класс", "Игроков", "Сум. дистанция, м", "Ср. дистанция, м"])
        root.addWidget(self.tbl_teams, stretch=1)

        root.addWidget(QLabel("Нарушения:"))
        self.tbl_events = self._make_table(
            ["Тип", "Кадры", "Атакующий", "Жертва", "peak"])
        root.addWidget(self.tbl_events, stretch=1)

    @staticmethod
    def _make_table(headers) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        return t

    def load(self, engine):
        self.engine = engine
        self._refresh()

    def _refresh(self):
        if self.engine is None:
            QMessageBox.information(
                self, "Нет данных", "Сначала запусти обработку видео.")
            return
        try:
            self._data = self.engine.compute_analytics()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка аналитики", str(e))
            return
        d = self._data

        q = d.get("quality", {})
        self.lbl_quality.setText(
            f"FPS: {d.get('fps')}  |  кадров: {q.get('frames_total', 0)}  |  "
            f"покрытие гомографии: {q.get('homography_coverage_pct', 0)}%  |  "
            f"валидных мировых проекций: {q.get('world_valid_pct', 0)}%  |  "
            f"ID до merge: {q.get('ids_before_merge', 0)} → "
            f"после: {q.get('ids_after_merge', 0)}")

        self._fill(self.tbl_players, d.get("players", []),
                   ["id", "team", "frames", "time_s", "distance_m",
                    "avg_speed_ms", "max_speed_ms", "pose_coverage_pct"])
        self._fill(self.tbl_teams, d.get("teams", []),
                   ["team", "players", "total_distance_m", "avg_distance_m"])
        ev_rows = [{
            "kind": e["kind"],
            "frames": f"{e['start_frame']}–{e['end_frame']}",
            "attacker_id": e["attacker_id"],
            "victim_id": e["victim_id"],
            "peak_score": e["peak_score"],
        } for e in d.get("events", [])]
        self._fill(self.tbl_events, ev_rows,
                   ["kind", "frames", "attacker_id", "victim_id", "peak_score"])

        has = bool(d.get("players"))
        self.btn_csv_players.setEnabled(has)
        self.btn_csv_events.setEnabled(bool(d.get("events")))
        self.btn_json.setEnabled(True)

    @staticmethod
    def _fill(table: QTableWidget, rows: List[dict], keys: List[str]):
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, k in enumerate(keys):
                table.setItem(r, c, QTableWidgetItem(str(row.get(k, ""))))

    def _save_players_csv(self):
        if not self._data:
            return
        self._write_csv(self._data["players"],
                        ["id", "team", "frames", "time_s", "distance_m",
                         "avg_speed_ms", "max_speed_ms", "pose_coverage_pct"],
                        "players.csv")

    def _save_events_csv(self):
        if not self._data:
            return
        self._write_csv(self._data["events"],
                        ["kind", "start_frame", "end_frame", "attacker_id",
                         "victim_id", "peak_score", "avg_score"],
                        "events.csv")

    def _write_csv(self, rows, cols, default_name):
        if not rows:
            QMessageBox.information(self, "Пусто", "Нет данных для сохранения.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить CSV", default_name, "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for row in rows:
                    w.writerow({k: row.get(k, "") for k in cols})
            QMessageBox.information(self, "Сохранено", f"Файл: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def _save_json(self):
        if not self._data:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить JSON", "match_analytics.json", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "Сохранено", f"Файл: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Football Analysis")
        self.resize(1320, 840)

        self.tabs = QTabWidget()
        self.setup_tab = SetupTab()
        self.live_tab = LiveTab()
        self.analytics_tab = AnalyticsTab()
        self.tabs.addTab(self.setup_tab, "Настройка")
        self.tabs.addTab(self.live_tab, "Просмотр")
        self.tabs.addTab(self.analytics_tab, "Аналитика матча")
        self.setCentralWidget(self.tabs)

        self.setup_tab.runRequested.connect(self._on_run)
        self.live_tab.finishedWithEngine.connect(self._on_finished)

    def _on_run(self, cfg: RunConfig):
        self.tabs.setCurrentWidget(self.live_tab)
        self.live_tab.start(cfg)

    def _on_finished(self, engine):
        self.analytics_tab.load(engine)

    def closeEvent(self, event):
        w = self.live_tab.worker
        if w is not None and w.isRunning():
            w.stop()
            w.wait(3000)
        event.accept()

APP_QSS = """
QWidget { background-color: #1f2127; color: #e6e6e6; font-size: 13px; }
QGroupBox {
    border: 1px solid #34363d; border-radius: 10px;
    margin-top: 12px; padding: 10px 8px 8px 8px; font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #9aa0a6;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #2a2c33; border: 1px solid #3a3d45; border-radius: 7px;
    padding: 4px 7px; min-height: 20px;
    selection-background-color: #4c8bf5; selection-color: #ffffff;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #4c8bf5;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background-color: #2a2c33; border: 1px solid #3a3d45;
    selection-background-color: #4c8bf5; outline: none;
}
QPushButton {
    background-color: #2f323a; border: 1px solid #3a3d45; border-radius: 7px;
    padding: 5px 10px; min-height: 18px;
}
QPushButton:hover { background-color: #3a3e48; }
QPushButton:pressed { background-color: #4c8bf5; color: #ffffff; }
QPushButton:disabled { color: #6b6e76; background-color: #26282e; }
QCheckBox { spacing: 7px; }
QCheckBox::indicator {
    width: 16px; height: 16px; border: 1px solid #4a4d55;
    border-radius: 4px; background: #2a2c33;
}
QCheckBox::indicator:checked {
    background: #4c8bf5; border: 1px solid #4c8bf5; image: url(__CHECK__);
}
QTabWidget::pane { border: 1px solid #34363d; border-radius: 8px; top: -1px; }
QTabBar::tab {
    background: #26282e; color: #b8bcc4; padding: 9px 18px; margin-right: 2px;
    border-top-left-radius: 8px; border-top-right-radius: 8px;
}
QTabBar::tab:selected { background: #4c8bf5; color: #ffffff; }
QTabBar::tab:hover:!selected { background: #32353d; }
QProgressBar {
    border: 1px solid #34363d; border-radius: 7px; text-align: center;
    background: #26282e; height: 18px;
}
QProgressBar::chunk { background: #4c8bf5; border-radius: 6px; }
QTreeWidget, QListWidget, QTableWidget {
    background: #1b1d22; border: 1px solid #34363d; border-radius: 8px;
    outline: none;
}
QHeaderView::section {
    background: #26282e; padding: 5px; border: none;
    border-right: 1px solid #34363d;
}
QScrollBar:vertical { background: transparent; width: 11px; margin: 2px; }
QScrollBar::handle:vertical {
    background: #3a3d45; border-radius: 5px; min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: #4a4d55; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: transparent; height: 11px; margin: 2px; }
QScrollBar::handle:horizontal {
    background: #3a3d45; border-radius: 5px; min-width: 24px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QToolTip {
    background: #2a2c33; color: #e6e6e6; border: 1px solid #4c8bf5;
    border-radius: 5px; padding: 4px 6px;
}
QDialog, QColorDialog, QMessageBox { background-color: #1f2127; }
"""

def main():
    if cv2 is None:
        print("OpenCV (cv2) не установлен: pip install opencv-python")
    app = QApplication(sys.argv)

    _translator = QTranslator()
    _tr_path = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    if _translator.load("qtbase_ru", _tr_path):
        app.installTranslator(_translator)
        app._ru_translator = _translator

    check_url = (Path(__file__).resolve().parent / "check.svg").as_posix()
    app.setStyleSheet(APP_QSS.replace("__CHECK__", check_url))
    win = MainWindow()

    for b in win.findChildren(QPushButton):
        b.setCursor(Qt.CursorShape.PointingHandCursor)
    win.showMaximized()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
