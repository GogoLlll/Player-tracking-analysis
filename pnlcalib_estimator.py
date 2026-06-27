from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from field_homography import FieldHomographyEstimator, HomographyResult

class PnLCalibHomography(FieldHomographyEstimator):

    def __init__(self,
                 repo_path: str,
                 kp_weights: str,
                 line_weights: str,
                 device: Optional[str] = None,
                 kp_threshold: float = 0.3434,
                 line_threshold: float = 0.7867,
                 pnl_refine: bool = True,
                 verbose: bool = True):
        self.repo_path = Path(repo_path).resolve()
        self.kp_weights = Path(kp_weights).resolve()
        self.line_weights = Path(line_weights).resolve()
        self.kp_threshold = float(kp_threshold)
        self.line_threshold = float(line_threshold)
        self.pnl_refine = bool(pnl_refine)
        self.verbose = verbose

        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"PnLCalib repo not found: {self.repo_path}\n"
                f"  git clone https://github.com/mguti97/PnLCalib.git"
            )
        if not self.kp_weights.exists():
            raise FileNotFoundError(
                f"Keypoints weights not found: {self.kp_weights}\n"
                f"  Скачай с https://github.com/mguti97/PnLCalib/releases"
            )
        if not self.line_weights.exists():
            raise FileNotFoundError(
                f"Line weights not found: {self.line_weights}"
            )

        repo_str = str(self.repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        try:
            import torch
            import torchvision.transforms as T
            import torchvision.transforms.functional as f
        except ImportError as e:
            raise ImportError(
                "PyTorch не установлен. Нужны torch + torchvision."
            ) from e

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._torch = torch
        self._tvF = f
        self._T = T

        try:
            from model.cls_hrnet import get_cls_net
            from model.cls_hrnet_l import get_cls_net as get_cls_net_l
            from utils.utils_calib import FramebyFrameCalib
            from utils.utils_heatmap import (
                get_keypoints_from_heatmap_batch_maxpool,
                get_keypoints_from_heatmap_batch_maxpool_l,
                complete_keypoints, coords_to_dict,
            )
            from PIL import Image
            import yaml
        except ImportError as e:
            raise ImportError(
                f"Не удалось импортировать модули PnLCalib.\n"
                f"  Подробно: {e}\n"
                f"  Проверьте путь к репозиторию PnLCalib ({self.repo_path}) "
                f"и установите зависимости проекта: pip install -r requirements.txt"
            ) from e

        self._get_cls_net = get_cls_net
        self._get_cls_net_l = get_cls_net_l
        self._FramebyFrameCalib = FramebyFrameCalib
        self._get_kp = get_keypoints_from_heatmap_batch_maxpool
        self._get_kp_l = get_keypoints_from_heatmap_batch_maxpool_l
        self._complete_keypoints = complete_keypoints
        self._coords_to_dict = coords_to_dict
        self._Image = Image

        kp_cfg_path = self.repo_path / "config" / "hrnetv2_w48.yaml"
        line_cfg_path = self.repo_path / "config" / "hrnetv2_w48_l.yaml"
        if not kp_cfg_path.exists():
            raise FileNotFoundError(f"Конфиг не найден: {kp_cfg_path}")
        if not line_cfg_path.exists():
            raise FileNotFoundError(f"Конфиг не найден: {line_cfg_path}")
        with open(kp_cfg_path) as fh:
            kp_cfg = yaml.safe_load(fh)
        with open(line_cfg_path) as fh:
            line_cfg = yaml.safe_load(fh)

        if self.verbose:
            print(f"[PnLCalib] Загрузка моделей на {self.device}...")
        try:
            loaded_state = torch.load(
                str(self.kp_weights), map_location=self.device
            )
            self.model = get_cls_net(kp_cfg)
            self.model.load_state_dict(loaded_state)
            self.model.to(self.device)
            self.model.eval()

            loaded_state_l = torch.load(
                str(self.line_weights), map_location=self.device
            )
            self.model_l = get_cls_net_l(line_cfg)
            self.model_l.load_state_dict(loaded_state_l)
            self.model_l.to(self.device)
            self.model_l.eval()
        except Exception as e:
            raise RuntimeError(
                f"Ошибка загрузки весов: {type(e).__name__}: {e}\n"
            ) from e

        self._transform_resize = T.Resize((540, 960))

        self._cam = None
        self._cam_iw = None
        self._cam_ih = None

        if self.verbose:
            print(f"[PnLCalib] Готов. "
                  f"kp_thresh={kp_threshold}, line_thresh={line_threshold}, "
                  f"pnl_refine={pnl_refine}")

    def estimate(self, frame: np.ndarray,
                 frame_num: int) -> Optional[HomographyResult]:
        try:
            return self._estimate_impl(frame, frame_num)
        except Exception as e:
            if self.verbose:
                print(f"[PnLCalib] frame {frame_num} failed: "
                      f"{type(e).__name__}: {e}")
            return None

    def _estimate_impl(self, frame: np.ndarray,
                        frame_num: int) -> Optional[HomographyResult]:
        torch = self._torch
        f = self._tvF

        H_in, W_in = frame.shape[:2]

        if (self._cam is None or self._cam_iw != W_in
                or self._cam_ih != H_in):
            self._cam = self._FramebyFrameCalib(
                iwidth=W_in, iheight=H_in, denormalize=True
            )
            self._cam_iw, self._cam_ih = W_in, H_in

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = self._Image.fromarray(frame_rgb)
        frame_tensor = f.to_tensor(frame_pil).float().unsqueeze(0)

        if frame_tensor.size()[-1] != 960:
            frame_tensor = self._transform_resize(frame_tensor)
        frame_tensor = frame_tensor.to(self.device)
        _, _, h, w = frame_tensor.size()

        with torch.no_grad():
            heatmaps = self.model(frame_tensor)
            heatmaps_l = self.model_l(frame_tensor)

        kp_coords = self._get_kp(heatmaps[:, :-1, :, :])
        line_coords = self._get_kp_l(heatmaps_l[:, :-1, :, :])
        kp_dict = self._coords_to_dict(kp_coords, threshold=self.kp_threshold)
        lines_dict = self._coords_to_dict(
            line_coords, threshold=self.line_threshold
        )
        kp_dict, lines_dict = self._complete_keypoints(
            kp_dict[0], lines_dict[0], w=w, h=h, normalize=True
        )

        self._cam.update(kp_dict, lines_dict)
        final_params = self._cam.heuristic_voting(
            refine_lines=self.pnl_refine
        )

        if final_params is None:
            return None

        P = self._projection_from_cam_params(final_params)

        H_img2world = self._extract_image_to_world_homography(P)
        if H_img2world is None:
            return None

        n_kp = len(kp_dict) if isinstance(kp_dict, dict) else 0
        n_lines = len(lines_dict) if isinstance(lines_dict, dict) else 0
        confidence = min(1.0, (n_kp + n_lines * 0.5) / 20.0)

        try:
            r = HomographyResult.from_image_to_world(
                H=H_img2world,
                frame_num=frame_num,
                confidence=confidence,
                method="pnlcalib",
            )
        except ValueError:
            return None

        if not r.is_valid():
            return None

        return r

    @staticmethod
    def _projection_from_cam_params(final_params_dict) -> np.ndarray:
        cam_params = final_params_dict["cam_params"]
        fx = cam_params["x_focal_length"]
        fy = cam_params["y_focal_length"]
        cxcy = np.array(cam_params["principal_point"])
        pos = np.array(cam_params["position_meters"])
        R = np.array(cam_params["rotation_matrix"])

        It = np.eye(4)[:-1]
        It[:, -1] = -pos
        Q = np.array([
            [fx, 0, cxcy[0]],
            [0, fy, cxcy[1]],
            [0,  0,       1],
        ])
        P = Q @ (R @ It)
        return P

    @staticmethod
    def _extract_image_to_world_homography(
        P: np.ndarray
    ) -> Optional[np.ndarray]:
        try:

            M = np.stack([P[:, 0], P[:, 1], P[:, 3]], axis=1)
            det = float(np.linalg.det(M))
            if abs(det) < 1e-9:
                return None
            H_img2world = np.linalg.inv(M)
            if abs(H_img2world[2, 2]) > 1e-12:
                H_img2world = H_img2world / H_img2world[2, 2]
            return H_img2world
        except (np.linalg.LinAlgError, ValueError):
            return None
